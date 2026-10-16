"""A fake AWS, small enough to read, strict enough to catch a misspelled request.

Every request the pipeline sends is checked against botocore's own copy of the AWS API definition (the
same check boto3 runs before it sends anything), so a wrong parameter name or type fails HERE, not in a
live run. Nothing here touches the network.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path

import botocore.session
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

import mlops

ACCOUNT = "123456789012"
REGION = "ap-southeast-2"
BUCKET = f"finrisk-ml-{ACCOUNT}-{REGION}-an"
GROUP = "finrisk-fraud-model"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/finrisk-sagemaker-exec"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/finrisk-train@sha256:{'ab' * 32}"
APP_URL = "https://abc123.lambda-url.ap-southeast-2.on.aws"
ENV = {"ML_BUCKET": BUCKET, "MODEL_PACKAGE_GROUP": GROUP, "SAGEMAKER_ROLE_ARN": ROLE, "AWS_REGION": REGION}
NOW = dt.datetime(2026, 10, 16, 3, 0, 0, tzinfo=dt.timezone.utc)

_SESSION = botocore.session.get_session()


def check_shape(service: str, operation: str, params: dict) -> None:
    """Raise if AWS would reject the request's SHAPE (names, types, required fields)."""
    model = _SESSION.get_service_model(service).operation_model(operation)
    validate_parameters(params, model.input_shape)


def client_error(code: str, operation: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakeSTS:
    def get_caller_identity(self):
        return {"Account": ACCOUNT}


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str, str]] = []

    def upload_file(self, Filename, Bucket, Key):                       # noqa: N803 - boto3's own names
        self.uploads.append((Filename, Bucket, Key))
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()

    def list_objects_v2(self, Bucket, Prefix, Delimiter, ContinuationToken=None):   # noqa: N803
        check_shape("s3", "ListObjectsV2", {"Bucket": Bucket, "Prefix": Prefix, "Delimiter": Delimiter})
        folders = sorted({k[len(Prefix):].split(Delimiter)[0] for (b, k) in self.objects
                          if b == Bucket and k.startswith(Prefix) and Delimiter in k[len(Prefix):]})
        return {"CommonPrefixes": [{"Prefix": f"{Prefix}{f}/"} for f in folders]}

    def head_object(self, Bucket, Key):                                 # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise client_error("404", "HeadObject")
        return {}

    def get_object(self, Bucket, Key):                                  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise client_error("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


class FakeSSM:
    def __init__(self):
        self.params: dict[str, str] = {}
        self.puts: list[dict] = []
        self.denied: set[str] = set()                                   # names whose read is AccessDenied

    def get_parameter(self, Name):                                      # noqa: N803
        check_shape("ssm", "GetParameter", {"Name": Name})
        if Name in self.denied:
            raise client_error("AccessDeniedException", "GetParameter")
        if Name not in self.params:
            raise client_error("ParameterNotFound", "GetParameter")
        return {"Parameter": {"Name": Name, "Value": self.params[Name]}}

    def put_parameter(self, **kw):
        check_shape("ssm", "PutParameter", kw)
        self.puts.append(kw)
        self.params[kw["Name"]] = kw["Value"]
        return {"Version": len(self.puts)}


class FakeSageMaker:
    """Just enough of SageMaker: one scripted training job and a registry that numbers versions."""

    def __init__(self, s3: FakeS3, statuses=None, failure=None, pr_auc=0.8631):
        self.s3, self.failure, self.pr_auc = s3, failure, pr_auc
        self.statuses = list(statuses or [("InProgress", "Starting"), ("InProgress", "Downloading"),
                                          ("InProgress", "Training"), ("InProgress", "Uploading"), ("Completed", "Completed")])
        self.jobs: dict[str, dict] = {}
        self.packages: dict[int, dict] = {}
        self.calls: list[tuple[str, dict]] = []
        self._polls = 0

    # -- training
    def create_training_job(self, **kw):
        check_shape("sagemaker", "CreateTrainingJob", kw)
        self.calls.append(("create_training_job", kw))
        self.jobs[kw["TrainingJobName"]] = kw
        return {"TrainingJobArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/{kw['TrainingJobName']}"}

    def describe_training_job(self, TrainingJobName):                   # noqa: N803
        check_shape("sagemaker", "DescribeTrainingJob", {"TrainingJobName": TrainingJobName})
        request = self.jobs[TrainingJobName]
        status, secondary = self.statuses[min(self._polls, len(self.statuses) - 1)]
        self._polls += 1
        if self.failure and status == "Completed":
            status, secondary = "Failed", "Failed"
        desc = {
            "TrainingJobName": TrainingJobName, "TrainingJobStatus": status, "SecondaryStatus": secondary,
            "AlgorithmSpecification": request["AlgorithmSpecification"], "InputDataConfig": request["InputDataConfig"],
            "Environment": request["Environment"],
        }
        if status == "Completed":
            desc.update({
                "ModelArtifacts": {"S3ModelArtifacts": f"{request['OutputDataConfig']['S3OutputPath']}{TrainingJobName}/output/model.tar.gz"},
                "TrainingTimeInSeconds": 187, "BillableTimeInSeconds": 187,
                "FinalMetricDataList": [{"MetricName": "pr_auc", "Value": self.pr_auc, "Timestamp": dt.datetime(2026, 10, 14)},
                                        {"MetricName": "recall", "Value": 0.8889, "Timestamp": dt.datetime(2026, 10, 14)}],
            })
        if status == "Failed":
            desc["FailureReason"] = self.failure
        return desc

    # -- registry
    def create_model_package(self, **kw):
        check_shape("sagemaker", "CreateModelPackage", kw)
        self.calls.append(("create_model_package", kw))
        version = len(self.packages) + 1
        arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package/{kw['ModelPackageGroupName']}/{version}"
        self.packages[version] = {
            "ModelPackageArn": arn, "ModelPackageGroupName": kw["ModelPackageGroupName"], "ModelPackageVersion": version,
            "InferenceSpecification": kw["InferenceSpecification"], "ModelMetrics": kw["ModelMetrics"],
            "ModelApprovalStatus": kw["ModelApprovalStatus"], "CustomerMetadataProperties": kw["CustomerMetadataProperties"],
            "CreationTime": dt.datetime(2026, 10, 14, 3, version), "ModelPackageDescription": kw["ModelPackageDescription"],
        }
        return {"ModelPackageArn": arn}

    def _find(self, arn):
        version = int(arn.rsplit("/", 1)[1])
        if version not in self.packages:
            raise client_error("ValidationException", "DescribeModelPackage")
        return self.packages[version]

    def describe_model_package(self, ModelPackageName):                 # noqa: N803
        check_shape("sagemaker", "DescribeModelPackage", {"ModelPackageName": ModelPackageName})
        return dict(self._find(ModelPackageName))

    def update_model_package(self, **kw):
        check_shape("sagemaker", "UpdateModelPackage", kw)
        self.calls.append(("update_model_package", kw))
        package = self._find(kw["ModelPackageArn"])
        package["ModelApprovalStatus"] = kw["ModelApprovalStatus"]
        if "ApprovalDescription" in kw:
            package["ApprovalDescription"] = kw["ApprovalDescription"]
        return {"ModelPackageArn": kw["ModelPackageArn"]}

    def list_model_packages(self, **kw):
        check_shape("sagemaker", "ListModelPackages", kw)
        newest_first = sorted(self.packages.values(), key=lambda p: -p["ModelPackageVersion"])
        return {"ModelPackageSummaryList": [{k: p[k] for k in ("ModelPackageArn", "ModelPackageVersion", "ModelApprovalStatus", "CreationTime")}
                                            for p in newest_first]}


class FakeApp:
    """The running app's /ready, scripted: a list of (status, body) answers, the last one repeats."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.urls: list[str] = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        answer = self.answers[min(len(self.urls) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return status, body if isinstance(body, str) else json.dumps(body)


def ready(version, **extra):
    """What /ready says (the part that matters here): which model version is loaded."""
    return (200, {"ready": True, "model": {"version": version, "source": "s3", "loaded_at": "2026-10-16T03:00:20+00:00"}, **extra})


class World:
    """One fake AWS plus a way to run the CLI against it and read what it printed."""

    def __init__(self, capsys, **sm_options):
        self.capsys = capsys
        self.s3 = FakeS3()
        self.ssm = FakeSSM()
        self.sm = FakeSageMaker(self.s3, **sm_options)
        self.clients = {"sagemaker": self.sm, "s3": self.s3, "sts": FakeSTS(), "ssm": self.ssm}
        self.clock_now = 0.0
        self.docker_runs: list[list[str]] = []
        self.app = FakeApp([ready(1)])

    def run(self, *argv: str, env: dict | None = None, app: FakeApp | None = None, docker=None) -> tuple[int, str, str]:
        def clock():                       # time only moves when the pipeline sleeps
            return self.clock_now

        def sleep(seconds):
            self.clock_now += seconds

        def fake_docker(command):
            self.docker_runs.append(command)
            return docker(command) if docker else 0

        code = mlops.main(list(argv), clients=self.clients, env={**ENV, **(env or {})}, sleep=sleep,
                          clock=clock, http=app or self.app, run=fake_docker, now=lambda: NOW)
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    # helpers that set up the state a later command needs
    def put_data(self, stamp="20261014-030000"):
        self.s3.objects[(BUCKET, f"data/{stamp}/train.parquet")] = b"parquet"

    def put_metrics(self, job, **values):
        body = {"pr_auc": self.sm.pr_auc, "roc_auc": 0.99, "threshold": 0.1125, "rows": 5000,
                "precision": 0.36, "recall": 0.89} | values
        self.s3.objects[(BUCKET, f"metrics/{job}/metrics.json")] = json.dumps(body).encode()

    def trained_job(self, job="finrisk-train-20261014-031500"):
        """Run `train` to completion and return the job name."""
        self.put_data()
        code, out, err = self.run("train", "--image", IMAGE, "--job-name", job, "--data", f"s3://{BUCKET}/data/20261014-030000/")
        assert code == 0, err
        return job

    def registered_version(self, env=None):
        job = self.trained_job(job=f"finrisk-train-{len(self.sm.packages) + 1:02d}")
        self.put_metrics(job)
        code, out, err = self.run("register", "--training-job", job, env=env)
        assert code == 0, err
        return int(out.strip().splitlines()[-1])

    def decide(self, version, decision="approve"):
        code, out, err = self.run("approve", "--version", str(version), "--decision", decision)
        assert code == 0, err

    def set_pointer(self, env_name, version, model_url="s3://x/model.tar.gz"):
        self.ssm.params[f"/finrisk/{env_name}/model"] = json.dumps(
            {"version": version, "model_data_url": model_url, "status": "Approved", "promoted_at": "2026-10-01T00:00:00+00:00", "run": "local"})

    def set_app_url(self, env_name):
        self.ssm.params[f"/finrisk/{env_name}/app_url"] = APP_URL


def last_line(text: str) -> str:
    return text.strip().splitlines()[-1]

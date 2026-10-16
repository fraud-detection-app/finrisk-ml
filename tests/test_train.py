"""pipeline/mlops.py train: the CreateTrainingJob request, the metric regexes, waiting, failing."""

from __future__ import annotations

import re

import pytest
from fakes import ACCOUNT, BUCKET, GROUP, IMAGE, ROLE, World, check_shape, last_line

import mlops
import sagemaker_train

DATA = f"s3://{BUCKET}/data/20261014-030000/"


def request(**kw):
    cfg = mlops.Config(ml_bucket=BUCKET, group=GROUP, role_arn=ROLE)
    return mlops.build_training_request(cfg, mlops.load_train_config(mlops.DEFAULT_CONFIG), IMAGE, "finrisk-train-1", DATA, **kw)


def test_training_request_has_the_shape_the_contract_promises():
    r = request(git_sha="a" * 40)
    check_shape("sagemaker", "CreateTrainingJob", r)                 # valid against the real AWS API definition

    algorithm = r["AlgorithmSpecification"]
    assert algorithm["TrainingImage"] == IMAGE and "@sha256:" in algorithm["TrainingImage"]
    assert algorithm["TrainingInputMode"] == "File"
    assert algorithm["ContainerEntrypoint"] == ["python", "/opt/ml/code/sagemaker_train.py"]    # absolute: no WORKDIR needed
    assert [m["Name"] for m in algorithm["MetricDefinitions"]] == ["pr_auc", "roc_auc", "threshold", "precision", "recall", "rows"]

    assert r["InputDataConfig"] == [{
        "ChannelName": "train",
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": DATA, "S3DataDistributionType": "FullyReplicated"}}}]
    assert r["OutputDataConfig"] == {"S3OutputPath": f"s3://{BUCKET}/models/"}
    assert r["ResourceConfig"] == {"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 30}
    assert r["StoppingCondition"] == {"MaxRuntimeInSeconds": 1800}
    assert r["RoleArn"] == ROLE
    assert r["HyperParameters"] == {"model_type": "optimized"}
    assert r["Environment"]["METRICS_S3_URI"] == f"s3://{BUCKET}/metrics/finrisk-train-1/metrics.json"
    assert r["Environment"]["ML_GIT_SHA"] == "a" * 40
    assert {t["Key"] for t in r["Tags"]} >= {"Project", "ManagedBy", "MlGitSha"}


def test_the_model_type_flag_overrides_the_config():
    r = request(model_type="baseline")
    assert r["HyperParameters"] == {"model_type": "baseline"} and r["Environment"]["MODEL_TYPE"] == "baseline"


def test_the_metric_regexes_match_exactly_the_lines_the_trainer_prints():
    metrics = {"pr_auc": 0.863103, "roc_auc": 0.991753, "threshold": 0.112528, "precision": 0.363636, "recall": 0.888889, "rows": 5000}
    log = "training data : x\nmodel type    : optimized\n" + "\n".join(sagemaker_train.metric_lines(metrics)) + "\n"
    for definition in mlops.metric_definitions():
        found = re.findall(definition["Regex"], log)
        assert len(found) == 1, definition
        assert float(found[0]) == pytest.approx(metrics[definition["Name"]])
    assert [d["Name"] for d in mlops.metric_definitions()] == list(sagemaker_train.METRIC_NAMES)


def test_the_entrypoint_in_the_dockerfile_is_the_one_in_the_request():
    dockerfile = (mlops.REPO_ROOT / "train" / "Dockerfile").read_text()
    assert 'ENTRYPOINT ["python", "/opt/ml/code/sagemaker_train.py"]' in dockerfile
    assert "COPY train/sagemaker_train.py /opt/ml/code/sagemaker_train.py" in dockerfile
    assert mlops.ENTRYPOINT == ["python", "/opt/ml/code/sagemaker_train.py"]


def test_train_waits_prints_each_status_change_billable_seconds_and_ends_with_the_job_name(world):
    world.put_data("20261013-010000")
    world.put_data("20261014-030000")                                # the newest folder wins when --data is not given
    code, out, err = world.run("train", "--image", IMAGE, "--job-name", "finrisk-train-20261014-031500",
                               env={"GITHUB_SHA": "b" * 40})
    assert code == 0, err
    sent = world.sm.calls[0][1]
    assert sent["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == DATA
    assert sent["Environment"]["ML_GIT_SHA"] == "b" * 40
    steps = re.findall(r"\]\s+(\w+)\s+(\w+)", out)                    # every status change is printed once, in order
    assert steps == [("InProgress", "Starting"), ("InProgress", "Downloading"), ("InProgress", "Training"),
                     ("InProgress", "Uploading"), ("Completed", "Completed")]
    assert "billable 187 s" in out and "pr_auc" in out and "0.8631" in out
    assert last_line(out) == "finrisk-train-20261014-031500"


def test_train_uses_the_given_data_uri(world):
    code, out, err = world.run("train", "--image", IMAGE, "--job-name", "j", "--data", "s3://other/data/x/")
    assert code == 0, err
    assert world.sm.calls[0][1]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == "s3://other/data/x/"


def test_train_prints_every_aws_call_with_account_ids_masked(world):
    world.put_data()
    code, out, err = world.run("train", "--image", IMAGE, "--job-name", "job-x")
    assert code == 0
    assert "AWS  sagemaker.create_training_job" in out and "AWS  s3.list_objects_v2" in out
    assert out.count("sagemaker.describe_training_job") == 1         # polling is shown once, not 5 times
    assert ACCOUNT not in "\n".join(out.splitlines()[:-1]) and "<account-id>" in out    # only the last line is exact


def test_train_refuses_an_image_tag(world):
    code, out, err = world.run("train", "--image", f"{ACCOUNT}.dkr.ecr.ap-southeast-2.amazonaws.com/finrisk-train:sha-abc")
    assert code == 1 and "digest" in err
    assert not world.sm.calls                                        # nothing was created


def test_train_with_no_data_says_to_prepare_first(world):
    code, out, err = world.run("train", "--image", IMAGE)
    assert code == 1 and "prepare-data" in err and not world.sm.calls


def test_a_failed_training_job_exits_1_with_the_aws_reason(capsys):
    world = World(capsys, failure="AlgorithmError: ExecuteUserScriptError")
    code, out, err = world.run("train", "--image", IMAGE, "--job-name", "job-bad", "--data", DATA)
    assert code == 1
    assert "Failed" in err and "AlgorithmError" in err and "/aws/sagemaker/TrainingJobs" in err
    assert last_line(out) != "job-bad"                               # no "success" value on failure


def test_train_can_skip_waiting(world):
    code, out, err = world.run("train", "--image", IMAGE, "--job-name", "job-nw", "--data", DATA, "--no-wait")
    assert code == 0 and last_line(out) == "job-nw" and "describe_training_job" not in out


def test_names_fall_back_to_ssm_then_to_the_convention(world):
    """No ML_BUCKET / role in the environment: SSM /finrisk/shared/* first, the naming convention if SSM says no."""
    blank = {"ML_BUCKET": "", "SAGEMAKER_ROLE_ARN": ""}
    world.ssm.params["/finrisk/shared/ml_bucket"] = "bucket-from-ssm"
    world.ssm.params["/finrisk/shared/sagemaker_role_arn"] = ROLE
    code, out, err = world.run("train", "--image", IMAGE, "--job-name", "j1", "--data", DATA, env=blank)
    assert code == 0, err
    assert world.sm.calls[-1][1]["OutputDataConfig"]["S3OutputPath"] == "s3://bucket-from-ssm/models/"
    assert "taken from SSM /finrisk/shared/ml_bucket" in out

    other = World(world.capsys)
    other.ssm.denied = {"/finrisk/shared/ml_bucket", "/finrisk/shared/sagemaker_role_arn"}     # a role that may not read SSM
    code, out, err = other.run("train", "--image", IMAGE, "--job-name", "j2", "--data", DATA, env=blank)
    assert code == 0, err
    sent = other.sm.calls[-1][1]
    assert sent["OutputDataConfig"]["S3OutputPath"] == f"s3://{BUCKET}/models/" and sent["RoleArn"] == ROLE
    assert "naming convention" in out

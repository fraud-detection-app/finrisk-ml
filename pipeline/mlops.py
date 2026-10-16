"""The MLOps pipeline as one small command-line tool: data -> train -> gate -> register -> approve -> promote.

    prepare-data  make the gold feature table inside the training image and upload it to S3
    train         a SageMaker training job runs the training image on that data and we wait for it
    evaluate      THE GATE: is the candidate good enough to be registered at all? (exit 3 if not)
    register      add the candidate to the Model Registry as a numbered version, PendingManualApproval
    approve       a person says yes (or no) to a version
    promote       point one environment at a version, then wait until the running app says it loaded it
    list / show   look at the registry

Each step is its own sub-command because each happens at a different time and a person sits in the
middle (approve). The model is trained ONCE and the same numbered version is promoted dev -> sit ->
pre-prod -> prod: nothing is retrained per environment, exactly like the app image.

    python pipeline/mlops.py prepare-data --image <repo>@sha256:...        # -> s3://<ml>/data/<time>/
    python pipeline/mlops.py train --image <repo>@sha256:... --data <uri>  # -> the training job name
    python pipeline/mlops.py evaluate --training-job <name>                # the gate; exit 3 = rejected
    python pipeline/mlops.py register --training-job <name>                # -> the new version number
    python pipeline/mlops.py approve --version 3 --decision approve --note "looks good"
    python pipeline/mlops.py promote --env dev --version 3
    python pipeline/mlops.py list
    python pipeline/mlops.py show --version 3

Names come from the environment, then from SSM Parameter Store (/finrisk/shared/*), then from the
naming convention. Flags override all of them:
    ML_BUCKET  SAGEMAKER_ROLE_ARN  MODEL_PACKAGE_GROUP (default finrisk-fraud-model)  AWS_REGION (ap-southeast-2)

What you will see: every AWS call this tool makes is printed first -- the operation and its main
parameters, account ids masked as <account-id>. Nothing happens behind your back.

The LAST line printed on success is the value the next step needs: the data folder (prepare-data), the
training job name (train, evaluate), the version number (register), the new status (approve), the
version (promote). Pipelines use `... | tail -n 1`; add --quiet to print nothing else.

Exit codes: 0 ok · 1 something failed (message on stderr) · 2 the model version cannot be used here
(not found, Rejected, or not Approved for pre-prod / prod) · 3 the gate said no.

Needs only boto3 and pyyaml -- no pandas or scikit-learn -- so a CI runner can use it. The training
itself (train/sagemaker_train.py) runs inside the training image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "train"))
from sagemaker_train import METRIC_NAMES  # noqa: E402 - one list of metric names, shared with the trainer

DEFAULT_REGION = "ap-southeast-2"
DEFAULT_GROUP = "finrisk-fraud-model"
DEFAULT_CONFIG = REPO_ROOT / "config" / "train.yaml"
ENVIRONMENTS = ("dev", "sit", "pre-prod", "prod")
NEEDS_APPROVAL = ("pre-prod", "prod")           # dev and sit may run a version nobody approved yet
# An absolute path inside the training image, so SageMaker does not depend on the image's WORKDIR.
ENTRYPOINT = ["python", "/opt/ml/code/sagemaker_train.py"]
DIGEST_IMAGE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
TERMINAL = {"Completed", "Failed", "Stopped"}

EXIT_OK, EXIT_FAILED, EXIT_UNUSABLE, EXIT_GATE = 0, 1, 2, 3


class MlopsError(Exception):
    """Something to tell the user, with the exit code to use (1 = failed, 2 = model unusable, 3 = gate)."""

    def __init__(self, message: str, code: int = EXIT_FAILED):
        super().__init__(message)
        self.code = code


def mask(text: str) -> str:
    """Hide 12-digit account ids before anything is printed on a shared screen or into a log."""
    return ACCOUNT_ID.sub("<account-id>", str(text))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(fmt: str) -> str:
    return utc_now().strftime(fmt)


def error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}          # botocore's ClientError carries one
    return response.get("Error", {}).get("Code", "")


# ---------------------------------------------------------------------------- configuration


@dataclass
class TrainConfig:
    """config/train.yaml: what the training job runs on, and what the gate demands."""

    hyperparameters: dict[str, str] = field(default_factory=lambda: {"model_type": "optimized"})
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    volume_gb: int = 30
    max_runtime_seconds: int = 1800
    demo_rows: int = 5000
    gate_metric: str = "pr_auc"
    floor: float = 0.70
    tolerance: float = 0.02
    champion_env: str = "prod"

    @property
    def model_type(self) -> str:
        return self.hyperparameters.get("model_type", "optimized")


def load_train_config(path: Path) -> TrainConfig:
    import yaml

    if not Path(path).is_file():
        raise MlopsError(f"config file not found: {path}")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    inst, data, gate = raw.get("instance") or {}, raw.get("data") or {}, raw.get("gate") or {}
    default = TrainConfig()
    hyper = {str(k): str(v) for k, v in (raw.get("hyperparameters") or default.hyperparameters).items()}
    return TrainConfig(
        hyperparameters=hyper,
        instance_type=str(inst.get("type", default.instance_type)),
        instance_count=int(inst.get("count", default.instance_count)),
        volume_gb=int(inst.get("volume_gb", default.volume_gb)),
        max_runtime_seconds=int(inst.get("max_runtime_seconds", default.max_runtime_seconds)),
        demo_rows=int(data.get("demo_rows", default.demo_rows)),
        gate_metric=str(gate.get("metric", default.gate_metric)),
        floor=float(gate.get("floor", default.floor)),
        tolerance=float(gate.get("tolerance", default.tolerance)),
        champion_env=str(gate.get("champion_env", default.champion_env)),
    )


@dataclass
class Config:
    ml_bucket: str = ""
    group: str = DEFAULT_GROUP
    role_arn: str = ""
    region: str = DEFAULT_REGION

    def need(self, *names: str) -> None:
        env_names = {"ml_bucket": "ML_BUCKET", "group": "MODEL_PACKAGE_GROUP", "role_arn": "SAGEMAKER_ROLE_ARN"}
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            how = ", ".join(f"{env_names[n]} (--{n.replace('_', '-')})" for n in missing)
            raise MlopsError(f"not set: {how}.")


def load_config(args: argparse.Namespace, env: dict[str, str]) -> Config:
    return Config(
        ml_bucket=args.ml_bucket or env.get("ML_BUCKET", ""),
        group=args.group or env.get("MODEL_PACKAGE_GROUP", "") or DEFAULT_GROUP,
        role_arn=args.role_arn or env.get("SAGEMAKER_ROLE_ARN", ""),
        region=args.region or env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or DEFAULT_REGION,
    )


# ---------------------------------------------------------------------------- showing every AWS call


class Traced:
    """Wraps a boto3 client: prints each call (operation + main parameters) and then makes it.

    ``quiet_after_first`` names polling calls: shown once, then silent, so a 5-minute wait does not
    print 20 identical blocks."""

    def __init__(self, client: Any, service: str, out: Callable[[str], None], quiet_after_first: tuple[str, ...] = ()):
        self._client, self._service, self._out = client, service, out
        self._quiet, self._seen = set(quiet_after_first), set()

    def __getattr__(self, operation: str) -> Callable[..., Any]:
        call = getattr(self._client, operation)

        def traced(**params: Any) -> Any:
            if operation not in self._quiet or operation not in self._seen:
                self._out(mask(f"  AWS  {self._service}.{operation}"))
                for key, value in params.items():
                    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), default=str)
                    text = text if len(text) <= 240 else text[:237] + "..."
                    self._out(mask(f"         {key:<26} {text}"))
            self._seen.add(operation)
            return call(**params)

        return traced


def make_clients(region: str) -> dict[str, Any]:
    import boto3

    return {name: boto3.client(name, region_name=region) for name in ("sagemaker", "s3", "sts", "ssm")}


def http_get(url: str, timeout: float) -> tuple[int, str]:
    """GET a URL; return (status, body). An error status (503) is an answer, not an exception."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) address: {url!r}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:      # noqa: S310 - scheme checked above
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def run_command(command: list[str]) -> int:
    """Run a command, letting its output flow to our own (a docker run prints its progress)."""
    return subprocess.run(command, check=False).returncode


@dataclass
class Context:
    cfg: Config
    tcfg: TrainConfig
    sm: Any
    s3: Any
    sts: Any
    ssm: Any
    out: Callable[[str], None]
    sleep: Callable[[float], None]
    clock: Callable[[], float]
    env: dict[str, str]
    http: Callable[[str, float], tuple[int, str]] = http_get
    run: Callable[[list[str]], int] = run_command
    now: Callable[[], datetime] = utc_now
    _account: str | None = None

    def account(self) -> str:
        if self._account is None:
            self._account = self.sts.get_caller_identity()["Account"]
        return self._account

    def package_arn(self, version: int) -> str:
        """arn:aws:sagemaker:<region>:<account>:model-package/<group>/<version> -- the account comes from STS."""
        return f"arn:aws:sagemaker:{self.cfg.region}:{self.account()}:model-package/{self.cfg.group}/{version}"

    def shared(self, attr: str) -> str:
        """A name that contains the account id (the ML bucket, the SageMaker role).

        Order: flag / environment variable -> SSM /finrisk/shared/<name> (written by finrisk-infra) ->
        the naming convention from the course. The source is printed, so you always know which won."""
        value = getattr(self.cfg, attr)
        if value:
            return value
        ssm_name = {"ml_bucket": "ml_bucket", "role_arn": "sagemaker_role_arn"}[attr]
        try:
            value = self.ssm.get_parameter(Name=f"/finrisk/shared/{ssm_name}")["Parameter"]["Value"]
            self.out(f"  {attr} taken from SSM /finrisk/shared/{ssm_name}")
        except Exception as exc:  # noqa: BLE001
            if error_code(exc) not in ("ParameterNotFound", "AccessDeniedException", "AccessDenied"):
                raise
            account = self.account()
            value = {
                "ml_bucket": f"finrisk-ml-{account}-{self.cfg.region}-an",
                "role_arn": f"arn:aws:iam::{account}:role/finrisk-sagemaker-exec",
            }[attr]
            self.out(f"  {attr} built from the naming convention (SSM said {error_code(exc)})")
        setattr(self.cfg, attr, value)
        return value


def write_summary(path: str | None, lines: list[str]) -> None:
    """Append markdown to the GitHub job summary (any file works). Account ids are masked."""
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(mask("\n".join(lines)) + "\n")


def run_url(env: dict[str, str]) -> str:
    server, repo, run_id = env.get("GITHUB_SERVER_URL"), env.get("GITHUB_REPOSITORY"), env.get("GITHUB_RUN_ID")
    return f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else "local"


# ---------------------------------------------------------------------------- 1. prepare-data


def docker_gold_command(image: str, workdir: Path, rows: int) -> list[str]:
    """`docker run` of the training image, running the APP's data pipeline (scripts/run_pipeline.py).

    Everything the pipeline writes as data goes into the mounted folder; the entrypoint is replaced
    (the training image's own entrypoint is the trainer), and the LLM is switched off."""
    return [
        "docker", "run", "--rm",
        "-v", f"{workdir}:/out",
        "-e", "DATA_MODE=demo",
        "-e", "RAW_DATA_PATH=/out/raw/paysim.csv",
        "-e", "BRONZE_PATH=/out/bronze/transactions.parquet",
        "-e", "SILVER_PATH=/out/silver/clean_transactions.parquet",
        "-e", "GOLD_PATH=/out/gold",
        "-e", "LLM_PROVIDER=none",
        "-e", "DEEPSEEK_API_KEY=",
        "--workdir", "/app",
        "--entrypoint", "python",
        image, "scripts/run_pipeline.py", "--demo-rows", str(rows),
    ]


def cmd_prepare_data(ctx: Context, a: argparse.Namespace) -> str:
    bucket = ctx.shared("ml_bucket")
    image = a.image or ctx.env.get("TRAIN_IMAGE", "")
    if not image:
        raise MlopsError("--image is required (the training image: it contains the app's scripts/run_pipeline.py)")
    rows = a.rows or ctx.tcfg.demo_rows
    workdir = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="finrisk-data-"))
    workdir.mkdir(parents=True, exist_ok=True)
    command = docker_gold_command(image, workdir, rows)
    ctx.out(f"  RUN  {' '.join(command)}")
    code = ctx.run(command)
    if code != 0:
        raise MlopsError(f"the data pipeline in the training image failed (docker exit {code}); its output is above.")
    gold = workdir / "gold" / "transaction_features.parquet"
    if not gold.is_file():
        raise MlopsError(f"the pipeline finished but {gold} does not exist: the app's run_pipeline.py wrote the gold table elsewhere.")
    stamp = a.stamp or utc_stamp("%Y%m%d-%H%M%S")                     # the DATA VERSION
    key = f"data/{stamp}/train.parquet"
    digest = hashlib.sha256(gold.read_bytes()).hexdigest()
    ctx.s3.upload_file(Filename=str(gold), Bucket=bucket, Key=key)
    ctx.out(f"  data version {stamp}: {gold.stat().st_size / 1024:.0f} KB, sha256 {digest[:12]}... uploaded as train.parquet")
    return f"s3://{bucket}/data/{stamp}/"


def latest_data_prefix(ctx: Context) -> str:
    """The newest folder under data/ (folder names are timestamps, so the last one sorts newest)."""
    bucket = ctx.shared("ml_bucket")
    prefixes: list[str] = []
    token = None
    while True:
        params: dict[str, Any] = {"Bucket": bucket, "Prefix": "data/", "Delimiter": "/"}
        if token:
            params["ContinuationToken"] = token
        page = ctx.s3.list_objects_v2(**params)
        prefixes += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        token = page.get("NextContinuationToken")
        if not token:
            break
    if not prefixes:
        raise MlopsError(f"no training data in s3://{bucket}/data/. Run: python pipeline/mlops.py prepare-data --image ...")
    return f"s3://{bucket}/{max(prefixes)}"


# ---------------------------------------------------------------------------- 2. train


def metric_definitions() -> list[dict[str, str]]:
    """One regular expression per metric, matching the ``name=value`` lines train/sagemaker_train.py prints."""
    return [{"Name": name, "Regex": rf"{name}=([0-9\.]+)"} for name in METRIC_NAMES]


def build_training_request(cfg: Config, tcfg: TrainConfig, image: str, job_name: str, data_uri: str,
                           model_type: str | None = None, git_sha: str = "unknown") -> dict[str, Any]:
    """The whole CreateTrainingJob request. A pure function: print it, test it, change one line of it."""
    hyper = dict(tcfg.hyperparameters)
    if model_type:
        hyper["model_type"] = model_type
    return {
        "TrainingJobName": job_name,
        "RoleArn": cfg.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image,                 # the training image, by digest: the exact bytes, not a moving tag
            "TrainingInputMode": "File",            # SageMaker downloads the data into /opt/ml/input/data/train first
            "ContainerEntrypoint": list(ENTRYPOINT),
            "MetricDefinitions": metric_definitions(),
        },
        "InputDataConfig": [{
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": data_uri, "S3DataDistributionType": "FullyReplicated"}},
        }],
        "OutputDataConfig": {"S3OutputPath": f"s3://{cfg.ml_bucket}/models/"},
        "ResourceConfig": {"InstanceType": tcfg.instance_type, "InstanceCount": tcfg.instance_count,
                           "VolumeSizeInGB": tcfg.volume_gb},
        "StoppingCondition": {"MaxRuntimeInSeconds": tcfg.max_runtime_seconds},
        "HyperParameters": hyper,
        "Environment": {
            "AWS_REGION": cfg.region,
            "MODEL_TYPE": hyper.get("model_type", "optimized"),
            # The training script uploads its metrics here, and `register` points the registry at it.
            "METRICS_S3_URI": f"s3://{cfg.ml_bucket}/metrics/{job_name}/metrics.json",
            "ML_GIT_SHA": git_sha,                  # which commit of THIS repo asked for the job
        },
        "Tags": [
            {"Key": "Project", "Value": "finrisk"},
            {"Key": "ManagedBy", "Value": "mlops.py"},
            {"Key": "Course", "Value": "financial-risk-assistant"},
            {"Key": "MlGitSha", "Value": git_sha},
        ],
    }


def wait_for_job(ctx: Context, job_name: str, poll: float, max_wait: float) -> dict[str, Any]:
    """Poll until the job ends, printing each time its status changes."""
    started, last = ctx.clock(), None
    ctx.out(f"  waiting; checking every {poll:.0f} s and printing only when the status changes")
    while True:
        desc = ctx.sm.describe_training_job(TrainingJobName=job_name)
        state = (desc["TrainingJobStatus"], desc.get("SecondaryStatus", ""))
        if state != last:
            ctx.out(f"  [{ctx.clock() - started:5.0f}s] {state[0]:<10} {state[1]}")
            last = state
        if state[0] in TERMINAL:
            return desc
        if ctx.clock() - started > max_wait:
            raise MlopsError(
                f"still {state[0]}/{state[1]} after {max_wait:.0f} s. The job keeps running in AWS (it stops itself at "
                f"{ctx.tcfg.max_runtime_seconds} s). Look at it with: aws sagemaker describe-training-job --training-job-name {job_name}")
        ctx.sleep(poll)


def cmd_train(ctx: Context, a: argparse.Namespace) -> str:
    ctx.shared("ml_bucket")
    ctx.shared("role_arn")
    if not DIGEST_IMAGE.match(a.image):
        raise MlopsError(
            "--image must be an image DIGEST (<repo>@sha256:<64 hex>), not a tag: a tag can move, a digest cannot, "
            "and the model must be trained with exactly the image you can name later.")
    job_name = a.job_name or f"finrisk-train-{utc_stamp('%Y%m%d-%H%M%S')}"
    data_uri = a.data or latest_data_prefix(ctx)
    git_sha = ctx.env.get("ML_GIT_SHA") or ctx.env.get("GITHUB_SHA") or "unknown"
    ctx.out(f"training job {job_name} on {data_uri}")
    ctx.sm.create_training_job(**build_training_request(ctx.cfg, ctx.tcfg, a.image, job_name, data_uri, a.model_type, git_sha))
    if a.no_wait:
        ctx.out("not waiting (--no-wait). Check it with: evaluate --training-job " + job_name)
        return job_name
    desc = wait_for_job(ctx, job_name, a.poll, a.max_wait)
    status = desc["TrainingJobStatus"]
    ctx.out(f"  training time {desc.get('TrainingTimeInSeconds', '?')} s, billable {desc.get('BillableTimeInSeconds', '?')} s "
            f"on {ctx.tcfg.instance_type} x{ctx.tcfg.instance_count}")
    if status != "Completed":
        reason = mask(desc.get("FailureReason", "no reason given"))
        raise MlopsError(f"training job {job_name} {status}: {reason}\n  logs: CloudWatch log group /aws/sagemaker/TrainingJobs, stream {job_name}/...")
    ctx.out("  final metrics:")
    for m in desc.get("FinalMetricDataList", []):
        ctx.out(f"    {m['MetricName']:<10} {m['Value']:.6g}")
    ctx.out(f"  model: {desc['ModelArtifacts']['S3ModelArtifacts']}")
    ctx.out(f"next: python pipeline/mlops.py evaluate --training-job {job_name}")
    return job_name


# ---------------------------------------------------------------------------- 3. evaluate (the gate)


@dataclass
class Check:
    name: str
    needs: str            # the rule in words, e.g. ">= 0.7000"
    passed: bool
    skipped: bool = False


def gate_checks(candidate: float, floor: float, tolerance: float, champion: float | None,
                champion_label: str = "champion") -> list[Check]:
    """The whole gate, as a pure function. The candidate passes only if EVERY check passes.

        1. candidate >= floor                      an absolute minimum, whatever runs today
        2. candidate >= champion - tolerance       not noticeably worse than the model in production
           (skipped when there is no champion yet: the first model only has to clear the floor)
    """
    eps = 1e-9                                           # 0.78 >= 0.80 - 0.02 must pass despite float noise
    checks = [Check("floor", f">= {floor:.4f}", candidate + eps >= floor)]
    if champion is None:
        checks.append(Check("champion", "no champion yet", True, skipped=True))
    else:
        needed = champion - tolerance
        checks.append(Check(f"{champion_label} - {tolerance:g}", f">= {needed:.4f}", candidate + eps >= needed))
    return checks


def gate_table(job: str, candidate: float, checks: list[Check]) -> list[str]:
    lines = [f"  gate for {job}  (candidate PR-AUC {candidate:.4f})",
             f"  {'check':<24} {'candidate':>9}   {'needs':<18} result"]
    for c in checks:
        shown = "-" if c.skipped else f"{candidate:.4f}"
        result = "skipped" if c.skipped else ("PASS" if c.passed else "FAIL")
        lines.append(f"  {c.name:<24} {shown:>9}   {c.needs:<18} {result}")
    return lines


def metrics_from_s3(ctx: Context, uri: str) -> dict[str, Any]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return json.loads(ctx.s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def candidate_metrics(ctx: Context, desc: dict[str, Any], job: str) -> dict[str, float]:
    """The numbers the training job reported: CloudWatch's view of the printed lines, else metrics.json."""
    found = {m["MetricName"]: float(m["Value"]) for m in desc.get("FinalMetricDataList", []) if "MetricName" in m}
    if ctx.tcfg.gate_metric in found:
        return found
    uri = (desc.get("Environment") or {}).get("METRICS_S3_URI") or f"s3://{ctx.cfg.ml_bucket}/metrics/{job}/metrics.json"
    ctx.out(f"  {ctx.tcfg.gate_metric} is not in the job's final metrics; reading {uri}")
    try:
        body = metrics_from_s3(ctx, uri)
    except Exception as exc:  # noqa: BLE001
        raise MlopsError(f"cannot read the candidate's {ctx.tcfg.gate_metric}: no final metric and {mask(uri)} is unreadable "
                         f"({error_code(exc) or type(exc).__name__}).") from exc
    return {k: float(v) for k, v in body.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def package_pr_auc(ctx: Context, detail: dict[str, Any]) -> float | None:
    """The PR-AUC of a registered version: its registry metadata, else its metrics file."""
    meta = detail.get("CustomerMetadataProperties", {})
    if meta.get("pr_auc") not in (None, ""):
        return float(meta["pr_auc"])
    uri = detail.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {}).get("S3Uri")
    if uri:
        value = metrics_from_s3(ctx, uri).get("pr_auc")
        return float(value) if value is not None else None
    return None


def read_pointer(ctx: Context, env_name: str) -> dict[str, Any] | None:
    """/finrisk/<env>/model as a dict, or None when it does not exist or has no version yet."""
    name = f"/finrisk/{env_name}/model"
    try:
        value = ctx.ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) == "ParameterNotFound":
            return None
        raise
    try:
        pointer = json.loads(value)
    except ValueError as exc:
        raise MlopsError(f"{name} does not hold JSON: {mask(value[:80])!r}") from exc
    return pointer if isinstance(pointer, dict) and pointer.get("version") is not None else None


def find_champion(ctx: Context) -> tuple[int, float] | None:
    """(version, PR-AUC) of the model running in the champion environment (prod), or None."""
    pointer = read_pointer(ctx, ctx.tcfg.champion_env)
    if pointer is None:
        return None
    version = int(pointer["version"])
    detail = describe_version(ctx, version)
    pr_auc = package_pr_auc(ctx, detail)
    if pr_auc is None:
        raise MlopsError(f"the champion (version {version}, in {ctx.tcfg.champion_env}) has no PR-AUC in the registry: "
                         "cannot compare against it, so the gate refuses to guess.")
    return version, pr_auc


def cmd_evaluate(ctx: Context, a: argparse.Namespace) -> str:
    ctx.shared("ml_bucket")
    desc = ctx.sm.describe_training_job(TrainingJobName=a.training_job)
    if desc["TrainingJobStatus"] != "Completed":
        raise MlopsError(f"training job {a.training_job} is {desc['TrainingJobStatus']}, not Completed: nothing to evaluate.")
    metric = ctx.tcfg.gate_metric
    metrics = candidate_metrics(ctx, desc, a.training_job)
    if metric not in metrics:
        raise MlopsError(f"the training job reported no {metric}. Its metrics: {sorted(metrics)}")
    candidate = metrics[metric]
    champion = find_champion(ctx)
    checks = gate_checks(candidate, ctx.tcfg.floor, ctx.tcfg.tolerance,
                         champion[1] if champion else None,
                         champion_label=f"champion v{champion[0]} ({champion[1]:.4f})" if champion else "champion")
    passed = all(c.passed for c in checks)
    table = gate_table(a.training_job, candidate, checks)
    for line in table:
        ctx.out(line)

    failed = [c for c in checks if not c.passed]
    where = ctx.tcfg.champion_env
    summary = [f"### Gate: {'PASSED' if passed else 'FAILED'} for `{a.training_job}`", "",
               f"| check | candidate {metric} | needs | result |", "| --- | --- | --- | --- |"]
    for c in checks:
        summary.append(f"| {c.name} | {'-' if c.skipped else f'{candidate:.4f}'} | {c.needs} | "
                       f"{'skipped' if c.skipped else ('pass' if c.passed else '**fail**')} |")
    summary += ["", (f"The champion is version {champion[0]} in `{where}` (PR-AUC {champion[1]:.4f})." if champion
                    else f"There is no champion yet: `/finrisk/{where}/model` has no version.")]
    if not passed:
        why = "; ".join(f"{c.name}: needs {c.needs}" for c in failed)
        summary += ["", f"**Rejected before the registry** ({why}). Nothing was registered; the models in use are unchanged."]
    write_summary(a.summary, summary)

    if not passed:
        raise MlopsError(f"GATE FAILED for {a.training_job}: candidate {metric} {candidate:.4f}; "
                         + "; ".join(f"{c.name} needs {c.needs}" for c in failed), EXIT_GATE)
    ctx.out(f"  gate passed: {a.training_job} may be registered")
    ctx.out(f"next: python pipeline/mlops.py register --training-job {a.training_job}")
    return a.training_job


# ---------------------------------------------------------------------------- 4. register


def cap(value: Any) -> str:
    return str(value)[:256]                      # a registry metadata value is at most 256 characters


def build_package_request(cfg: Config, image: str, model_url: str, metrics_uri: str, job_name: str, data_uri: str,
                          metrics: dict[str, Any], git_sha: str, run: str) -> dict[str, Any]:
    """The whole CreateModelPackage request: the model, how to run it, how good it is, and where it came from."""
    pr_auc = metrics.get("pr_auc")
    digest = image.partition("@")[2]
    metadata = {
        "training_job": job_name,
        "train_image": image,                    # exactly which training image (by digest) made this model
        "train_image_digest": digest,
        "data_uri": data_uri,
        "data_version": data_uri.rstrip("/").rsplit("/", 1)[-1],
        "ml_git_sha": git_sha,
        "run": run,
    }
    for name in METRIC_NAMES:
        if metrics.get(name) is not None:
            metadata[name] = f"{float(metrics[name]):.6f}" if name != "rows" else str(int(metrics[name]))
    quality = f"PR-AUC {pr_auc:.4f}" if pr_auc is not None else "PR-AUC unknown"
    threshold = metrics.get("threshold")
    return {
        "ModelPackageGroupName": cfg.group,
        "ModelPackageDescription": f"Fraud model from {job_name}: {quality}" + (f", threshold {threshold:.4f}" if threshold is not None else ""),
        "InferenceSpecification": {
            "Containers": [{"Image": image, "ModelDataUrl": model_url}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        "ModelMetrics": {"ModelQuality": {"Statistics": {"ContentType": "application/json", "S3Uri": metrics_uri}}},
        # A new version always starts here. A person moves it on: `approve --version N`.
        "ModelApprovalStatus": "PendingManualApproval",
        "CustomerMetadataProperties": {k: cap(v) for k, v in metadata.items()},
    }


def cmd_register(ctx: Context, a: argparse.Namespace) -> str:
    ctx.cfg.need("group")
    ctx.shared("ml_bucket")
    desc = ctx.sm.describe_training_job(TrainingJobName=a.training_job)
    if desc["TrainingJobStatus"] != "Completed":
        raise MlopsError(f"training job {a.training_job} is {desc['TrainingJobStatus']}, not Completed: nothing to register yet.")
    image = desc["AlgorithmSpecification"]["TrainingImage"]
    model_url = desc["ModelArtifacts"]["S3ModelArtifacts"]
    data_uri = desc["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    job_env = desc.get("Environment") or {}
    bucket = model_url.removeprefix("s3://").partition("/")[0]
    metrics_uri = job_env.get("METRICS_S3_URI") or f"s3://{bucket}/metrics/{a.training_job}/metrics.json"
    try:
        ctx.s3.head_object(Bucket=metrics_uri.removeprefix("s3://").partition("/")[0], Key=metrics_uri.removeprefix("s3://").partition("/")[2])
    except Exception as exc:  # noqa: BLE001
        raise MlopsError(f"metrics file {mask(metrics_uri)} is missing ({error_code(exc) or type(exc).__name__}). A registered model needs its metrics.") from exc
    metrics = metrics_from_s3(ctx, metrics_uri)
    git_sha = job_env.get("ML_GIT_SHA") or ctx.env.get("GITHUB_SHA") or "unknown"
    request = build_package_request(ctx.cfg, image, model_url, metrics_uri, a.training_job, data_uri, metrics, git_sha, run_url(ctx.env))
    arn = ctx.sm.create_model_package(**request)["ModelPackageArn"]
    version = int(arn.rsplit("/", 1)[1])
    pr_auc = metrics.get("pr_auc")
    ctx.out(f"registered {mask(arn)}")
    ctx.out(f"  version {version}, status PendingManualApproval, PR-AUC {pr_auc if pr_auc is None else format(pr_auc, '.4f')}")
    ctx.out(f"next: run approve-model for version {version} (python pipeline/mlops.py approve --version {version} --decision approve)")
    meta = request["CustomerMetadataProperties"]
    write_summary(a.summary, [
        "", f"### Registered: model version {version} (PendingManualApproval)", "",
        "| | |", "| --- | --- |",
        f"| training job | `{a.training_job}` |",
        f"| data version | `{meta['data_version']}` |",
        f"| ml commit | `{meta['ml_git_sha'][:12]}` |",
        f"| training image | `{meta['train_image_digest'][:19]}...` |"] + [
        f"| {name} | {meta[name]} |" for name in ("pr_auc", "roc_auc", "precision", "recall", "threshold") if name in meta] + [
        "", f"Next: run **approve-model** with version {version}, then **promote-model**."])
    return str(version)


# ---------------------------------------------------------------------------- 5. approve / reject


def describe_version(ctx: Context, version: int) -> dict[str, Any]:
    try:
        return ctx.sm.describe_model_package(ModelPackageName=ctx.package_arn(version))
    except MlopsError:
        raise
    except Exception as exc:  # noqa: BLE001
        if error_code(exc) in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException"):
            raise MlopsError(f"model version {version} not found in group {ctx.cfg.group}. See the versions: python pipeline/mlops.py list", EXIT_UNUSABLE) from exc
        raise MlopsError(f"could not read model version {version}: {mask(str(exc))}") from exc


def cmd_approve(ctx: Context, a: argparse.Namespace) -> str:
    describe_version(ctx, a.version)            # a clear "not found" before we try to change anything
    new_status = "Approved" if a.decision == "approve" else "Rejected"
    params: dict[str, Any] = {"ModelPackageArn": ctx.package_arn(a.version), "ModelApprovalStatus": new_status}
    if a.note:
        params["ApprovalDescription"] = a.note[:1024]
    ctx.sm.update_model_package(**params)
    status = describe_version(ctx, a.version)["ModelApprovalStatus"]
    ctx.out(f"model version {a.version} is now {status}")
    write_summary(a.summary, [f"### Model version {a.version}: {a.decision} -> {status}"] + ([f"Note: {a.note}"] if a.note else []))
    return status


# ---------------------------------------------------------------------------- 6. promote


def find_load_error(ready: dict[str, Any]) -> str:
    """The app's own explanation of why it could not load an artifact, wherever /ready puts it."""
    model = ready.get("model") if isinstance(ready.get("model"), dict) else {}
    if "load_error" in model:
        # The real /ready: the model section carries the model's OWN verdict. The top-level load_error is "the first
        # problem of the model OR the index", so a broken index must not fail a model promotion.
        return str(model["load_error"]) if model["load_error"] else ""
    for holder in (ready, model):
        for key in ("load_error", "error", "last_error"):
            if holder.get(key):
                return str(holder[key])
    return ""


def parse_time(value: Any) -> datetime | None:
    """ISO text or epoch seconds -> an aware datetime, or None when it is neither."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def wait_for_model(ctx: Context, env_name: str, app_url: str, version: int, promoted_at: datetime,
                   timeout: float, poll: float) -> float:
    """Poll <app_url>/ready until model.version == version. Returns the seconds it took.

    The app re-reads the pointer on its own schedule (POINTER_TTL_SECONDS), so this is normally one or
    two polls. A load_error reported AFTER the pointer was written is a verdict: stop with its text.
    An error the app reported before that (about an older pointer) is not."""
    url = app_url.rstrip("/") + "/ready"
    started = ctx.clock()
    last_error = ""
    ctx.out(f"  waiting for {mask(url)} to report model.version {version} (timeout {timeout:.0f} s, polling every {poll:.0f} s)")
    while True:
        elapsed = ctx.clock() - started
        data: dict[str, Any] | None = None
        try:
            status, body = ctx.http(url, 10)
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
        except Exception as exc:  # noqa: BLE001 - a cold Lambda or a starting task may not answer yet
            ctx.out(f"  [{elapsed:5.0f}s] GET /ready -> no usable answer ({type(exc).__name__}: {mask(str(exc))[:80]})")
        if data is not None:
            model = data.get("model") if isinstance(data.get("model"), dict) else {}
            seen = model.get("version")
            ctx.out(f"  [{elapsed:5.0f}s] GET /ready -> {status}  model.version={seen}  (want {version})"
                    + (f"  source={model['source']}" if model.get("source") else ""))
            if seen is not None and str(seen) == str(version):
                return elapsed
            error = find_load_error(data)
            if error:
                last_error = error
                checked = parse_time(data.get("pointers_checked_at"))
                if checked is None or checked >= promoted_at:
                    raise MlopsError(f"{env_name} could not load model version {version}; the app says: {mask(error)}. "
                                     f"It keeps serving version {seen}. The pointer was written; promote the previous version to put it back.")
        if ctx.clock() - started > timeout:
            more = f" Last load_error from the app: {mask(last_error)}" if last_error else ""
            raise MlopsError(f"{env_name} did not report model version {version} within {timeout:.0f} s (last seen: "
                             f"{(data or {}).get('model', {}).get('version') if data else 'no answer'}).{more}")
        ctx.sleep(poll)


def cmd_promote(ctx: Context, a: argparse.Namespace) -> str:
    detail = describe_version(ctx, a.version)
    status = detail["ModelApprovalStatus"]
    ctx.out(f"model version {a.version}: {status}")
    if status == "Rejected":
        raise MlopsError(f"model version {a.version} was Rejected: it must not run in any environment. Pick another version.", EXIT_UNUSABLE)
    if a.env in NEEDS_APPROVAL and status != "Approved":
        raise MlopsError(
            f"model version {a.version} is {status}, not Approved: {a.env} only runs approved models. "
            f"Run approve-model first (python pipeline/mlops.py approve --version {a.version} --decision approve).", EXIT_UNUSABLE)

    app_url = ""
    if not a.no_wait:
        # Look the app up BEFORE changing anything: promoting what we cannot verify is how a bad model hides.
        try:
            app_url = ctx.ssm.get_parameter(Name=f"/finrisk/{a.env}/app_url")["Parameter"]["Value"].strip()
        except Exception as exc:  # noqa: BLE001
            if error_code(exc) != "ParameterNotFound":
                raise
            raise MlopsError(f"/finrisk/{a.env}/app_url does not exist: the app is not deployed to {a.env} yet, so there is nothing "
                             "to check the model against. Deploy finrisk-app first (or use --no-wait to only move the pointer).") from exc

    promoted_at = ctx.now()
    pointer = {
        "version": a.version,
        "model_data_url": detail["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
        "status": status,
        "promoted_at": promoted_at.isoformat(timespec="seconds"),
        "run": run_url(ctx.env),
    }
    ctx.ssm.put_parameter(Name=f"/finrisk/{a.env}/model", Value=json.dumps(pointer), Type="String", Overwrite=True)
    ctx.out(f"  /finrisk/{a.env}/model now points at version {a.version}")

    took = 0.0
    if a.no_wait:
        ctx.out("not waiting (--no-wait): the app picks the pointer up within its POINTER_TTL_SECONDS.")
    else:
        took = wait_for_model(ctx, a.env, app_url, a.version, promoted_at, a.timeout, a.poll)
        ctx.out(f"  {a.env} serves model version {a.version} (confirmed by /ready after {took:.0f} s)")
    write_summary(a.summary, [f"- **{a.env}**: model version {a.version} ({status})" + ("" if a.no_wait else f", confirmed by `/ready` after {took:.0f} s")])
    return str(a.version)


# ---------------------------------------------------------------------------- 7. list / show


def when(value: Any) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if hasattr(value, "strftime") else str(value)[:16]


def cmd_list(ctx: Context, a: argparse.Namespace) -> str | None:
    ctx.cfg.need("group")
    page = ctx.sm.list_model_packages(ModelPackageGroupName=ctx.cfg.group, SortBy="CreationTime", SortOrder="Descending", MaxResults=a.limit)
    rows = []
    for summary in page.get("ModelPackageSummaryList", []):
        detail = describe_version(ctx, int(summary["ModelPackageVersion"]))
        meta = detail.get("CustomerMetadataProperties", {})
        rows.append((summary["ModelPackageVersion"], summary["ModelApprovalStatus"], meta.get("pr_auc", "-"),
                     when(summary["CreationTime"]), meta.get("training_job", "-")))
    if not rows:
        ctx.out(f"no versions in {ctx.cfg.group} yet. Train and register one: train, evaluate, register.")
        return None
    ctx.out(f"{'version':>7}  {'status':<22} {'PR-AUC':>8}  {'created (UTC)':<16}  training job")
    for version, status, pr_auc, created, job in rows:
        shown = f"{float(pr_auc):.4f}" if pr_auc not in ("-", None) else "-"
        ctx.out(f"{version:>7}  {status:<22} {shown:>8}  {created:<16}  {job}")
    return None


def cmd_show(ctx: Context, a: argparse.Namespace) -> str | None:
    d = describe_version(ctx, a.version)
    meta = d.get("CustomerMetadataProperties", {})
    container = d["InferenceSpecification"]["Containers"][0]
    ctx.out(f"model version {a.version} of {d['ModelPackageGroupName']}")
    ctx.out(f"  status        {d['ModelApprovalStatus']}" + (f"  ({d['ApprovalDescription']})" if d.get("ApprovalDescription") else ""))
    ctx.out(f"  created       {when(d.get('CreationTime'))} UTC")
    ctx.out(f"  model file    {container['ModelDataUrl']}")
    ctx.out(f"  image         {container['Image']}")
    for key, value in meta.items():
        ctx.out(f"  {key:<18} {value}")
    return None


# ---------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ml-bucket", help="the ML bucket (env ML_BUCKET; else SSM /finrisk/shared/ml_bucket)")
    common.add_argument("--group", help=f"the model package group (env MODEL_PACKAGE_GROUP, default {DEFAULT_GROUP})")
    common.add_argument("--role-arn", help="the SageMaker execution role (env SAGEMAKER_ROLE_ARN; else SSM)")
    common.add_argument("--region", help=f"AWS Region (env AWS_REGION, default {DEFAULT_REGION})")
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="the training config (default config/train.yaml)")
    common.add_argument("--quiet", action="store_true", help="print only the final value (and errors)")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", parents=[common], help="make the gold table in the training image; upload to s3://<ml>/data/<time>/")
    p.add_argument("--image", help="the training image (env TRAIN_IMAGE); it contains the app's scripts/run_pipeline.py")
    p.add_argument("--rows", type=int, help="demo rows (default from the config)")
    p.add_argument("--stamp", help="data version = folder name under data/ (default: the current UTC time)")
    p.add_argument("--workdir", help="folder for the pipeline's output (default: a new temp folder)")

    p = sub.add_parser("train", parents=[common], help="run a SageMaker training job with the training image and wait for it")
    p.add_argument("--image", required=True, help="<repo>@sha256:<digest> of the training image")
    p.add_argument("--data", help="s3 folder holding train.parquet (default: the newest folder under data/)")
    p.add_argument("--job-name")
    p.add_argument("--model-type", choices=["optimized", "baseline"], help="override the config's model_type")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll", type=float, default=15.0, help="seconds between status checks")
    p.add_argument("--max-wait", type=float, default=2700.0, help="stop waiting (not the job) after this many seconds")

    p = sub.add_parser("evaluate", parents=[common], help="THE GATE: exit 3 unless the candidate clears the floor and the champion")
    p.add_argument("--training-job", required=True)
    p.add_argument("--summary", help="append a markdown table here (the GitHub job summary)")

    p = sub.add_parser("register", parents=[common], help="add a finished training job to the registry as a new version")
    p.add_argument("--training-job", required=True)
    p.add_argument("--summary", help="append a markdown note here (the GitHub job summary)")

    p = sub.add_parser("approve", parents=[common], help="approve (or reject) a model version")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--decision", choices=["approve", "reject"], default="approve")
    p.add_argument("--note", help="why (stored with the decision)")
    p.add_argument("--summary", help="append a markdown note here (the GitHub job summary)")

    p = sub.add_parser("promote", parents=[common], help="point an environment at a version; wait for /ready to confirm")
    p.add_argument("--env", choices=ENVIRONMENTS, required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--poll", type=float, default=15.0, help="seconds between /ready checks")
    p.add_argument("--timeout", type=float, default=600.0, help="give up waiting for /ready after this many seconds")
    p.add_argument("--no-wait", action="store_true", help="only move the pointer; do not look at the app")
    p.add_argument("--summary", help="append a markdown line here (the GitHub job summary)")

    p = sub.add_parser("list", parents=[common], help="the versions in the registry")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("show", parents=[common], help="everything about one version")
    p.add_argument("--version", type=int, required=True)
    return parser


COMMANDS = {
    "prepare-data": cmd_prepare_data, "train": cmd_train, "evaluate": cmd_evaluate, "register": cmd_register,
    "approve": cmd_approve, "promote": cmd_promote, "list": cmd_list, "show": cmd_show,
}


def main(argv: list[str] | None = None, clients: dict[str, Any] | None = None, env: dict[str, str] | None = None,
         out: Callable[[str], None] | None = None, sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic, http: Callable[[str, float], tuple[int, str]] = http_get,
         run: Callable[[list[str]], int] = run_command, now: Callable[[], datetime] = utc_now) -> int:
    """Run one sub-command; return the exit code. ``clients`` lets a test pass fakes instead of boto3."""
    args = build_parser().parse_args(argv)
    env = dict(os.environ if env is None else env)
    echo = out or (lambda line: print(line, flush=True))
    say = (lambda line: None) if args.quiet else (lambda line: echo(mask(line)))   # ids hidden in everything but the result
    try:
        cfg = load_config(args, env)
        tcfg = load_train_config(Path(args.config))
        raw = clients or make_clients(cfg.region)
        ctx = Context(
            cfg=cfg, tcfg=tcfg,
            sm=Traced(raw["sagemaker"], "sagemaker", say, quiet_after_first=("describe_training_job",)),
            s3=Traced(raw["s3"], "s3", say),
            sts=Traced(raw["sts"], "sts", say),
            ssm=Traced(raw["ssm"], "ssm", say),
            out=say, sleep=sleep, clock=clock, env=env, http=http, run=run, now=now)
        result = COMMANDS[args.command](ctx, args)
    except MlopsError as exc:
        print(f"error: {mask(str(exc))}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # noqa: BLE001 - an AWS error should be one readable line, not a traceback
        print(f"error: {type(exc).__name__}: {mask(str(exc))}", file=sys.stderr)
        if "botocore[crt]" in str(exc):
            print("hint: you signed in with `aws login`. Either `pip install \"botocore[crt]\"`, or hand boto3 plain "
                  "credentials: eval \"$(aws configure export-credentials --format env)\"", file=sys.stderr)
        return EXIT_FAILED
    if result is not None:
        echo(result)                  # the value the next step needs -- always the last line, never masked
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

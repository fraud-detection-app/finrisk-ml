"""The training entrypoint that runs INSIDE the training image -- on SageMaker, or on your laptop.

SageMaker's training contract is small. It starts our container and says, with folders:

    /opt/ml/input/data/train/     the training data we asked for (a "channel" called train)
    /opt/ml/model/                put the model here; SageMaker tars it to model.tar.gz in S3
    /opt/ml/output/data/          anything else worth keeping (it becomes output.tar.gz)

and, with the log: every line shaped like ``name=value`` can be turned into a CloudWatch metric
(``pipeline/mlops.py train`` registers a regular expression per name in METRIC_NAMES).

This script is the whole bridge. It adds no new modeling: it calls the SAME ``train_model`` the app
uses (the training image starts FROM the app image, so the feature code, the model code and the file
format are exactly what the service loads). A model trained here is served with no conversion.

What it writes to the model folder -- the "model package" (a flat model.tar.gz) the app reads:

    model_v2.pkl            the model + threshold + feature list   (model_v1.pkl for model_type=baseline)
    latest_model.joblib     a pointer: {"model_path": "model_v2.pkl"}  (a file NAME, never a path)
    features.parquet        a copy of the gold table the model was trained on: the app scores
                            /api/predict/<row_id> from it
    metrics.json            pr_auc, roc_auc, threshold, precision, recall, rows ...

Run it on SageMaker (the normal way):   python pipeline/mlops.py train --image <repo>@sha256:...
Run it on your laptop, no AWS needed:

    FINRISK_APP_SRC=../finrisk-app/src SM_CHANNEL_TRAIN=/tmp/in SM_MODEL_DIR=/tmp/m SM_OUTPUT_DATA_DIR=/tmp/o \\
        python train/sagemaker_train.py            # /tmp/in holds train.parquet

Environment (all optional; the defaults are SageMaker's own paths):
    SM_CHANNEL_TRAIN     folder with the feature parquet   (default /opt/ml/input/data/train)
    SM_MODEL_DIR         where the model is written        (default /opt/ml/model)
    SM_OUTPUT_DATA_DIR   where extra output is written     (default /opt/ml/output/data)
    FINRISK_APP_SRC      folder that contains the ``financial_risk`` package. Inside the image the
                         base image already has it on PYTHONPATH (/app/src); on your laptop point here.
    MODEL_TYPE           optimized (XGBoost) or baseline (logistic regression); default optimized.
                         SageMaker's HyperParameters {"model_type": ...} work too (hyperparameters.json).
    METRICS_S3_URI       if set, also upload metrics.json to this s3:// address (the registry points at it)
    TRAINING_JOB_NAME    set by SageMaker; recorded in metrics.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# The metrics we print, one ``name=value`` line each. pipeline/mlops.py builds its CloudWatch
# regular expressions from this list, so the two cannot drift apart (a test checks they match).
METRIC_NAMES = ("pr_auc", "roc_auc", "threshold", "precision", "recall", "rows")

HYPERPARAMETERS_FILE = Path("/opt/ml/input/config/hyperparameters.json")
IMAGE_APP_SRC = Path("/app/src")           # where the app image keeps the financial_risk package


def find_training_file(channel: Path) -> Path:
    """The feature table inside the channel folder: train.parquet if present, else the only parquet."""
    if not channel.is_dir():
        raise FileNotFoundError(f"training channel folder not found: {channel}")
    named = channel / "train.parquet"
    if named.exists():
        return named
    parquets = sorted(channel.glob("*.parquet"))
    if len(parquets) == 1:
        return parquets[0]
    found = sorted(p.name for p in channel.iterdir())
    raise FileNotFoundError(
        f"expected train.parquet (or exactly one .parquet) in {channel}; found {found or 'nothing'}")


def model_type_setting(environ: dict[str, str]) -> str:
    """MODEL_TYPE env var, else SageMaker's hyperparameters.json, else optimized."""
    value = environ.get("MODEL_TYPE")
    if not value and HYPERPARAMETERS_FILE.exists():
        raw = json.loads(HYPERPARAMETERS_FILE.read_text())
        value = str(raw.get("model_type", "")).strip('"')
    value = (value or "optimized").strip()
    if value not in ("optimized", "baseline"):
        raise ValueError(f"model_type must be 'optimized' or 'baseline', got {value!r}")
    return value


def metric_lines(metrics: dict[str, object]) -> list[str]:
    """``name=value`` lines in the exact shape the CloudWatch regexes expect (plain decimals)."""
    lines = []
    for name in METRIC_NAMES:
        value = metrics.get(name)
        if value is None:                # roc_auc is None when the test slice has one class only
            continue
        lines.append(f"{name}={int(value)}" if name == "rows" else f"{name}={float(value):.6f}")
    return lines


def studio_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """The same numbers in the layout SageMaker Studio shows on a model version's Quality tab."""
    def entry(value):
        return {"value": float(value), "standard_deviation": "NaN"}

    wanted = {"precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "auc", "pr_auc": "pr_auc"}
    return {"binary_classification_metrics": {
        studio: entry(metrics[ours]) for ours, studio in wanted.items() if metrics.get(ours) is not None}}


def upload_metrics(path: Path, uri: str, client=None) -> None:
    """Copy metrics.json to s3://... (the model registry points at this copy)."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not uri.startswith("s3://") or not bucket or not key:
        raise ValueError(f"METRICS_S3_URI must look like s3://bucket/key, got {uri!r}")
    if client is None:
        import boto3

        client = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    client.upload_file(str(path), bucket, key)


def add_app_package_to_path(env: dict[str, str]) -> str:
    """Make ``import financial_risk`` work and say where it came from.

    Order: FINRISK_APP_SRC (your laptop, the tests), then whatever PYTHONPATH already gives us (the
    image: PYTHONPATH=/app/src), then /app/src itself in case the platform dropped the image's ENV."""
    explicit = env.get("FINRISK_APP_SRC")
    if explicit:
        src = Path(explicit).resolve()
        if not (src / "financial_risk").is_dir():
            raise FileNotFoundError(f"FINRISK_APP_SRC={explicit} has no financial_risk/ package inside")
        sys.path.insert(0, str(src))
        return str(src)
    import importlib.util

    if importlib.util.find_spec("financial_risk") is None and (IMAGE_APP_SRC / "financial_risk").is_dir():
        sys.path.insert(0, str(IMAGE_APP_SRC))
    spec = importlib.util.find_spec("financial_risk")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("cannot import financial_risk: set FINRISK_APP_SRC to the app's src folder, "
                          "or run inside the training image (it is built FROM the app image)")
    return str(Path(next(iter(spec.submodule_search_locations))).parent)


def main(environ: dict[str, str] | None = None, s3_client=None) -> dict[str, object]:
    env = dict(os.environ if environ is None else environ)
    channel = Path(env.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    model_dir = Path(env.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(env.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    # Imported here, not at the top: pipeline/mlops.py imports METRIC_NAMES from this file and
    # must stay light (no pandas / scikit-learn) on a CI runner that only needs boto3.
    src = add_app_package_to_path(env)
    from financial_risk.models.training import load_transaction_features, train_model

    data_file = find_training_file(channel)
    model_type = model_type_setting(env)
    print(f"app package   : {src}")
    print(f"training data : {data_file} ({data_file.stat().st_size / 1024:.0f} KB)")
    print(f"model type    : {model_type}")

    model_path, metrics = train_model(data_file, model_dir, model_type=model_type)
    rows = int(len(load_transaction_features(data_file)))
    metrics["rows"] = rows
    metrics["training_job"] = env.get("TRAINING_JOB_NAME")
    metrics["trained_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics.update(studio_metrics(metrics))

    # The gold table travels WITH the model: the service scores /api/predict/<row_id> from it.
    shutil.copyfile(data_file, model_dir / "features.parquet")

    # Same metrics file in two places: inside the model (so it travels with it) and in the output folder.
    for folder in (model_dir, output_dir):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"model written : {model_path}")
    print(f"files in {model_dir} : {sorted(p.name for p in model_dir.iterdir())}")

    metrics_uri = env.get("METRICS_S3_URI")
    if metrics_uri:
        upload_metrics(model_dir / "metrics.json", metrics_uri, s3_client)
        print(f"metrics.json uploaded to {metrics_uri}")

    # Last, so the lines are easy to find in the log. SageMaker reads these.
    for line in metric_lines(metrics):
        print(line)
    return metrics


if __name__ == "__main__":
    main()

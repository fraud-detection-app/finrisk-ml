"""train/sagemaker_train.py: the entrypoint that runs inside the training image.

Tested the way SageMaker uses it: a folder with the training data goes in, a folder with a model comes out.
No AWS, no Docker. The real training code is the APP's (financial_risk.models.training), so these tests need
the app's source: set FINRISK_APP_SRC (CI checks the finrisk-app repo out for this), or keep the sibling
folder ../finrisk-app next to this repo, as in the course workspace. Without it the end-to-end tests skip.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mlops
import sagemaker_train

ROOT = Path(__file__).resolve().parents[1]


def find_app_src() -> Path | None:
    for candidate in (os.environ.get("FINRISK_APP_SRC"), ROOT.parent / "finrisk-app" / "src"):
        if candidate and (Path(candidate) / "financial_risk" / "models" / "training.py").is_file():
            return Path(candidate).resolve()
    return None


APP_SRC = find_app_src()
needs_app = pytest.mark.skipif(APP_SRC is None, reason="the app's src is not available: set FINRISK_APP_SRC")


@pytest.fixture(scope="module")
def gold(tmp_path_factory) -> Path:
    """The gold feature table, made the way the app's pipeline makes it (5,000 synthetic rows)."""
    if APP_SRC is None:
        pytest.skip("app src missing")
    sys.path.insert(0, str(APP_SRC))
    from financial_risk.data.paysim import generate_demo_data, ingest_to_parquet
    from financial_risk.features.sql_features import build_feature_tables

    base = tmp_path_factory.mktemp("gold")
    raw, bronze, silver = base / "raw.csv", base / "bronze.parquet", base / "silver.parquet"
    generate_demo_data(5000, raw)
    ingest_to_parquet(raw, bronze, silver)
    build_feature_tables(silver, base / "gold")
    return base / "gold" / "transaction_features.parquet"


def sm_env(tmp_path: Path, gold: Path, **extra: str) -> dict[str, str]:
    """The three folders SageMaker provides, as env vars, pointing into tmp_path."""
    channel = tmp_path / "input" / "data" / "train"
    channel.mkdir(parents=True, exist_ok=True)
    (channel / "train.parquet").write_bytes(gold.read_bytes())
    return {"SM_CHANNEL_TRAIN": str(channel), "SM_MODEL_DIR": str(tmp_path / "model"),
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"), "FINRISK_APP_SRC": str(APP_SRC), **extra}


@pytest.fixture(scope="module")
def trained(tmp_path_factory, gold):
    """One real run of the script as a subprocess, exactly like `python /opt/ml/code/sagemaker_train.py`."""
    tmp = tmp_path_factory.mktemp("sagemaker")
    env = sm_env(tmp, gold, MODEL_TYPE="baseline", TRAINING_JOB_NAME="finrisk-train-test")   # logistic regression: fast; same code path
    clean = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "DEEPSEEK_API_KEY")}
    done = subprocess.run([sys.executable, str(ROOT / "train" / "sagemaker_train.py")], cwd=tmp, text=True,
                          capture_output=True, env=clean | env | {"LLM_PROVIDER": "none"}, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return {"env": env, "stdout": done.stdout, "model_dir": Path(env["SM_MODEL_DIR"]), "output_dir": Path(env["SM_OUTPUT_DATA_DIR"])}


@needs_app
def test_writes_the_model_package_the_app_reads(trained, gold):
    files = sorted(p.name for p in trained["model_dir"].iterdir())
    assert files == ["features.parquet", "latest_model.joblib", "metrics.json", "model_v1.pkl"]      # baseline -> model_v1.pkl
    # features.parquet is a copy of the training table: the app scores /api/predict/<row_id> from it
    assert (trained["model_dir"] / "features.parquet").read_bytes() == gold.read_bytes()


@needs_app
def test_the_pointer_holds_a_file_name_not_a_path(trained):
    import joblib

    assert joblib.load(trained["model_dir"] / "latest_model.joblib") == {"model_path": "model_v1.pkl"}


@needs_app
def test_the_optimized_model_is_model_v2(tmp_path, gold):
    sagemaker_train.main(sm_env(tmp_path, gold, MODEL_TYPE="optimized"))
    assert (tmp_path / "model" / "model_v2.pkl").is_file() and not (tmp_path / "model" / "model_v1.pkl").exists()


@needs_app
def test_metrics_json_is_in_the_model_and_in_the_output_folder(trained):
    inside = json.loads((trained["model_dir"] / "metrics.json").read_text())
    outside = json.loads((trained["output_dir"] / "metrics.json").read_text())
    assert inside == outside
    for key in ("pr_auc", "roc_auc", "threshold", "precision", "recall", "rows", "model_type", "trained_at_utc", "training_job"):
        assert key in inside, key
    assert inside["rows"] == 5000 == inside["train_rows"] + inside["valid_rows"] + inside["test_rows"]
    assert 0.0 <= inside["pr_auc"] <= 1.0 and inside["model_type"] == "baseline"
    assert inside["training_job"] == "finrisk-train-test"
    assert inside["binary_classification_metrics"]["precision"]["value"] == pytest.approx(inside["precision"])


@needs_app
def test_prints_one_name_value_line_per_metric_and_the_pipeline_regexes_find_them(trained):
    """SageMaker turns log lines into metrics with the regexes `mlops.py train` registers."""
    metrics = json.loads((trained["model_dir"] / "metrics.json").read_text())
    for definition in mlops.metric_definitions():
        hits = re.findall(definition["Regex"], trained["stdout"])
        assert len(hits) == 1, f"{definition['Name']}: expected exactly one match, got {hits}"
        if metrics.get(definition["Name"]) is not None:
            assert float(hits[0]) == pytest.approx(float(metrics[definition["Name"]]), abs=1e-6)


@needs_app
def test_model_type_comes_from_the_environment(tmp_path, gold):
    assert sagemaker_train.main(sm_env(tmp_path, gold, MODEL_TYPE="baseline"))["model_type"] == "baseline"
    with pytest.raises(ValueError, match="model_type"):
        sagemaker_train.main(sm_env(tmp_path / "again", gold, MODEL_TYPE="deep-magic"))


@needs_app
def test_metrics_are_also_uploaded_when_the_job_says_where(tmp_path, gold):
    uploads = []

    class FakeS3:
        def upload_file(self, path, bucket, key):
            uploads.append((json.loads(Path(path).read_text()), bucket, key))

    sagemaker_train.main(
        sm_env(tmp_path, gold, MODEL_TYPE="baseline", TRAINING_JOB_NAME="finrisk-train-up",
               METRICS_S3_URI="s3://ml-bucket/metrics/finrisk-train-up/metrics.json"), s3_client=FakeS3())
    (sent, bucket, key), = uploads
    assert (bucket, key) == ("ml-bucket", "metrics/finrisk-train-up/metrics.json") and sent["training_job"] == "finrisk-train-up"


def test_a_wrong_app_source_folder_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="financial_risk"):
        sagemaker_train.add_app_package_to_path({"FINRISK_APP_SRC": str(tmp_path)})


def test_the_metrics_address_must_be_an_s3_address(tmp_path):
    with pytest.raises(ValueError, match="METRICS_S3_URI"):
        sagemaker_train.upload_metrics(tmp_path / "m.json", "https://example.com/m.json", client=object())


def test_hyperparameters_file_is_read_when_no_env_var(tmp_path, monkeypatch):
    hp = tmp_path / "hyperparameters.json"
    hp.write_text(json.dumps({"model_type": '"baseline"'}))          # SageMaker JSON-encodes every value
    monkeypatch.setattr(sagemaker_train, "HYPERPARAMETERS_FILE", hp)
    assert sagemaker_train.model_type_setting({}) == "baseline"
    assert sagemaker_train.model_type_setting({"MODEL_TYPE": "optimized"}) == "optimized"       # env wins
    monkeypatch.setattr(sagemaker_train, "HYPERPARAMETERS_FILE", tmp_path / "missing.json")
    assert sagemaker_train.model_type_setting({}) == "optimized"


def test_finding_the_training_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        sagemaker_train.find_training_file(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="found nothing"):
        sagemaker_train.find_training_file(tmp_path)
    (tmp_path / "a.parquet").write_bytes(b"x")
    assert sagemaker_train.find_training_file(tmp_path).name == "a.parquet"           # the only one
    (tmp_path / "b.parquet").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="train.parquet"):                     # ambiguous
        sagemaker_train.find_training_file(tmp_path)
    (tmp_path / "train.parquet").write_bytes(b"x")
    assert sagemaker_train.find_training_file(tmp_path).name == "train.parquet"


def test_metric_lines_skip_a_missing_roc_auc_and_print_rows_as_an_integer():
    lines = sagemaker_train.metric_lines({"pr_auc": 0.5, "roc_auc": None, "threshold": 0.1, "precision": 1.0, "recall": 0.5, "rows": 5000})
    assert "rows=5000" in lines and not any(line.startswith("roc_auc") for line in lines)
    assert "pr_auc=0.500000" in lines

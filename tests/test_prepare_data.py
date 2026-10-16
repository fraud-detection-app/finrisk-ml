"""prepare-data: the training image makes the gold table with the APP's pipeline; we upload it."""

from __future__ import annotations

from pathlib import Path

from fakes import BUCKET, IMAGE, last_line


def fake_pipeline(command):
    """What `docker run ... scripts/run_pipeline.py` would leave in the mounted folder."""
    mount = command[command.index("-v") + 1].split(":")[0]
    gold = Path(mount) / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    (gold / "transaction_features.parquet").write_bytes(b"gold-table")
    return 0


def test_prepare_data_runs_the_app_pipeline_in_the_training_image_and_uploads_train_parquet(world, tmp_path):
    code, out, err = world.run("prepare-data", "--image", IMAGE, "--stamp", "20261016-031500", "--workdir", str(tmp_path), docker=fake_pipeline)
    assert code == 0, err
    (command,) = world.docker_runs
    assert command[:3] == ["docker", "run", "--rm"] and IMAGE in command
    assert command[command.index("--entrypoint") + 1] == "python"            # the image's own entrypoint is the trainer
    assert command[-3:] == ["scripts/run_pipeline.py", "--demo-rows", "5000"]
    assert "DATA_MODE=demo" in command and "GOLD_PATH=/out/gold" in command and "LLM_PROVIDER=none" in command
    assert world.s3.uploads == [(str(tmp_path / "gold" / "transaction_features.parquet"), BUCKET, "data/20261016-031500/train.parquet")]
    assert "data version 20261016-031500" in out
    assert last_line(out) == f"s3://{BUCKET}/data/20261016-031500/"           # the one exact value, for `train --data`


def test_the_row_count_comes_from_the_config_and_can_be_overridden(world, tmp_path):
    world.run("prepare-data", "--image", IMAGE, "--workdir", str(tmp_path / "a"), "--rows", "800", docker=fake_pipeline)
    assert world.docker_runs[-1][-1] == "800"


def test_the_data_version_defaults_to_the_utc_time(world, tmp_path):
    code, out, err = world.run("prepare-data", "--image", IMAGE, "--workdir", str(tmp_path), docker=fake_pipeline)
    key = world.s3.uploads[0][2]
    assert code == 0 and key.startswith("data/") and key.endswith("/train.parquet") and len(key.split("/")[1]) == 15


def test_a_failing_pipeline_stops_before_any_upload(world, tmp_path):
    code, out, err = world.run("prepare-data", "--image", IMAGE, "--workdir", str(tmp_path), docker=lambda c: 125)
    assert code == 1 and "docker exit 125" in err and not world.s3.uploads


def test_a_pipeline_that_wrote_no_gold_table_is_reported(world, tmp_path):
    code, out, err = world.run("prepare-data", "--image", IMAGE, "--workdir", str(tmp_path), docker=lambda c: 0)
    assert code == 1 and "does not exist" in err and not world.s3.uploads


def test_the_image_can_come_from_the_environment(world, tmp_path):
    code, out, err = world.run("prepare-data", "--workdir", str(tmp_path), docker=fake_pipeline, env={"TRAIN_IMAGE": IMAGE})
    assert code == 0 and IMAGE in world.docker_runs[0]
    code, out, err = world.run("prepare-data", "--workdir", str(tmp_path))
    assert code == 1 and "--image" in err

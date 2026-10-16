"""The gate (`evaluate`): pass, fail on the floor, fail against the champion, no champion."""

from __future__ import annotations

import json

import pytest
from fakes import ROLE, World, last_line

import mlops


# ---------------------------------------------------------------- the pure logic


def verdict(candidate, champion=None, floor=0.70, tolerance=0.02):
    checks = mlops.gate_checks(candidate, floor, tolerance, champion)
    return all(c.passed for c in checks), checks


def test_passes_above_the_floor_when_there_is_no_champion():
    ok, checks = verdict(0.75)
    assert ok and [c.name for c in checks] == ["floor", "champion"] and checks[1].skipped


def test_fails_below_the_floor_even_with_no_champion():
    ok, checks = verdict(0.69)
    assert not ok and not checks[0].passed


def test_the_floor_itself_passes():
    assert verdict(0.70)[0]


def test_passes_when_close_enough_to_the_champion():
    assert verdict(0.79, champion=0.80)[0]                  # 0.01 worse: inside the 0.02 tolerance
    assert verdict(0.78, champion=0.80)[0]                  # exactly champion - tolerance still passes (float noise ignored)


def test_fails_when_worse_than_the_champion_by_more_than_the_tolerance():
    ok, checks = verdict(0.77, champion=0.80)
    assert not ok and checks[0].passed and not checks[1].passed


def test_a_much_better_candidate_passes_both():
    assert verdict(0.95, champion=0.80)[0]


def test_the_thresholds_come_from_the_config_file():
    cfg = mlops.load_train_config(mlops.DEFAULT_CONFIG)
    assert (cfg.floor, cfg.tolerance, cfg.gate_metric, cfg.champion_env) == (0.70, 0.02, "pr_auc", "prod")
    assert (cfg.instance_type, cfg.max_runtime_seconds, cfg.demo_rows) == ("ml.m5.large", 1800, 5000)


def test_a_missing_config_file_is_one_readable_error(world, tmp_path):
    code, out, err = world.run("list", "--config", str(tmp_path / "nope.yaml"))
    assert code == 1 and "config file not found" in err


# ---------------------------------------------------------------- the command


def trained(world, pr_auc, job="finrisk-train-g"):
    world.sm.pr_auc = pr_auc
    world.put_metrics(job)
    return world.trained_job(job)


def test_evaluate_passes_and_prints_the_table(world):
    job = trained(world, 0.8631)
    code, out, err = world.run("evaluate", "--training-job", job)
    assert code == 0, err
    assert "gate for finrisk-train-g" in out and "floor" in out and "PASS" in out and "skipped" in out
    assert last_line(out) == job                              # the value the next step (register) needs


def test_evaluate_fails_on_the_floor_with_exit_3(world, tmp_path):
    job = trained(world, 0.60)
    summary = tmp_path / "summary.md"
    code, out, err = world.run("evaluate", "--training-job", job, "--summary", str(summary))
    assert code == 3
    assert "GATE FAILED" in err and "floor" in err and "0.6000" in err
    text = summary.read_text()
    assert "Gate: FAILED" in text and "**fail**" in text and "Nothing was registered" in text
    assert not any(name == "create_model_package" for name, _ in world.sm.calls)


def test_evaluate_fails_against_the_champion(world, tmp_path):
    v = world.registered_version()                             # a registered champion with PR-AUC 0.8631 ...
    world.decide(v)
    world.set_pointer("prod", v)                               # ... running in prod
    job = trained(world, 0.80, job="finrisk-train-worse")     # 0.80 clears the floor but is 0.063 below the champion
    summary = tmp_path / "summary.md"
    code, out, err = world.run("evaluate", "--training-job", job, "--summary", str(summary))
    assert code == 3
    assert "champion v1 (0.8631)" in out and "FAIL" in out
    assert "champion v1 (0.8631) - 0.02" in err
    assert "version 1 in `prod`" in summary.read_text()


def test_evaluate_passes_against_a_champion_within_tolerance(world):
    v = world.registered_version()
    world.set_pointer("prod", v)
    job = trained(world, 0.85, job="finrisk-train-close")      # 0.0131 below: inside the tolerance
    code, out, err = world.run("evaluate", "--training-job", job)
    assert code == 0, err
    assert "PASS" in out


def test_no_champion_when_the_pointer_has_no_version_yet(world):
    world.ssm.params["/finrisk/prod/model"] = json.dumps({"version": None})
    job = trained(world, 0.75)
    code, out, err = world.run("evaluate", "--training-job", job)
    assert code == 0, err
    assert "skipped" in out


def test_the_champion_is_read_from_prods_pointer(world):
    job = trained(world, 0.75)
    world.run("evaluate", "--training-job", job)
    code, out, err = world.run("evaluate", "--training-job", job)
    assert "ssm.get_parameter" in out and "/finrisk/prod/model" in out


def test_a_champion_without_a_pr_auc_stops_the_gate_instead_of_guessing(world):
    v = world.registered_version()
    del world.sm.packages[v]["CustomerMetadataProperties"]["pr_auc"]
    del world.sm.packages[v]["ModelMetrics"]
    world.sm.packages[v]["ModelMetrics"] = {}
    world.set_pointer("prod", v)
    job = trained(world, 0.9, job="finrisk-train-x")
    code, out, err = world.run("evaluate", "--training-job", job)
    assert code == 1 and "cannot compare" in err


def test_evaluate_falls_back_to_metrics_json_when_cloudwatch_has_no_final_metric(world):
    job = trained(world, 0.91)
    original = world.sm.describe_training_job

    def no_final_metrics(TrainingJobName):                    # noqa: N803
        desc = original(TrainingJobName)
        desc.pop("FinalMetricDataList", None)
        return desc

    world.sm.describe_training_job = no_final_metrics
    code, out, err = world.run("evaluate", "--training-job", job)
    assert code == 0, err
    assert "reading s3://" in out and "0.9100" in out


def test_evaluate_refuses_a_job_that_is_not_completed(capsys):
    w = World(capsys, statuses=[("InProgress", "Training")])
    w.sm.create_training_job(**mlops.build_training_request(
        mlops.Config(ml_bucket="b", role_arn=ROLE), mlops.load_train_config(mlops.DEFAULT_CONFIG), "i@sha256:" + "0" * 64, "jj", "s3://b/d/"))
    code, out, err = w.run("evaluate", "--training-job", "jj")
    assert code == 1 and "not Completed" in err

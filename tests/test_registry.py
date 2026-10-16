"""register, approve, list, show: the Model Registry side of the pipeline."""

from __future__ import annotations

import json

from fakes import ACCOUNT, BUCKET, GROUP, IMAGE, World, check_shape, last_line


def test_register_creates_a_pending_version_with_everything_it_needs(world):
    version = world.registered_version(env={"GITHUB_SHA": "c" * 40})
    assert version == 1
    request = next(kw for name, kw in world.sm.calls if name == "create_model_package")
    check_shape("sagemaker", "CreateModelPackage", request)           # valid against the real AWS API definition
    assert request["ModelPackageGroupName"] == GROUP
    assert request["ModelApprovalStatus"] == "PendingManualApproval"
    container = request["InferenceSpecification"]["Containers"][0]
    assert container["Image"] == IMAGE and container["ModelDataUrl"].endswith("/output/model.tar.gz")
    assert request["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"].startswith(f"s3://{BUCKET}/metrics/")


def test_register_stores_where_the_model_came_from_and_how_good_it_is(world):
    world.registered_version()
    meta = next(kw for name, kw in world.sm.calls if name == "create_model_package")["CustomerMetadataProperties"]
    assert meta["train_image"] == IMAGE
    assert meta["train_image_digest"] == "sha256:" + "ab" * 32
    assert meta["data_uri"] == f"s3://{BUCKET}/data/20261014-030000/" and meta["data_version"] == "20261014-030000"
    assert meta["pr_auc"] == "0.863100" and meta["recall"] == "0.890000" and meta["rows"] == "5000"
    assert meta["training_job"].startswith("finrisk-train-")
    assert meta["ml_git_sha"] == "unknown"                               # train ran without GITHUB_SHA in this test
    assert all(len(v) <= 256 for v in meta.values())


def test_the_ml_git_sha_travels_from_the_training_job_to_the_registry(world):
    job = "finrisk-train-sha"
    world.put_data()
    world.run("train", "--image", IMAGE, "--job-name", job, "--data", f"s3://{BUCKET}/data/20261014-030000/",
              env={"GITHUB_SHA": "d" * 40})
    world.put_metrics(job)
    code, out, err = world.run("register", "--training-job", job)        # a different process, no GITHUB_SHA
    assert code == 0, err
    meta = next(kw for name, kw in world.sm.calls if name == "create_model_package")["CustomerMetadataProperties"]
    assert meta["ml_git_sha"] == "d" * 40


def test_register_numbers_versions_and_prints_the_number_last(world):
    assert world.registered_version() == 1
    assert world.registered_version() == 2


def test_register_writes_the_job_summary(world, tmp_path):
    job = world.trained_job("finrisk-train-sum")
    world.put_metrics(job)
    summary = tmp_path / "s.md"
    code, out, err = world.run("register", "--training-job", job, "--summary", str(summary))
    assert code == 0, err
    text = summary.read_text()
    assert "Registered: model version 1 (PendingManualApproval)" in text and "| pr_auc | 0.863100 |" in text
    assert ACCOUNT not in text


def test_register_without_the_metrics_file_says_so(world):
    job = world.trained_job()
    code, out, err = world.run("register", "--training-job", job)
    assert code == 1 and "metrics file" in err and "missing" in err
    assert not any(name == "create_model_package" for name, _ in world.sm.calls)


def test_register_refuses_a_job_that_did_not_complete(capsys):
    w = World(capsys, failure="boom")
    code, out, err = w.run("train", "--image", IMAGE, "--job-name", "bad", "--data", "s3://b/d/")
    assert code == 1
    code, out, err = w.run("register", "--training-job", "bad")
    assert code == 1 and "Failed" in err


# ---------------------------------------------------------------- approve / reject


def test_approve_and_reject_move_the_status_and_keep_the_note(world):
    v = world.registered_version()
    code, out, err = world.run("approve", "--version", str(v), "--decision", "approve", "--note", "reviewed the metrics")
    assert code == 0, err
    assert last_line(out) == "Approved"
    update = next(kw for name, kw in world.sm.calls if name == "update_model_package")
    check_shape("sagemaker", "UpdateModelPackage", update)
    assert update["ModelApprovalStatus"] == "Approved" and update["ApprovalDescription"] == "reviewed the metrics"
    code, out, err = world.run("approve", "--version", str(v), "--decision", "reject", "--note", "recall too low")
    assert code == 0 and last_line(out) == "Rejected"


def test_approve_defaults_to_approve(world):
    v = world.registered_version()
    code, out, err = world.run("approve", "--version", str(v))
    assert code == 0 and last_line(out) == "Approved"


def test_approving_a_version_that_does_not_exist_exits_2(world):
    code, out, err = world.run("approve", "--version", "9", "--decision", "approve")
    assert code == 2 and "not found" in err
    assert not any(name == "update_model_package" for name, _ in world.sm.calls)


# ---------------------------------------------------------------- list / show


def test_list_shows_versions_newest_first(world):
    world.registered_version()
    world.sm.pr_auc = 0.9
    world.registered_version()
    code, out, err = world.run("list")
    assert code == 0, err
    rows = [line for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert [r.split()[0] for r in rows] == ["2", "1"] and "PendingManualApproval" in rows[0]


def test_list_with_nothing_registered_says_what_to_do(world):
    code, out, err = world.run("list")
    assert code == 0 and "no versions" in out


def test_show_prints_the_provenance_with_ids_masked(world):
    v = world.registered_version()
    code, out, err = world.run("show", "--version", str(v))
    assert code == 0, err
    assert "train_image_digest" in out and "data_uri" in out and "ml_git_sha" in out
    assert ACCOUNT not in out


def test_quiet_prints_only_the_value(world):
    job = world.trained_job()
    world.put_metrics(job)
    code, out, err = world.run("register", "--training-job", job, "--quiet")
    assert code == 0 and out.strip() == "1"

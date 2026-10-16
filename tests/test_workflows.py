"""The four workflows and the repo's own files: names, chain order, the gate before register, no secrets echoed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
NAMES = ["ml-ci", "train", "approve-model", "promote-model"]
ENVS = ["dev", "sit", "pre-prod", "prod"]


def load(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text())
    data["on"] = data.pop(True, data.get("on"))              # YAML reads a bare `on:` as the boolean True
    return data


def text(name: str) -> str:
    return (WORKFLOWS / f"{name}.yml").read_text()


def steps(job: dict) -> list[dict]:
    return job["steps"]


def step_index(job: dict, needle: str) -> int:
    for i, step in enumerate(steps(job)):
        if needle in step.get("name", "") or needle in step.get("run", ""):
            return i
    raise AssertionError(f"no step mentioning {needle!r}")


def test_exactly_these_workflows_exist_and_parse():
    assert sorted(p.stem for p in WORKFLOWS.glob("*.yml")) == sorted(NAMES)
    for name in NAMES:
        assert load(name)["jobs"]


# ---------------------------------------------------------------- ml-ci


def test_ml_ci_runs_pytest_in_a_job_called_test_on_prs_and_main():
    wf = load("ml-ci")
    assert set(wf["on"]) == {"pull_request", "push"} and wf["on"]["push"]["branches"] == ["main"]
    assert list(wf["jobs"]) == ["test"] and wf["jobs"]["test"]["name"] == "test"
    assert any("pytest" in s.get("run", "") for s in steps(wf["jobs"]["test"]))
    assert "id-token" not in str(wf["permissions"]) and wf["permissions"] == {"contents": "read"}   # CI touches no AWS


def test_ml_ci_checks_out_the_app_so_the_training_test_can_run():
    job = load("ml-ci")["jobs"]["test"]
    repos = [s["with"].get("repository") for s in steps(job) if s.get("uses", "").startswith("actions/checkout") and "with" in s]
    assert "fraud-detection-app/finrisk-app" in repos
    pytest_step = next(s for s in steps(job) if "pytest" in s.get("run", ""))
    assert pytest_step["env"]["FINRISK_APP_SRC"].endswith("/finrisk-app/src")


# ---------------------------------------------------------------- train


def test_train_triggers():
    on = load("train")["on"]
    assert set(on) == {"workflow_dispatch", "push", "schedule"}
    assert on["push"]["branches"] == ["main"] and on["push"]["paths"] == ["train/**", "pipeline/**", "config/**"]
    assert re.fullmatch(r"[\d*/, -]+( [\d*/, -]+){4}", on["schedule"][0]["cron"])


def test_train_uses_the_ml_environment_and_its_role_secret():
    job = load("train")["jobs"]["train"]
    assert job["environment"] == "ml"
    assert job["permissions"]["id-token"] == "write"
    login = next(s for s in steps(job) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials"))
    assert login["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"
    assert login["with"]["aws-region"] == "${{ vars.AWS_REGION }}"


def test_train_masks_the_account_id_right_after_login():
    job = load("train")["jobs"]["train"]
    login = next(i for i, s in enumerate(steps(job)) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials"))
    assert "::add-mask::" in steps(job)[login + 1]["run"] and "sts get-caller-identity" in steps(job)[login + 1]["run"]


def test_train_steps_run_in_order_and_the_gate_comes_before_register():
    job = load("train")["jobs"]["train"]
    order = [step_index(job, n) for n in ("Training image", "prepare-data", "mlops.py train", "mlops.py evaluate", "mlops.py register")]
    assert order == sorted(order) and len(set(order)) == 5
    evaluate, register = steps(job)[order[3]], steps(job)[order[4]]
    assert "--training-job" in evaluate["run"] and "--summary" in evaluate["run"]
    for step in (evaluate, register):                      # a failed gate must stop the job: nothing may override it
        assert "continue-on-error" not in step and "if" not in step
    assert not any("if" in s and "always" in str(s["if"]) for s in steps(job)[order[3]:])


def test_train_builds_with_the_default_builder_and_skips_an_existing_image():
    body = text("train")
    assert "setup-buildx" not in body and "docker/build-push-action" not in body
    run = next(s for s in steps(load("train")["jobs"]["train"]) if "Training image" in s.get("name", ""))["run"]
    assert "describe-images" in run and "ImageNotFoundException" in run and "docker build" in run and "docker push" in run
    assert "finrisk-train" in run and "sha-${GITHUB_SHA}" in run
    assert "ecr get-login-password" in run              # logs in to ECR to pull the base image
    assert "0" * 64 in run                              # refuses the placeholder pin


def test_train_passes_the_image_by_digest():
    run = "\n".join(s.get("run", "") for s in steps(load("train")["jobs"]["train"]))
    assert "TRAIN_IMAGE=${REPO_URI}@${DIGEST}" in run and '--image "$TRAIN_IMAGE"' in run


# ---------------------------------------------------------------- approve


def test_approve_model_inputs_environment_and_role():
    wf = load("approve-model")
    inputs = wf["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"version", "decision", "note"} and inputs["decision"]["options"] == ["approve", "reject"]
    job = wf["jobs"]["decide"]
    assert job["environment"] == "model-approval"
    login = next(s for s in steps(job) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials"))
    assert login["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"
    run = "\n".join(s.get("run", "") for s in steps(job))
    assert "mlops.py approve" in run and "[0-9]+" in run     # the version is checked to be a number first


# ---------------------------------------------------------------- promote


def test_promote_model_chain_order_environments_and_per_env_roles():
    wf = load("promote-model")
    assert list(wf["jobs"]) == [f"promote-{e}" for e in ENVS]
    assert wf["on"]["workflow_dispatch"]["inputs"]["version"]["required"] is True
    previous = None
    for env in ENVS:
        job = wf["jobs"][f"promote-{env}"]
        assert job["environment"] == env
        assert job.get("needs") == previous                   # dev has no needs; every other job needs the one before it
        login = next(s for s in steps(job) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials"))
        assert login["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"        # that Environment's own secret
        assert any(f"mlops.py promote --env {env} --version" in s.get("run", "") for s in steps(job))
        previous = f"promote-{env}"


def test_every_role_assuming_job_has_an_environment_and_reads_the_one_secret_name():
    """The course rule: a job WITH `environment:` reads secrets.AWS_ROLE_ARN (an Environment secret)."""
    for name in ("train", "approve-model", "promote-model"):
        for job in load(name)["jobs"].values():
            logins = [s for s in steps(job) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials")]
            assert len(logins) == 1 and job.get("environment")
            assert logins[0]["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"
        assert set(re.findall(r"secrets\.(\w+)", text(name))) == {"AWS_ROLE_ARN"}
    assert not re.search(r"AWS_ML_\w*ROLE_ARN", "\n".join(text(n) for n in NAMES))      # the old per-role names are gone


def test_promote_masks_the_account_id_in_every_job():
    for job in load("promote-model")["jobs"].values():
        assert any("::add-mask::" in s.get("run", "") for s in steps(job))


# ---------------------------------------------------------------- every workflow


@pytest.mark.parametrize("name", NAMES)
def test_actions_are_pinned_to_the_course_majors(name):
    allowed = {"actions/checkout": "v7", "actions/setup-python": "v7", "aws-actions/configure-aws-credentials": "v6"}
    for action, version in re.findall(r"uses:\s*([\w./-]+)@([\w.]+)", text(name)):
        assert allowed.get(action) == version, (action, version)


@pytest.mark.parametrize("name", NAMES)
def test_no_secret_is_echoed_or_traced(name):
    body = text(name)
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        if "secrets." in code:
            assert "role-to-assume" in code, f"a secret used somewhere other than role-to-assume: {line.strip()}"
    assert "set -x" not in body and "env | sort" not in body and "printenv" not in body
    assert not re.search(r"echo[^\n]*(SECRET|TOKEN|PASSWORD)", body, re.I)
    assert "AWS_SECRET_ACCESS_KEY" not in body and "AWS_ACCESS_KEY_ID" not in body   # OIDC only: no long-lived keys


@pytest.mark.parametrize("name", NAMES)
def test_user_input_never_goes_straight_into_a_shell_script(name):
    """`${{ inputs.x }}` inside run: is script injection. Inputs reach scripts through env: only."""
    for job in load(name)["jobs"].values():
        for step in steps(job):
            assert "${{" not in step.get("run", "") or "inputs." not in step["run"], step.get("name")
            assert "github.event" not in step.get("run", "")


@pytest.mark.parametrize("name", ["train", "approve-model", "promote-model"])
def test_the_secrets_table_at_the_top_lists_exactly_the_names_the_file_uses(name):
    body = text(name)
    header = "\n".join(line for line in body.splitlines() if line.startswith("#"))
    used = set(re.findall(r"secrets\.(\w+)", body)) | set(re.findall(r"vars\.(\w+)", body))
    for variable in used:
        assert re.search(rf"^#\s+{variable}\b", header, re.M), f"{variable} is used but not in the table at the top of {name}.yml"


@pytest.mark.parametrize("name", NAMES)
def test_workflows_use_bash_and_least_privilege(name):
    wf = load(name)
    assert wf["defaults"]["run"]["shell"] == "bash"
    assert wf["permissions"] == {"contents": "read"}
    for job in wf["jobs"].values():
        assert job.get("permissions", {}).get("id-token", "none") in ("write", "none")
        assert "timeout-minutes" in job


# ---------------------------------------------------------------- the repo's own files


def test_version_is_semver_and_starts_at_0_1_0():
    assert (ROOT / "VERSION").read_text().strip() == "0.1.0"


def test_base_image_pin_is_a_digest_with_no_registry_address():
    lines = [l for l in (ROOT / "train" / "base_image.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(lines) == 1 and re.fullmatch(r"finrisk-app@sha256:[0-9a-f]{64}", lines[0])


def test_dockerfile_builds_from_the_pinned_base_and_sets_the_entrypoint():
    dockerfile = (ROOT / "train" / "Dockerfile").read_text()
    assert re.search(r"^ARG BASE_IMAGE$", dockerfile, re.M) and re.search(r"^FROM \$\{BASE_IMAGE\}$", dockerfile, re.M)
    assert "COPY train/sagemaker_train.py /opt/ml/code/sagemaker_train.py" in dockerfile
    assert "bumping" in (ROOT / "train" / "base_image.txt").read_text().lower() or "pull request" in (ROOT / "train" / "base_image.txt").read_text().lower()


def test_no_account_id_or_key_is_committed_anywhere():
    account = re.compile(r"(?<!\d)\d{12}(?!\d)")
    key = re.compile(r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git/" in str(path) or "__pycache__" in str(path) or ".pytest_cache" in str(path):
            continue
        if path.suffix in (".pyc", ".parquet", ".pkl") or path.name == "fakes.py" or path.name.startswith("test_"):
            continue                                        # the tests use a fake 123456789012 on purpose
        content = path.read_text(errors="ignore")
        assert not account.search(content), f"12-digit number in {path.relative_to(ROOT)}"
        assert not key.search(content), f"key-like string in {path.relative_to(ROOT)}"


def test_gitignore_keeps_models_data_and_secrets_out():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in ("*.pkl", "*.parquet", ".env"):
        assert pattern in ignored

"""promote: approval rules, the pointer JSON, and the readiness poll."""

from __future__ import annotations

import http.server
import json
import threading
from datetime import datetime, timezone

import pytest
from fakes import ACCOUNT, APP_URL, BUCKET, FakeApp, NOW, check_shape, last_line, ready

import mlops


def registered(world, approve=False):
    v = world.registered_version()
    if approve:
        world.decide(v)
    return v


# ---------------------------------------------------------------- who may run what


@pytest.mark.parametrize("env", ["pre-prod", "prod"])
def test_pre_prod_and_prod_refuse_a_pending_version(world, env):
    v = registered(world)
    world.set_app_url(env)
    code, out, err = world.run("promote", "--env", env, "--version", str(v))
    assert code == 2
    assert "PendingManualApproval" in err and "not Approved" in err
    assert not world.ssm.puts                                        # nothing was written


@pytest.mark.parametrize("env", ["dev", "sit"])
def test_dev_and_sit_accept_a_pending_version(world, env):
    v = registered(world)
    world.set_app_url(env)
    world.app = FakeApp([ready(v)])
    code, out, err = world.run("promote", "--env", env, "--version", str(v))
    assert code == 0, err
    assert json.loads(world.ssm.params[f"/finrisk/{env}/model"])["status"] == "PendingManualApproval"


@pytest.mark.parametrize("env", ["dev", "sit", "pre-prod", "prod"])
def test_a_rejected_version_is_never_accepted(world, env):
    v = registered(world)
    world.decide(v, "reject")
    world.set_app_url(env)
    code, out, err = world.run("promote", "--env", env, "--version", str(v))
    assert code == 2 and "Rejected" in err and not world.ssm.puts


@pytest.mark.parametrize("env", ["pre-prod", "prod"])
def test_pre_prod_and_prod_accept_an_approved_version(world, env):
    v = registered(world, approve=True)
    world.set_app_url(env)
    code, out, err = world.run("promote", "--env", env, "--version", str(v), env={"GITHUB_SHA": "x"})
    assert code == 0, err
    assert world.ssm.puts[0]["Name"] == f"/finrisk/{env}/model"


def test_an_unknown_version_exits_2(world):
    world.set_app_url("dev")
    code, out, err = world.run("promote", "--env", "dev", "--version", "7")
    assert code == 2 and "not found" in err


def test_an_unknown_environment_is_refused_by_the_parser(world):
    with pytest.raises(SystemExit):
        world.run("promote", "--env", "staging", "--version", "1")


# ---------------------------------------------------------------- the pointer


def test_the_pointer_json_is_exactly_the_contract(world):
    v = registered(world, approve=True)
    world.set_app_url("prod")
    env = {"GITHUB_SERVER_URL": "https://github.com", "GITHUB_REPOSITORY": "fraud-detection-app/finrisk-ml", "GITHUB_RUN_ID": "99"}
    code, out, err = world.run("promote", "--env", "prod", "--version", str(v), env=env)
    assert code == 0, err
    (put,) = world.ssm.puts
    check_shape("ssm", "PutParameter", put)
    assert put["Overwrite"] is True and put["Name"] == "/finrisk/prod/model" and put["Type"] == "String"
    pointer = json.loads(put["Value"])
    assert list(pointer) == ["version", "model_data_url", "status", "promoted_at", "run"]
    assert pointer == {
        "version": 1,                                                # a JSON number, not a string
        "model_data_url": f"s3://{BUCKET}/models/finrisk-train-01/output/model.tar.gz",
        "status": "Approved",
        "promoted_at": "2026-10-16T03:00:00+00:00",
        "run": "https://github.com/fraud-detection-app/finrisk-ml/actions/runs/99",
    }
    assert isinstance(pointer["version"], int)


def test_run_is_local_outside_github(world):
    v = registered(world)
    world.set_app_url("dev")
    world.app = FakeApp([ready(v)])
    world.run("promote", "--env", "dev", "--version", str(v))
    assert json.loads(world.ssm.puts[0]["Value"])["run"] == "local"


def test_promoting_an_older_version_is_a_rollback_with_the_same_command(world):
    v1, v2 = registered(world, approve=True), registered(world, approve=True)
    world.set_app_url("prod")
    world.app = FakeApp([ready(v2)])
    assert world.run("promote", "--env", "prod", "--version", str(v2))[0] == 0
    world.app = FakeApp([ready(v1)])
    code, out, err = world.run("promote", "--env", "prod", "--version", str(v1))
    assert code == 0, err
    assert json.loads(world.ssm.params["/finrisk/prod/model"])["version"] == v1


def test_the_app_url_is_looked_up_before_anything_is_written(world):
    v = registered(world)
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v))     # no /finrisk/dev/app_url
    assert code == 1 and "app_url" in err and "not deployed" in err
    assert not world.ssm.puts


def test_no_wait_only_moves_the_pointer(world):
    v = registered(world)
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), "--no-wait")
    assert code == 0 and world.ssm.puts and not world.app.urls


# ---------------------------------------------------------------- the readiness poll


def test_the_poll_waits_until_ready_reports_the_new_version(world):
    v = registered(world)
    world.set_app_url("dev")
    app = FakeApp([ready(None), ready(0), (503, {"ready": False, "model": {"version": None}}), ready(v)])
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=app)
    assert code == 0, err
    assert len(app.urls) == 4 and all(u == f"{APP_URL}/ready" for u in app.urls)
    polls = [line for line in out.splitlines() if "GET /ready" in line]
    assert len(polls) == 4 and "(want 1)" in polls[0] and "-> 503" in polls[2]      # every poll is printed
    assert "dev serves model version 1" in out and last_line(out) == "1"


def test_the_poll_accepts_a_version_reported_as_text(world):
    v = registered(world)
    world.set_app_url("dev")
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=FakeApp([ready("1")]))
    assert code == 0, err


def test_the_poll_times_out_after_ten_minutes(world):
    v = registered(world)
    world.set_app_url("dev")
    app = FakeApp([ready(0)])                                         # the old version, forever
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=app)
    assert code == 1 and "did not report model version 1 within 600 s" in err
    assert len(app.urls) == 42                                        # looks at t = 0, 15, ... 615 s: the first one past 600 s gives up
    assert world.ssm.puts                                             # the pointer WAS written: the message says what to do


def test_a_network_error_is_not_fatal_until_the_timeout(world):
    v = registered(world)
    world.set_app_url("dev")
    app = FakeApp([OSError("connection refused"), TimeoutError("cold start"), ready(v)])
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=app)
    assert code == 0, err and "no usable answer" in out


def test_a_load_error_from_the_app_fails_the_promotion_with_its_text(world):
    v = registered(world)
    world.set_app_url("dev")
    bad = (200, {"ready": True, "model": {"version": 0, "source": "baked"}, "load_error": "model download failed: AccessDenied",
                 "pointers_checked_at": "2026-10-16T03:00:30+00:00"})
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=FakeApp([ready(0), bad]))
    assert code == 1
    assert "could not load model version 1" in err and "model download failed: AccessDenied" in err


def test_an_old_error_from_before_the_pointer_moved_is_not_a_verdict(world):
    v = registered(world)
    world.set_app_url("dev")
    stale = (200, {"model": {"version": 0}, "load_error": "old problem", "pointers_checked_at": "2026-10-16T02:00:00+00:00"})
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=FakeApp([stale, stale, ready(v)]))
    assert code == 0, err                                             # checked an hour BEFORE the pointer was written


def test_the_error_can_live_inside_model(world):
    v = registered(world)
    world.set_app_url("dev")
    bad = (200, {"model": {"version": 0, "error": "unpickling failed"}})
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=FakeApp([bad]))
    assert code == 1 and "unpickling failed" in err


def test_an_index_problem_is_not_blamed_on_the_model(world):
    """The real /ready puts the model's own error in model.load_error; the top-level load_error may be about the index."""
    v = registered(world)
    world.set_app_url("dev")
    index_broken = (200, {"ready": True, "model": {"version": 0, "source": "s3", "load_error": None},
                          "index": {"version": "x", "load_error": "index download failed: AccessDenied"},
                          "load_error": "index download failed: AccessDenied", "pointers_checked_at": "2026-10-16T03:00:30+00:00"})
    code, out, err = world.run("promote", "--env", "dev", "--version", str(v), app=FakeApp([index_broken, ready(v)]))
    assert code == 0, err                                             # waited on, then saw its own version
    assert mlops.find_load_error(index_broken[1]) == ""
    assert mlops.find_load_error({"model": {"load_error": "boom"}, "load_error": "index problem"}) == "boom"


def test_find_load_error_and_parse_time():
    assert mlops.find_load_error({"load_error": "a"}) == "a"
    assert mlops.find_load_error({"model": {"load_error": "b"}}) == "b"
    assert mlops.find_load_error({"model": {"version": 1}}) == ""
    assert mlops.parse_time("2026-10-16T03:00:00Z") == datetime(2026, 10, 16, 3, tzinfo=timezone.utc)
    assert mlops.parse_time(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert mlops.parse_time("garbage") is None


def test_promote_writes_the_job_summary_without_account_ids(world, tmp_path):
    v = registered(world)
    world.set_app_url("dev")
    world.app = FakeApp([ready(v)])
    summary = tmp_path / "s.md"
    world.run("promote", "--env", "dev", "--version", str(v), "--summary", str(summary))
    assert "**dev**: model version 1" in summary.read_text()


# ---------------------------------------------------------------- the real HTTP path (urllib against a local server)


class _Handler(http.server.BaseHTTPRequestHandler):
    answers: list = []

    def do_GET(self):                                                 # noqa: N802
        status, body = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_app():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_http_get_returns_error_statuses_as_answers(local_app):
    server, url = local_app
    _Handler.answers = [(503, {"ready": False}), (200, {"ready": True})]
    assert mlops.http_get(url + "/ready", 5) == (503, '{"ready": false}')
    assert mlops.http_get(url + "/ready", 5)[0] == 200
    with pytest.raises(ValueError):
        mlops.http_get("file:///etc/passwd", 5)


def test_a_full_promotion_against_a_real_local_server(world, local_app):
    server, url = local_app
    v = registered(world)
    world.ssm.params["/finrisk/dev/app_url"] = url
    _Handler.answers = [(503, {"model": {"version": None}}), (200, {"model": {"version": v, "source": "s3"}})]
    code = mlops.main(["promote", "--env", "dev", "--version", str(v)], clients=world.clients,
                      env={**__import__("fakes").ENV}, out=lambda line: None, sleep=lambda s: None, now=lambda: NOW)
    assert code == 0

import sys
import time

from fastapi.testclient import TestClient

from serve.control_plane import CONTROL_MANAGER, _quote_shell
from serve.server_cpu import app


def _login(client: TestClient) -> None:
    response = client.post("/api/login", json={"password": "test-password"})
    assert response.status_code == 200


def _wait_for_terminal_state(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, object]:
    deadline = time.time() + timeout
    last_payload: dict[str, object] | None = None
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        last_payload = response.json()
        status = last_payload["job"]["status"]
        if status in {"completed", "failed", "stopped"}:
            return last_payload
        time.sleep(0.1)
    raise AssertionError(f"Job {job_id} did not finish in time: {last_payload}")


def test_control_plane_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_WEB_PASSWORD", "test-password")
    CONTROL_MANAGER.reset_for_tests()
    client = TestClient(app)
    response = client.get("/api/commands/presets")
    assert response.status_code == 401


def test_login_and_html_index(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_WEB_PASSWORD", "test-password")
    CONTROL_MANAGER.reset_for_tests()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "SAGE Remote Control" in response.text

    bad = client.post("/api/login", json={"password": "wrong"})
    assert bad.status_code == 401

    _login(client)
    response = client.get("/api/commands/presets")
    assert response.status_code == 200
    payload = response.json()
    preset_ids = {item["id"] for item in payload["presets"]}
    assert "data_bootstrap" in preset_ids
    assert "data_pipeline" in preset_ids
    assert "serve_cpu" in preset_ids
    assert "git_status" in preset_ids


def test_preset_job_launch_and_logs(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_WEB_PASSWORD", "test-password")
    CONTROL_MANAGER.reset_for_tests()
    client = TestClient(app)
    _login(client)

    response = client.post("/api/commands/run", json={"preset_id": "git_status"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "job"
    detail = _wait_for_terminal_state(client, payload["job"]["id"])
    assert detail["job"]["status"] == "completed"
    assert any(line for line in detail["logs"])


def test_raw_command_job_and_sse(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_WEB_PASSWORD", "test-password")
    CONTROL_MANAGER.reset_for_tests()
    client = TestClient(app)
    _login(client)

    command = "Write-Output alpha" if sys.platform.startswith("win") else "printf 'alpha\n'"
    response = client.post("/api/commands/run", json={"command": command})
    assert response.status_code == 200
    job_id = response.json()["job"]["id"]

    with client.stream("GET", f"/api/jobs/{job_id}/stream") as stream:
        body = "".join(stream.iter_text())
    assert "event: log" in body
    assert "alpha" in body
    assert "event: status" in body

    detail = _wait_for_terminal_state(client, job_id)
    assert detail["job"]["exit_code"] == 0


def test_stop_long_running_job(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_WEB_PASSWORD", "test-password")
    CONTROL_MANAGER.reset_for_tests()
    client = TestClient(app)
    _login(client)

    script = "import time; print('start', flush=True); time.sleep(30)"
    command = f"{_quote_shell(sys.executable)} -c {_quote_shell(script)}"
    response = client.post("/api/commands/run", json={"command": command})
    assert response.status_code == 200
    job_id = response.json()["job"]["id"]

    stop_response = client.post(f"/api/jobs/{job_id}/stop")
    assert stop_response.status_code == 200

    detail = _wait_for_terminal_state(client, job_id)
    assert detail["job"]["status"] == "stopped"


def test_health_api_preset(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_WEB_PASSWORD", "test-password")
    CONTROL_MANAGER.reset_for_tests()
    client = TestClient(app)
    _login(client)

    response = client.post("/api/commands/run", json={"preset_id": "health_check"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "api"
    assert payload["result"]["status"] == "ok"


def test_required_preset_field_validation(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_WEB_PASSWORD", "test-password")
    CONTROL_MANAGER.reset_for_tests()
    client = TestClient(app)
    _login(client)

    response = client.post("/api/commands/run", json={"preset_id": "tokenizer_train", "args": {"input_paths": ""}})
    assert response.status_code == 400
    assert "Input Paths" in response.json()["detail"]

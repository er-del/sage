from fastapi.testclient import TestClient

from serve.server import app as gpu_app
from serve.server_cpu import app as cpu_app


def test_gpu_server_health() -> None:
    client = TestClient(gpu_app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_cpu_server_health() -> None:
    client = TestClient(cpu_app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"

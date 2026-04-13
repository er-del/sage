from fastapi.testclient import TestClient

from serve.server import app as gpu_app
import serve.server as gpu_server
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


def test_gpu_server_generate(monkeypatch) -> None:
    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def to(self, _device) -> "FakeModel":
            return self

        def __call__(self, input_ids, past_key_values=None):
            import torch

            batch, seq = input_ids.shape
            logits = torch.zeros((batch, seq, 8), dtype=torch.float32)
            logits[:, :, 3] = 1.0
            cache = [] if past_key_values is None else past_key_values
            return logits, cache

    monkeypatch.setattr(gpu_server, "get_model", lambda: FakeModel())
    monkeypatch.setattr(gpu_server.torch.cuda, "is_available", lambda: False)
    client = TestClient(gpu_app)
    response = client.post("/generate", json={"input_ids": [1, 2], "max_new_tokens": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["tokens"] == [1, 2, 3, 3, 3]

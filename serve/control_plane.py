"""Minimal remote control plane for the SAGE FastAPI server."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import string
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_INDEX = REPO_ROOT / "serve" / "static" / "index.html"
SESSION_COOKIE = "sage_session"
SESSION_AGE_SECONDS = 60 * 60 * 12
PASSWORD_LENGTH = 12
_RUNTIME_PASSWORD: str | None = None
_RUNTIME_LOCAL_URL: str | None = None
_RUNTIME_PUBLIC_URL: str | None = None


@dataclass(frozen=True)
class PresetField:
    """One UI field for a preset action."""

    name: str
    label: str
    kind: str = "text"
    default: Any = ""
    placeholder: str = ""
    required: bool = False


@dataclass(frozen=True)
class CommandPreset:
    """One preset exposed in the browser UI."""

    identifier: str
    label: str
    description: str
    mode: str
    fields: tuple[PresetField, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "label": self.label,
            "description": self.description,
            "mode": self.mode,
            "fields": [
                {
                    "name": field.name,
                    "label": field.label,
                    "kind": field.kind,
                    "default": field.default,
                    "placeholder": field.placeholder,
                    "required": field.required,
                }
                for field in self.fields
            ],
        }


class LoginRequest(BaseModel):
    """Login payload for the control UI."""

    password: str


class RunCommandRequest(BaseModel):
    """Run either a preset action or a raw shell command."""

    preset_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    command: str | None = None
    cwd: str | None = None


@dataclass
class CommandJob:
    """One tracked subprocess job."""

    identifier: str
    label: str
    mode: str
    command: str
    cwd: str
    status: str = "running"
    exit_code: int | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    stop_requested: bool = False
    process: subprocess.Popen[str] | None = None
    logs: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 0
    condition: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        with self.condition:
            self.events.append({"id": self.next_event_id, "event": event, "data": payload})
            self.next_event_id += 1
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        clean = line.rstrip("\n")
        self.logs.append(clean)
        self.emit("log", {"line": clean})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "label": self.label,
            "mode": self.mode,
            "command": self.command,
            "cwd": self.cwd,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "log_lines": len(self.logs),
        }


def _quote_shell(value: str) -> str:
    if os.name == "nt":
        return "'" + value.replace("'", "''") + "'"
    return shlex.quote(value)


def _split_multi_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).replace(",", "\n")
    return [item.strip() for item in text.splitlines() if item.strip()]


def _build_presets(enable_generate: bool) -> list[CommandPreset]:
    presets = [
        CommandPreset(
            "health_check",
            "Health Check",
            "Call the local /health API and show the JSON response.",
            "api",
        ),
        CommandPreset(
            "data_bootstrap",
            "Bootstrap Dataset",
            "Create small JSONL corpora under data/raw for tokenizer and smoke-training runs.",
            "job",
            (
                PresetField("output_dir", "Output Dir", default="data/raw"),
                PresetField("overwrite", "Overwrite Existing Files", kind="boolean", default=False),
            ),
        ),
        CommandPreset(
            "data_pipeline",
            "Build Data Shards",
            "Filter raw JSONL corpora, deduplicate them, then write parquet shards with the trained tokenizer.",
            "job",
            (
                PresetField("tokenizer_model", "Tokenizer Model", default="tokenizer/tokenizer.model"),
                PresetField("output_dir", "Output Dir", default="data/processed"),
                PresetField(
                    "sources",
                    "Sources",
                    kind="textarea",
                    placeholder="general_web\ncode\nmath_science\nmultilingual\nsynthetic",
                ),
                PresetField("shard_size", "Shard Size", kind="number", default=2048),
                PresetField("limit_per_source", "Limit Per Source", kind="number", default=0),
            ),
        ),
        CommandPreset(
            "serve_gpu",
            "Serve GPU",
            "Start the GPU-oriented FastAPI server with uvicorn.",
            "job",
            (
                PresetField("host", "Host", default="0.0.0.0"),
                PresetField("port", "Port", kind="number", default=8000),
            ),
        ),
        CommandPreset(
            "serve_cpu",
            "Serve CPU",
            "Start the CPU-oriented FastAPI server with uvicorn.",
            "job",
            (
                PresetField("host", "Host", default="0.0.0.0"),
                PresetField("port", "Port", kind="number", default=8001),
            ),
        ),
        CommandPreset(
            "tokenizer_train",
            "Tokenizer Train",
            "Train the SentencePiece tokenizer from plain-text corpora.",
            "job",
            (
                PresetField(
                    "input_paths",
                    "Input Paths",
                    kind="textarea",
                    placeholder="data/raw/general_web.jsonl\ndata/raw/code.jsonl",
                    required=True,
                ),
                PresetField("model_prefix", "Model Prefix", default="tokenizer/tokenizer"),
                PresetField("vocab_size", "Vocab Size", kind="number", default=50000),
                PresetField("training_text", "Training Text", default="tokenizer/training_corpus.txt"),
            ),
        ),
        CommandPreset(
            "tokenizer_validate",
            "Tokenizer Validate",
            "Run the tokenizer smoke validation suite.",
            "job",
            (PresetField("model_path", "Model Path", default="tokenizer/tokenizer.model"),),
        ),
        CommandPreset(
            "training_run",
            "Training Run",
            "Launch the trainer with explicit shard and config paths.",
            "job",
            (
                PresetField("model_config", "Model Config", default="configs/model/1b.yaml"),
                PresetField("schedule_config", "Schedule Config", default="configs/train/schedule.yaml"),
                PresetField(
                    "train_shards",
                    "Train Shards",
                    kind="textarea",
                    placeholder="data/processed/shard-00000.parquet",
                    required=True,
                ),
                PresetField(
                    "validation_shards",
                    "Validation Shards",
                    kind="textarea",
                    placeholder="data/processed/shard-00001.parquet",
                ),
                PresetField("output_dir", "Output Dir", default="runs/default"),
                PresetField("steps", "Steps", kind="number", default=20),
                PresetField("disable_wandb", "Disable W&B", kind="boolean", default=True),
            ),
        ),
        CommandPreset(
            "eval_run",
            "Eval Run",
            "Run the registered eval benchmarks.",
            "job",
        ),
        CommandPreset(
            "git_status",
            "Git Status",
            "Show the current repository status.",
            "job",
        ),
        CommandPreset(
            "git_commit_push",
            "Git Add Commit Push",
            "Add selected paths, create a commit, and push it to the remote branch.",
            "shell",
            (
                PresetField(
                    "paths",
                    "Paths",
                    kind="textarea",
                    placeholder="serve\nserve/static\ntests\ntest.ipynb\nREADME.md",
                    required=True,
                ),
                PresetField("commit_message", "Commit Message", placeholder="feat: add control UI", required=True),
                PresetField("remote", "Remote", default="origin"),
                PresetField("branch", "Branch", default="main"),
            ),
        ),
        CommandPreset(
            "hf_sync",
            "Hugging Face Sync",
            "Push the current folder contents to the configured Hugging Face repo.",
            "job",
        ),
    ]
    if enable_generate:
        presets.insert(
            1,
            CommandPreset(
                "generate",
                "Generate",
                "Call the local /generate API and show the token output.",
                "api",
                (
                    PresetField("input_ids", "Input IDs", kind="json", default=[1, 42, 99]),
                    PresetField("max_new_tokens", "Max New Tokens", kind="number", default=8),
                ),
            ),
        )
    return presets


class CommandManager:
    """Track subprocess commands and expose their logs."""

    def __init__(self) -> None:
        self._jobs: dict[str, CommandJob] = {}
        self._lock = threading.Lock()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.started_at, reverse=True)
            return [job.to_dict() for job in jobs]

    def get_job(self, job_id: str) -> CommandJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def reset_for_tests(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            process = job.process
            if process is not None and process.poll() is None:
                try:
                    self._terminate_process(process)
                except Exception:
                    pass
        with self._lock:
            self._jobs.clear()

    def start_job(self, label: str, command: list[str] | str, cwd: str, mode: str) -> CommandJob:
        cwd_path = self._resolve_cwd(cwd)
        shell = isinstance(command, str)
        popen_command = self._build_shell_command(command) if shell else list(command)
        rendered = self._render_command(command)
        process = subprocess.Popen(
            popen_command,
            cwd=str(cwd_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **self._process_group_kwargs(),
        )
        job = CommandJob(identifier=str(uuid4()), label=label, mode=mode, command=rendered, cwd=str(cwd_path), process=process)
        job.emit("status", {"status": "running"})
        with self._lock:
            self._jobs[job.identifier] = job
        threading.Thread(target=self._read_output, args=(job,), daemon=True).start()
        return job

    def stop_job(self, job_id: str) -> CommandJob:
        job = self.get_job(job_id)
        process = job.process
        if process is None or process.poll() is not None:
            return job
        job.stop_requested = True
        job.status = "stopping"
        job.emit("status", {"status": "stopping"})
        self._terminate_process(process)
        threading.Thread(target=self._force_kill_if_needed, args=(job,), daemon=True).start()
        return job

    def _resolve_cwd(self, cwd: str) -> Path:
        if not cwd:
            return REPO_ROOT
        requested = Path(cwd)
        if not requested.is_absolute():
            requested = REPO_ROOT / requested
        return requested.resolve()

    def _build_shell_command(self, command: str) -> list[str]:
        if os.name == "nt":
            return ["powershell", "-Command", command]
        shell = "bash" if shutil.which("bash") else "sh"
        return [shell, "-lc", command]

    def _render_command(self, command: list[str] | str) -> str:
        if isinstance(command, str):
            return command
        if os.name == "nt":
            return subprocess.list2cmdline(command)
        return shlex.join(command)

    def _process_group_kwargs(self) -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            process.terminate()
            return
        os.killpg(process.pid, signal.SIGTERM)

    def _kill_process(self, process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            process.kill()
            return
        os.killpg(process.pid, signal.SIGKILL)

    def _force_kill_if_needed(self, job: CommandJob) -> None:
        process = job.process
        if process is None:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill_process(process)

    def _read_output(self, job: CommandJob) -> None:
        process = job.process
        if process is None:
            return
        stream = process.stdout
        if stream is not None:
            for line in iter(stream.readline, ""):
                if line == "":
                    break
                job.append_log(line)
        return_code = process.wait()
        job.exit_code = return_code
        job.ended_at = time.time()
        if job.stop_requested:
            job.status = "stopped"
        elif return_code == 0:
            job.status = "completed"
        else:
            job.status = "failed"
        job.emit("status", {"status": job.status, "exit_code": return_code})


CONTROL_MANAGER = CommandManager()


def _get_password() -> str | None:
    global _RUNTIME_PASSWORD
    if _RUNTIME_PASSWORD is None:
        alphabet = string.ascii_letters + string.digits
        _RUNTIME_PASSWORD = "".join(secrets.choice(alphabet) for _ in range(PASSWORD_LENGTH))
    return _RUNTIME_PASSWORD


def get_runtime_access_info() -> dict[str, str | None]:
    """Return the current runtime login password and access URLs."""
    return {
        "password": _get_password(),
        "local_url": _RUNTIME_LOCAL_URL,
        "public_url": _RUNTIME_PUBLIC_URL,
    }


def set_runtime_access_urls(local_url: str | None = None, public_url: str | None = None) -> None:
    """Record the URLs that should be shown in the startup banner."""
    global _RUNTIME_LOCAL_URL, _RUNTIME_PUBLIC_URL
    _RUNTIME_LOCAL_URL = local_url
    _RUNTIME_PUBLIC_URL = public_url


def _get_signing_secret() -> str:
    return os.environ.get("SAGE_WEB_SECRET") or _get_password() or "sage-control-plane"


def _encode_cookie_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).decode("ascii")
    digest = hmac.new(_get_signing_secret().encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return f"{body}.{digest}"


def _decode_cookie_payload(token: str | None) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    body, signature = token.split(".", 1)
    try:
        raw = base64.urlsafe_b64decode(body.encode("ascii"))
    except Exception:
        return None
    expected = hmac.new(_get_signing_secret().encode("utf-8"), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    payload = json.loads(raw.decode("utf-8"))
    issued_at = float(payload.get("iat", 0))
    if time.time() - issued_at > SESSION_AGE_SECONDS:
        return None
    return payload


def _require_session(request: Request) -> dict[str, Any]:
    payload = _decode_cookie_payload(request.cookies.get(SESSION_COOKIE))
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return payload


def _parse_number(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _api_response(handler: Callable[[dict[str, Any]], dict[str, Any]], args: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "api", "result": handler(args)}


def _validate_preset_args(preset: CommandPreset, args: dict[str, Any]) -> None:
    missing: list[str] = []
    for field in preset.fields:
        if not field.required:
            continue
        value = args.get(field.name)
        if value is None:
            missing.append(field.label)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field.label)
            continue
        if isinstance(value, list) and not value:
            missing.append(field.label)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")


def _build_command_for_preset(preset_id: str, args: dict[str, Any]) -> list[str] | str:
    if preset_id == "data_bootstrap":
        command = [sys.executable, "-m", "data.bootstrap", "--output-dir", str(args.get("output_dir") or "data/raw")]
        if bool(args.get("overwrite", False)):
            command.append("--overwrite")
        return command
    if preset_id == "data_pipeline":
        command = [
            sys.executable,
            "-m",
            "data.pipeline",
            "--tokenizer-model",
            str(args.get("tokenizer_model") or "tokenizer/tokenizer.model"),
            "--output-dir",
            str(args.get("output_dir") or "data/processed"),
            "--shard-size",
            str(_parse_number(args.get("shard_size"), 2048)),
        ]
        sources = _split_multi_value(args.get("sources"))
        if sources:
            command.extend(["--sources", *sources])
        limit_per_source = _parse_number(args.get("limit_per_source"), 0)
        if limit_per_source > 0:
            command.extend(["--limit-per-source", str(limit_per_source)])
        return command
    if preset_id == "serve_gpu":
        return [
            sys.executable,
            "-m",
            "uvicorn",
            "serve.server:app",
            "--host",
            str(args.get("host") or "0.0.0.0"),
            "--port",
            str(_parse_number(args.get("port"), 8000)),
        ]
    if preset_id == "serve_cpu":
        return [
            sys.executable,
            "-m",
            "uvicorn",
            "serve.server_cpu:app",
            "--host",
            str(args.get("host") or "0.0.0.0"),
            "--port",
            str(_parse_number(args.get("port"), 8001)),
        ]
    if preset_id == "tokenizer_train":
        input_paths = _split_multi_value(args.get("input_paths"))
        if not input_paths:
            raise HTTPException(status_code=400, detail="Tokenizer training requires at least one input path.")
        command = [
            sys.executable,
            "-m",
            "tokenizer.train_tokenizer",
            "--input",
            *input_paths,
            "--model-prefix",
            str(args.get("model_prefix") or "tokenizer/tokenizer"),
            "--vocab-size",
            str(_parse_number(args.get("vocab_size"), 50000)),
            "--training-text",
            str(args.get("training_text") or "tokenizer/training_corpus.txt"),
        ]
        return command
    if preset_id == "tokenizer_validate":
        return [sys.executable, "-m", "tokenizer.validate_tokenizer", str(args.get("model_path") or "tokenizer/tokenizer.model")]
    if preset_id == "training_run":
        train_shards = _split_multi_value(args.get("train_shards"))
        if not train_shards:
            raise HTTPException(status_code=400, detail="Training run requires at least one training shard.")
        command = [
            sys.executable,
            "-m",
            "train.trainer",
            "--model-config",
            str(args.get("model_config") or "configs/model/1b.yaml"),
            "--schedule-config",
            str(args.get("schedule_config") or "configs/train/schedule.yaml"),
            "--train-shards",
            *train_shards,
            "--output-dir",
            str(args.get("output_dir") or "runs/default"),
            "--steps",
            str(_parse_number(args.get("steps"), 20)),
        ]
        validation_shards = _split_multi_value(args.get("validation_shards"))
        if validation_shards:
            command.extend(["--validation-shards", *validation_shards])
        if bool(args.get("disable_wandb", True)):
            command.append("--disable-wandb")
        return command
    if preset_id == "eval_run":
        return [sys.executable, "-m", "eval.run_benchmarks"]
    if preset_id == "git_status":
        return ["git", "status", "--short", "--branch"]
    if preset_id == "hf_sync":
        return [sys.executable, "hf_push.py"]
    if preset_id == "git_commit_push":
        paths = _split_multi_value(args.get("paths"))
        if not paths:
            raise HTTPException(status_code=400, detail="Git add/commit/push requires explicit paths.")
        message = str(args.get("commit_message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Git add/commit/push requires a commit message.")
        remote = str(args.get("remote") or "origin")
        branch = str(args.get("branch") or "main")
        add_paths = " ".join(_quote_shell(path) for path in paths)
        if os.name == "nt":
            return (
                f"git add -- {add_paths}; "
                f"if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}; "
                f"git commit -m {_quote_shell(message)}; "
                f"if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}; "
                f"git push {_quote_shell(remote)} {_quote_shell(branch)}; "
                "exit $LASTEXITCODE"
            )
        return f"git add -- {add_paths} && git commit -m {_quote_shell(message)} && git push {_quote_shell(remote)} {_quote_shell(branch)}"
    raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")


def build_control_router(api_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]) -> APIRouter:
    """Create the shared HTML UI + command control router."""

    router = APIRouter()
    presets = _build_presets(enable_generate="generate" in api_handlers)
    preset_map = {item.identifier: item for item in presets}

    @router.get("/", response_class=HTMLResponse)
    def index() -> str:
        return STATIC_INDEX.read_text(encoding="utf-8")

    @router.post("/api/login")
    def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
        password = _get_password()
        if not hmac.compare_digest(payload.password, password):
            raise HTTPException(status_code=401, detail="Invalid password.")
        token = _encode_cookie_payload({"iat": time.time(), "nonce": secrets.token_hex(8)})
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=SESSION_AGE_SECONDS,
            httponly=True,
            samesite="lax",
        )
        return {"success": True}

    @router.get("/api/commands/presets")
    def list_presets(_: dict[str, Any] = Depends(_require_session)) -> dict[str, Any]:
        return {"presets": [preset.to_dict() for preset in presets], "repo_root": str(REPO_ROOT)}

    @router.post("/api/commands/run")
    def run_command(payload: RunCommandRequest, _: dict[str, Any] = Depends(_require_session)) -> dict[str, Any]:
        if payload.preset_id:
            preset = preset_map.get(payload.preset_id)
            if preset is None:
                raise HTTPException(status_code=404, detail=f"Unknown preset: {payload.preset_id}")
            _validate_preset_args(preset, payload.args)
            if preset.mode == "api":
                handler = api_handlers.get(payload.preset_id)
                if handler is None:
                    raise HTTPException(status_code=400, detail=f"Preset {payload.preset_id} is not available on this server.")
                return _api_response(handler, payload.args)
            command = _build_command_for_preset(payload.preset_id, payload.args)
            mode = "shell" if isinstance(command, str) else "job"
            try:
                job = CONTROL_MANAGER.start_job(preset.label, command, cwd=str(REPO_ROOT), mode=mode)
            except OSError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"kind": "job", "job": job.to_dict()}
        if payload.command:
            cwd = payload.cwd or str(REPO_ROOT)
            try:
                job = CONTROL_MANAGER.start_job("Raw Command", payload.command, cwd=cwd, mode="shell")
            except OSError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"kind": "job", "job": job.to_dict()}
        raise HTTPException(status_code=400, detail="Provide either preset_id or command.")

    @router.get("/api/jobs")
    def list_jobs(_: dict[str, Any] = Depends(_require_session)) -> dict[str, Any]:
        return {"jobs": CONTROL_MANAGER.list_jobs()}

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str, _: dict[str, Any] = Depends(_require_session)) -> dict[str, Any]:
        try:
            job = CONTROL_MANAGER.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc
        return {"job": job.to_dict(), "logs": job.logs[-200:]}

    @router.get("/api/jobs/{job_id}/stream")
    def stream_job(job_id: str, _: dict[str, Any] = Depends(_require_session)) -> StreamingResponse:
        try:
            job = CONTROL_MANAGER.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

        def event_stream() -> Iterator[str]:
            index = 0
            while True:
                heartbeat = False
                with job.condition:
                    while index >= len(job.events) and job.status in {"running", "stopping"}:
                        job.condition.wait(timeout=15)
                        if index >= len(job.events) and job.status in {"running", "stopping"}:
                            heartbeat = True
                            break
                    pending = job.events[index:]
                if heartbeat:
                    yield ": keep-alive\n\n"
                    continue
                for item in pending:
                    index = item["id"] + 1
                    payload = json.dumps(item["data"])
                    yield f"id: {item['id']}\nevent: {item['event']}\ndata: {payload}\n\n"
                if job.status not in {"running", "stopping"} and index >= len(job.events):
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str, _: dict[str, Any] = Depends(_require_session)) -> dict[str, Any]:
        try:
            job = CONTROL_MANAGER.stop_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc
        return {"job": job.to_dict()}

    return router

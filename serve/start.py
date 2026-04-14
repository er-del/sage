"""Start SAGE with a generated password and optional public tunnel."""

from __future__ import annotations

import argparse
import atexit

import uvicorn

from serve.control_plane import set_runtime_access_urls
from serve.server import app as gpu_app, configure_runtime_paths
from serve.server_cpu import app as cpu_app


def _display_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _start_public_tunnel(port: int, auth_token: str | None = None) -> str:
    try:
        from pyngrok import ngrok
    except ImportError:
        raise RuntimeError("pyngrok is required for public tunneling. Install with: pip install pyngrok")

    if auth_token:
        ngrok.set_auth_token(auth_token)
    tunnel = ngrok.connect(addr=port, proto="http", bind_tls=True) # type: ignore

    def _close_tunnel() -> None:
        ngrok.disconnect(tunnel.public_url) # type: ignore

    atexit.register(_close_tunnel)
    return str(tunnel.public_url)


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Start the SAGE server with an auto-generated login password.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Start the CPU control-plane server instead of the PyTorch server.")
    parser.add_argument("--share", action="store_true", help="Create a public ngrok URL for the running server.")
    parser.add_argument("--ngrok-token", default=None, help="Optional ngrok auth token used when --share is enabled.")
    parser.add_argument("--public-url", default=None, help="Optional existing public URL to print in the startup banner.")
    parser.add_argument("--model-config", default=None, help="Optional model config path for the PyTorch server.")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional checkpoint directory for the PyTorch server.")
    parser.add_argument("--tokenizer-model", default=None, help="Optional tokenizer model path for the PyTorch server.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Start the configured SAGE server."""
    args = build_argparser().parse_args(argv)
    port = args.port or (8001 if args.cpu else 8000)
    local_url = f"http://{_display_host(args.host)}:{port}"
    public_url = args.public_url
    if args.share:
        public_url = _start_public_tunnel(port, auth_token=args.ngrok_token)
    set_runtime_access_urls(local_url=local_url, public_url=public_url)
    if not args.cpu:
        configure_runtime_paths(
            model_config=args.model_config,
            checkpoint_dir=args.checkpoint_dir,
            tokenizer_model=args.tokenizer_model,
        )
    uvicorn.run(cpu_app if args.cpu else gpu_app, host=args.host, port=port)


if __name__ == "__main__":
    main()

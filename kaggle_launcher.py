# Kaggle one-cell launcher for the SAGE server
import os
import sys
import time
import atexit
import subprocess
import importlib
import secrets
from pathlib import Path

# --- Kaggle Specific Setup ---
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    NGROK_AUTHTOKEN = user_secrets.get_secret("NGROK_AUTHTOKEN")
except Exception:
    NGROK_AUTHTOKEN = None

# Kaggle working directory is /kaggle/working/
REPO_URL = "https://huggingface.co/sage002/sage"
REPO_DIR = Path("/kaggle/working/sage")
PORT = 8000
RUN_GENERATE_SMOKE = False

def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

# 1. Clone or update repo
if not REPO_DIR.exists():
    run(["git", "clone", REPO_URL, str(REPO_DIR)])
else:
    run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])

# 2. Install dependencies (Kaggle pre-installs many, but we ensure versions)
run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"])
run([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi>=0.110.0", "uvicorn>=0.29.0", "python-multipart>=0.0.9",
    "pydantic>=2.7.0", "pyyaml>=6.0.1", "psutil>=5.9.8",
    "pyngrok>=7.2.0", "requests>=2.31.0"
])

import torch
import requests
from pyngrok import ngrok
importlib.invalidate_caches()

# 3. Get password from control_plane module (same logic as server uses)
from serve.control_plane import _get_password as _get_sage_password

PASSWORD = _get_sage_password()
print(f"SAGE_WEB_PASSWORD: {PASSWORD}")

if not NGROK_AUTHTOKEN:
    raise ValueError("NGROK_AUTHTOKEN Missing. Add it to 'Add-ons' -> 'Secrets' in Kaggle.")

# 4. Supply necessary SAGE environment variables
env = os.environ.copy()
env["SAGE_WEB_PASSWORD"] = PASSWORD
env["SAGE_MODEL_CONFIG"] = env.get("SAGE_MODEL_CONFIG", "configs/model/1b.yaml")
env["SAGE_CHECKPOINT_DIR"] = env.get("SAGE_CHECKPOINT_DIR", "runs/sage-1b")
env["SAGE_TOKENIZER_MODEL"] = env.get("SAGE_TOKENIZER_MODEL", "tokenizer/tokenizer.model")

USE_GPU_SERVER = torch.cuda.is_available()
APP_TARGET = "serve.server:app" if USE_GPU_SERVER else "serve.server_cpu:app"

print(f"GPU available: {USE_GPU_SERVER}")
print(f"Starting app target: {APP_TARGET}")
print(f"SAGE_WEB_PASSWORD: {PASSWORD}  <-- Use this to login")

# 5. Start Uvicorn Server
log_path = REPO_DIR / "uvicorn.log"
log_file = open(log_path, "w", encoding="utf-8")

server_proc = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn",
        APP_TARGET,
        "--host", "0.0.0.0",
        "--port", str(PORT),
    ],
    cwd=str(REPO_DIR),
    env=env,
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

def cleanup():
    global server_proc, log_file
    print("Cleaning up...")
    try:
        ngrok.disconnect(public_url)
        ngrok.kill()
    except: pass
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
    try: log_file.close()
    except: pass
    print("Cleanup complete.")

atexit.register(cleanup)

# 6. Wait for health check
health_url = f"http://127.0.0.1:{PORT}/health"
for _ in range(60):
    if server_proc.poll() is not None:
        log_file.flush()
        raise RuntimeError("Uvicorn exited early. Check logs.")
    try:
        r = requests.get(health_url, timeout=2)
        if r.ok:
            print("Local health OK:", r.json())
            break
    except: pass
    time.sleep(2)

# 7. Start Ngrok HTTPs Tunnel
try:
    ngrok.kill()
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    tunnel = ngrok.connect(addr=PORT, proto="http", bind_tls=True)
    public_url = tunnel.public_url

    print("\n" + "="*45)
    print("            SAGE DASHBOARD (KAGGE)            ")
    print("="*45)
    print(f"URL: {public_url}")
    print(f"PWD: {PASSWORD}")
    print("="*45 + "\n")

except Exception as e:
    print("Could not start Ngrok: ", e)

print(f"Server log path: {log_path}")
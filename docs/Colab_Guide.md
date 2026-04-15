# SAGE UI — Colab Quickstart Guide

So you started the SAGE server in Google Colab, got your Ngrok link, and loaded the webpage. Welcome to the **SAGE Browser IDE**!

Right now, you are looking at the "Control Plane." The AI is essentially a blank slate. To get a "proper agent" that can chat with you, you need to use this interface to prepare data and train the model step-by-step.

Here is exactly what to do.

---

### Step 1: Open the Terminal & Download Data

To train a model, you need text. We recently added a 5-Billion-Token downloader, but we don't need all 5 billion for a quick test.

1. In the SAGE IDE, open the **CLI Terminal** (click the `>_` icon on the left sidebar, or press `Ctrl + \``).
2. Type the following command and press Enter to download a small 1% slice (~50 Million tokens):
   ```bash
   python debug/download_5b_tokens.py --output-dir data/raw --scale 0.01
   ```
3. Watch the terminal. It will take a few minutes to download the General Web, Code, Math, Wikipedia, and Synthetic datasets into `data/raw/`.

### Step 2: Train Your Tokenizer

The AI doesn't read English words; it reads "Tokens". It needs to learn its vocabulary from your downloaded data.

1. Click the **Presets** tab (the rocket icon 🚀) on the left sidebar.
2. Select **Tokenizer Train** from the dropdown menu.
3. Click the purple **Run Job** button.
4. A new panel will slide out showing you the live logs. Wait until it says `Job finished successfully`.

### Step 3: Fast-Pack Your Data (Sharding)

Training directly from text files is too slow for a GPU. We need to tokenize the text and pack it into high-speed Parquet "shards".

1. Go back to the **Presets** tab.
2. Select **Build Data Shards**.
3. Set the `shard_size` to `2048`.
4. Click **Run Job**.
5. Wait for the logs to finish. When done, your data is packed and ready!

### Step 4: Begin Training the AI

Now it's time to put the GPU to work.

1. Go back to the **Presets** tab.
2. Select **Training Run**.
3. You can leave the steps at the default (e.g., `20` for a smoke test, or change it to `2000` for a real micro-run). Make sure `disable_wandb` is checked so it doesn't ask for a Weights & Biases login.
4. Click **Run Job**.
5. The live log viewer will now stream training metrics. You will see `loss` going down and `tokens_per_second` showing how fast your Colab T4 GPU is churning through data.
6. The trainer automatically saves checkpoints (e.g., `ckpt_step_1000.pt`) into the `runs/` folder.

### Step 5: Chat with Your New Agent

Once the training has run for a decent amount of steps and a checkpoint is saved, the model is ready to talk!

1. Click the **Chat** tab (the speech bubble icon 💬) on the left sidebar.
2. Type a message like _"What is Python?"_ and hit Enter.
3. The UI will send this prompt to the backend, run inference using your newly trained checkpoint and tokenizer, and stream the generated response back to your screen.

_(Note: If you only trained for 20 steps, the AI will probably respond with random gibberish. Real reasoning requires thousands of steps over billions of tokens!)_

---
# Connect your models with ngrok for public IP 
~ Do not forgot to set up and ngrok authentication token with this!


```bash
# Colab one-cell launcher for the real SAGE server
# Before running:
# 1. In Colab, open the Secrets panel (Key icon on the left) and add your NGROK_AUTHTOKEN
# 2. If you want /generate, switch Colab to a T4 GPU runtime

import os
import sys
import time
import atexit
import subprocess
import importlib
import secrets
from pathlib import Path

REPO_URL = "https://huggingface.co/sage002/sage"
REPO_DIR = Path("/content/sage")
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

# 2. Install dependencies
run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"])
run([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi>=0.110.0", "uvicorn>=0.29.0", "python-multipart>=0.0.9",
    "pydantic>=2.7.0", "pyyaml>=6.0.1", "psutil>=5.9.8",
    "pyngrok>=7.2.0", "requests>=2.31.0"
])

try:
    import torch
except ImportError:
    run([sys.executable, "-m", "pip", "install", "-q", "torch>=2.1.0"])
    import torch

# Refresh path caches so the cell can instantly import newly installed modules
importlib.invalidate_caches()
import requests
from pyngrok import ngrok

# 3. Retrieve Ngrok token securely via Colab Secrets (or fallback to environment variable)
try:
    from google.colab import userdata
    NGROK_AUTHTOKEN = userdata.get("NGROK_AUTHTOKEN")
except Exception:
    NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")

if not NGROK_AUTHTOKEN:
    raise ValueError("Missing NGROK_AUTHTOKEN. Please add it to your Colab Secrets.")

# 4. Supply necessary SAGE environment variables for the server
env = os.environ.copy()
env["SAGE_WEB_PASSWORD"] = env.get("SAGE_WEB_PASSWORD") or secrets.token_urlsafe(12)
env["SAGE_MODEL_CONFIG"] = env.get("SAGE_MODEL_CONFIG", "configs/model/1b.yaml")
env["SAGE_CHECKPOINT_DIR"] = env.get("SAGE_CHECKPOINT_DIR", "runs/sage-1b")
env["SAGE_TOKENIZER_MODEL"] = env.get("SAGE_TOKENIZER_MODEL", "tokenizer/tokenizer.model")

USE_GPU_SERVER = torch.cuda.is_available()
APP_TARGET = "serve.server:app" if USE_GPU_SERVER else "serve.server_cpu:app"

print(f"GPU available: {USE_GPU_SERVER}")
print(f"Starting app target: {APP_TARGET}")
print(f"SAGE_WEB_PASSWORD: {env['SAGE_WEB_PASSWORD']}  <-- Use this to login to the IDE")

# 5. Start Uvicorn Server attached to the log file via Popen
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
    env=env,                    # Required: Passes the SAGE environment variables to Uvicorn
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

def cleanup():
    global server_proc, log_file
    print("Cleaning up...")
    try:
        ngrok.disconnect(public_url)
        ngrok.kill()
    except Exception:
        pass
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    try:
        log_file.close()
    except Exception:
        pass
    print("Cleanup complete.")

atexit.register(cleanup)

# 6. Wait for health check success
health_url = f"http://127.0.0.1:{PORT}/health"
for _ in range(60):
    if server_proc.poll() is not None:
        log_file.flush()
        raise RuntimeError("Uvicorn exited early.\n\n" + log_path.read_text(encoding="utf-8", errors="ignore"))
    try:
        r = requests.get(health_url, timeout=2)
        if r.ok:
            print("Local health OK:", r.json())
            break
    except Exception:
        pass
    time.sleep(2)
else:
    log_file.flush()
    raise TimeoutError("Server did not become healthy.\n\n" + log_path.read_text(encoding="utf-8", errors="ignore"))

# 7. Start Ngrok HTTPs Tunnel
try:
    ngrok.kill()
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    tunnel = ngrok.connect(addr=PORT, proto="http", bind_tls=True) # Forces HTTPS UI which stops browser mixed-content blocks
    public_url = tunnel.public_url

    print("\n============================================")
    print("        SAGE DASHBOARD        ")
    print("==============================================")
    print(f"URL: {public_url}")
    print(f"PWD: {env['SAGE_WEB_PASSWORD']}")
    print("==============================================\n")

    if USE_GPU_SERVER:
        print("Generate          :", f"{public_url}/generate")
    else:
        print("Wait: Generate is not available on CPU server in this repo")
        print("Switch Colab to a GPU runtime if you want /generate.")
except Exception as e:
    print("Could not start Ngrok: ", e)


# Optional /generate smoke test
if USE_GPU_SERVER and RUN_GENERATE_SMOKE:
    print("\nRunning /generate smoke test...")
    try:
        resp = requests.post(
            f"http://127.0.0.1:{PORT}/generate",
            json={"input_ids": [1, 42, 99], "max_new_tokens": 4},
            timeout=300,
        )
        print("Generate response:", resp.json())
    except Exception as e:
        print("Generate timeout or failure:", e)


print(f"\nServer log path: {log_path}")
print("The server will continuously run until you stop the Code Cell manually.")

```
---

### Pro-Tips for the IDE

- **Command Palette:** Press `Ctrl + K` anywhere to quickly jump between tools.
- **Function Inspector:** You can click the Book 📖 icon on the right to browse the actual Python codebase from within the browser while your model trains.
- **Stop a stray training job:** Go to the **Jobs** panel (the clipboard icon) and click the red "Stop" button on any running task to free up your GPU.

# 🪐 SAGE: Kaggle & Colab Quickstart Guide

Welcome to the **Self-Adaptive General Engine (SAGE)**. This guide will help you get SAGE v2 running on a cloud environment (like Kaggle's 2x T4 or Google Colab) in under 5 minutes.

---

## 🛠️ Step 1: Environment Setup
Run this cell first to install dependencies and fix any common binary incompatibilities (like the Numpy/Torch mismatch).

```python
# Install Core Dependencies
!pip install "numpy<2.0.0" --force-reinstall
!pip install bitsandbytes tqdm tiktoken faiss-cpu datasets wandb --upgrade

# Install SAGE v2 directly from the codebase
# (Assuming you have cloned the repo or uploaded sage_single.py)
print("✅ Environment ready. Please RESTART YOUR KERNEL now if this is your first run.")
```

---

## 🔑 Step 2: Weights & Biases Logging (Optional but Recommended)
To track your training progress with professional charts:
1. Get your API Key from [wandb.ai/authorize](https://wandb.ai/authorize).
2. Add it to your Kaggle **Secrets** with the label `WANDB_API_KEY`.
3. Run this:

```python
import wandb
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    wandb.login(key=user_secrets.get_secret("WANDB_API_KEY"))
except:
    import os
    os.environ["WANDB_MODE"] = "offline"
    print("⚠️ W&B Secret not found. Running in offline mode.")
```

---

## 💬 Step 3: Launch the SAGE Chat Interface
This is a premium, multi-GPU enabled chat widget. Paste this into a cell to start interacting with SAGE.

```python
import torch, os, random
import torch.nn as nn
import ipywidgets as widgets
from IPython.display import display, HTML
from sage_single import SageModel, SageConfig, SageTokenizer, generate, ConversationHistory, train_model, finetune

# -- Initialization --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = SageConfig() 
tokenizer = SageTokenizer()
history = ConversationHistory(tokenizer, max_tokens=1024)
model = SageModel(config)

# -- Multi-GPU Logic --
gpu_count = torch.cuda.device_count()
if gpu_count > 1:
    print(f"🚀 Multi-GPU active: {gpu_count} GPUs.")
    model = nn.DataParallel(model)
model = model.to(device)

# -- Load Weights --
ckpt_path = "checkpoints/sage_latest.pt"
if os.path.exists(ckpt_path):
    base_model = getattr(model, "module", model)
    ckpt = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt['model_state_dict'])
    print("✅ Weights loaded from checkpoint.")
else:
    print("⚠️ RANDOM WEIGHTS (Type /train <steps> to begin learning).")

# -- Render UI --
chat_display = widgets.Output(layout={'border': '1px solid #444', 'height': '450px', 'overflow_y': 'scroll', 'padding': '10px'})
text_input = widgets.Text(placeholder="Chat or type /train 1000...", layout={'width': '80%'})
send_button = widgets.Button(description="Send", button_style='primary', layout={'width': '18%'})
display(HTML("<style>.user-msg { background: #2b2d42; color: #fff; padding: 10px; border-radius: 10px; margin: 5px; border-left: 5px solid #ef233c; } .sage-msg { background: #1a1b2e; color: #fff; padding: 10px; border-radius: 10px; margin: 5px; border-left: 5px solid #4cc9f0; }</style>"))

def on_send(_=None):
    user_text = text_input.value.strip()
    if not user_text: return
    text_input.value = "" 
    with chat_display:
        if user_text.startswith("/train"):
            steps = int(user_text.split()[1]) if len(user_text.split()) > 1 else 100
            print(f"🚀 TRAINING STARTING ({steps} steps)...")
            train_model(model, config, total_steps=steps)
            print("✅ DONE.")
            return
        display(HTML(f'<div class="user-msg"><b>You:</b> {user_text}</div>'))
        response = generate(model, tokenizer, history.build_prompt(user_text), stream=False)
        res = response.split("SAGE:")[-1].split("</response>")[0].replace("<response>", "").strip()
        history.add("user", user_text); history.add("assistant", res)
        display(HTML(f'<div class="sage-msg"><b>SAGE:</b> {res}</div>'))

text_input.on_submit(on_send); send_button.on_click(lambda b: on_send())
display(chat_display, widgets.HBox([text_input, send_button]))
```

---

## 🎮 Command Cheat Sheet
| Command | Action |
| :--- | :--- |
| `/train <steps>` | Starts pre-training (Base knowledge). Recommended: 5000+ |
| `/clear` | Resets the conversation history. |
| `/finetune <steps>`| (Coming Soon) Starts instruction fine-tuning. |

---

## 💡 Pro Tips for T4 GPUs
1. **Batch Size**: The default `batch_size=4` with `gradient_accumulation=16` is perfect for a 2x T4 setup (32GB VRAM total). 
2. **Persistence**: Kaggle outputs are deleted when the session ends. Make sure to **download** the `checkpoints/` folder or sync it to **Hugging Face** regularly.
3. **Patience**: Loss will fluctuate. Look for a steady downward trend on your W&B dashboard!

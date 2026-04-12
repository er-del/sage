# SAGE — Self-Adaptive General Engine

**SAGE** is a senior-grade, production-structured Large Language Model (LLM) system built entirely from scratch using Python and PyTorch. It implements modern transformer architectures including Mixture of Experts (MoE), Rotary Positional Embeddings (RoPE), and Low-Rank Adaptation (LoRA).

Designed to be both educational and functional, SAGE can be trained, fine-tuned, quantized, and deployed on a single consumer GPU (e.g., NVIDIA T4 with 16GB VRAM).

---

## ☁️ Cloud Quickstart (Kaggle / Colab)
Running SAGE in the cloud? Check out the **[Kaggle & Colab Quickstart Guide](file:///c:/Users/Lenovo/OneDrive/Desktop/Documents/LLM_MOdel/SAGE_KAGGLE_GUIDE.md)** for one-click setup and a premium interactive chat interface.

---

## 🚀 Key Features

- **Decoder-Only Transformer**: A GPT-style architecture with pre-layer normalization.
- **Mixture of Experts (MoE)**: Efficient scaling with a learned router selecting top-k experts per token.
- **Rotary Positional Embeddings (RoPE)**: Enhanced long-sequence generalization.
- **KV-Cache Inference**: O(1) time-per-token generation for high-speed response.
- **Retrieval-Augmented Generation (RAG)**: Integration with FAISS for document-based context lookup.
- **Efficient Fine-Tuning**: Support for LoRA and instruction tuning with loss masking.
- **Post-Training Quantization**: INT8 support to reduce memory footprint.
- **Interactive CLI**: A full REPL (Read-Eval-Print Loop) for chatting and system management.

---

## 📂 Project Structure

```text
sage/
├── model.py        # Core architecture (Transformer, MoE, RoPE, Attention)
├── data.py         # Tokenization (tiktoken) & Streaming Datasets (HuggingFace)
├── train.py        # Pre-training loop with AdamW, AMP, and Cosine Decay
├── inference.py    # Text generation (Greedy, Temp, Top-k, Top-p sampling)
├── finetune.py     # LoRA implementation & Instruction Tuning
├── optimize.py     # INT8 Quantization & Pruning utilities
├── memory.py       # RAG Vector Store & Conversation History
├── cli.py          # Interactive Terminal Interface
├── utils.py        # Logging, Checkpointing, and Helper functions
├── config.py       # Central Hyperparameter Configuration
└── requirements.txt # System dependencies
sage_single.py      # Consolidated single-file version for easy portability
```

---

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.9+ and a CUDA-compatible GPU (recommended).

```bash
# Clone the repository (GitHub)
git clone https://github.com/er-del/sage.git
cd sage

# OR Clone from Hugging Face
git clone https://huggingface.co/sage002/sage
cd sage

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependencies
- **PyTorch**: Core deep learning framework.
- **tiktoken**: Fast BPE tokenization (OpenAI's cl100k_base).
- **datasets**: For streaming training data from HuggingFace.
- **faiss-cpu**: For vector-based retrieval (RAG).
- **tqdm**: Progress bars for training.
- **bitsandbytes**: (Optional) For advanced quantization.

---

## 🎮 Getting Started

### Launching the CLI
You can run the modular version or the single-file version:

```bash
# Modular version
python -m sage.cli

# Single-file version
python sage_single.py
```

### Basic Chat
Once launched, simply type your message to chat with SAGE. The system uses a rolling conversation history to maintain context.

---

## 👨‍🏫 Training SAGE

SAGE supports real-time training either directly from the interactive REPL or via simple one-liner CLI commands (useful for background scripts).

### Non-Interactive "One-Liner" Commands
If you want to bypass the chat interface and just run a training job, pass the command as a CLI argument:
```bash
python sage_single.py --train 100       # Pre-train for 100 steps
python sage_single.py --finetune 200    # Instruction-tune for 200 steps
python sage_single.py --quantize        # Apply INT8 quantization
```

### Interactive REPL Commands
If you are inside the chat interface, use the slash commands:

### /train [steps]
Run pre-training using the `TinyStories` dataset (default).
- `/train 100` — Trains for 100 steps and saves a checkpoint.

### /finetune [steps]
Perform instruction fine-tuning using LoRA adapters.
- `/finetune 200` — Trains on instruction/response pairs and merges weights.

---

## 🧠 Advanced Commands

| Command | Action |
| :--- | :--- |
| `/save` | Manually save the current model checkpoint. |
| `/load` | Reload the latest checkpoint from the `checkpoints` directory. |
| `/quantize` | Convert model weights to INT8 (CPU) for reduced memory usage. |
| `/rag on` | Enable Retrieval-Augmented Generation. |
| `/rag add <text>` | Add new knowledge to SAGE's retrieval database. |
| `/clear` | Clear the current conversation history. |
| `/help` | Show the list of available commands. |
| `/exit` | Exit the program cleanly. |

---

## 🏗️ Architecture Details

### Mixture of Experts (MoE)
SAGE swaps standard FFN layers for MoE blocks. Each block contains 4 experts, where exactly 2 are activated per token via a learned linear router. This allows for higher total capacity without increasing the computational cost per token.

### Rotary Positional Embeddings (RoPE)
Positions are encoded via complex-valued rotations of query and key vectors. This allows SAGE to better handle sequences longer than what it was trained on compared to absolute position embeddings.

### Inference Engine
Generation supports:
- **Temperature**: Adjusts randomness.
- **Top-k**: Limits sampling to the most likely 'k' tokens.
- **Top-p (Nucleus)**: Limits sampling to a cumulative probability threshold.
- **KV-Caching**: Caches Attention keys and values to avoid redundant computation.

---

## 🤗 Hugging Face Model Hub

This project is actively maintained on Hugging Face. You can find pre-trained checkpoints, datasets, and community discussions here:

🔗 **[huggingface.co/sage002/sage](https://huggingface.co/sage002/sage)**

**Developed by Antigravity AI Systems.**

---

## 📜 Disclaimer
SAGE is an experimental engine. While architecturally complete, the quality of generated responses depends heavily on the amount of training data and compute steps provided.

**Developed by Antigravity AI Systems.**

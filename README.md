# SAGE — Self-Adaptive General Engine

**SAGE** is a senior-grade, production-structured Large Language Model (LLM) system built entirely from scratch using Python and PyTorch. It implements modern transformer architectures including Mixture of Experts (MoE), Rotary Positional Embeddings (RoPE), and Low-Rank Adaptation (LoRA).

Designed to be both educational and functional, SAGE can be trained, fine-tuned, quantized, and deployed on a single consumer GPU (e.g., NVIDIA T4 with 16GB VRAM).

---
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


### Basic Chat
Once launched, simply type your message to chat with SAGE. The system uses a rolling conversation history to maintain context.

---

## 👨‍🏫 Training SAGE

SAGE supports real-time training either directly from the interactive REPL or via simple one-liner CLI commands (useful for background scripts).

### Non-Interactive "One-Liner" Commands

---

## 🧠 Advanced Commands

---

## 🏗️ Architecture Details

---

## 🤗 Hugging Face Model Hub

This project is actively maintained on Hugging Face. You can find pre-trained checkpoints, datasets, and community discussions here:

🔗 **[huggingface.co/sage002/sage](https://huggingface.co/sage002/sage)**


---

## 📜 Disclaimer
SAGE is an experimental engine. While architecturally complete, the quality of generated responses depends heavily on the amount of training data and compute steps provided.

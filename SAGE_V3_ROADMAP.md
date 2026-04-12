# SAGE v3: The "Long-Vision" Roadmap 🗺️

This document outlines the high-impact architectural upgrades that will transform SAGE into a multi-thousand token reasoning assistant with multimedia capabilities.

---

## 🏗️ 1. Long-Context Scaling (YaRN / RoPE-Interpolation)

**Goal**: Increase SAGE's maximum comprehension from 1,024 to **4,096+ tokens**.

### Technical Strategy:
Currently, our `freqs_cis` are precomputed for a fixed window. In v3, we will implement **NTK-Aware Interpolation**.
- **Implementation**: We will add a `scaling_factor` to `SageConfig`.
- **Logic**: During inference, if the sequence length exceeds the original training window, we will "stretch" the rotary frequencies dynamically rather than letting them overflow.
- **Benefit**: SAGE can read entire source code files or long essays without "losing its mind" at the 1,025th token.

---

## 📂 2. Interactive RAG (Kaggle UI Integration)

**Goal**: Allow users to "Upload and Chat" with any file instantly in the notebook.

### Technical Strategy:
- **Widget Update**: Add a `widgets.FileUpload` component to the Kaggle chat interface.
- **Auto-Ingestion**: When a file is uploaded, a background hook will:
    1.  Parse the text (PDF, `.py`, `.md`).
    2.  Chunk it into 200-token segments.
    3.  Generate embeddings and add them to the **FAISS Vector Store**.
- **Real-time Recall**: SAGE will automatically pull context from these uploaded files using the `retrieve_context` logic we've already built.

---

## 👁️ 3. Multimodal Foundation (Vision Projection)

**Goal**: Let SAGE "see" images.

### Technical Strategy:
Since SAGE is a small, efficient model (133M), it is the perfect candidate for a **Vision-Language Model (VLM)**.
- **Architecture**: We will add a frozen **CLIP-ViT** image encoder.
- **The Bridge**: We will implement a `VisionProjector` (a simple 2-layer MLP) that converts CLIP image embeddings (e.g., 768-dim) into SAGE token embeddings (512-dim).
- **Outcome**: You will be able to provide an image URL and a prompt like "What is in this picture?", and SAGE will respond based on the visual tokens.

---

## ⚡ 4. Training Stability: LayerNorm Tuning

To support these advanced features, we will move to **RMSNorm** (Root Mean Square Layer Normalization) for even faster convergence and better numerical stability on the double-T4 setup.

---

### Which one first?
We can begin implementing **RoPE Scaling** immediately to give SAGE a massive context boost without needing new weights. Just let me know when you're ready!

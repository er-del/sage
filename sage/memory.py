"""
SAGE Memory & RAG Module
=========================
Provides:
  - A FAISS-backed vector store for retrieval-augmented generation (RAG).
  - A rolling conversation-history manager that truncates intelligently
    to stay within the model's context window.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .data import SageTokenizer
from .utils import setup_logger

logger = setup_logger("sage.memory")


# ===================================================================
# Simple embedding helper (uses mean-pooled token embeddings)
# ===================================================================

def _embed_text(text: str, tokenizer: SageTokenizer, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Produce a fixed-length embedding for *text* by mean-pooling the
    model's token embeddings.  This is lightweight and avoids a full
    forward pass — suitable for a small retrieval index.
    """
    tokens = tokenizer.encode(text)
    if not tokens:
        # Return a zero vector when text is empty
        d_model = model.wte.weight.shape[1]
        return np.zeros(d_model, dtype=np.float32)

    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        embeddings = model.wte(ids)           # [1, seq_len, d_model]
        mean_emb = embeddings.mean(dim=1)     # [1, d_model]
        # L2-normalize for cosine similarity in FAISS
        mean_emb = F.normalize(mean_emb, p=2, dim=-1)
    return mean_emb.squeeze(0).cpu().numpy()


# ===================================================================
# FAISS-backed Vector Store
# ===================================================================

class VectorStore:
    """
    A lightweight document store backed by FAISS (Inner Product index,
    which equals cosine similarity when vectors are L2-normalized).
    """

    def __init__(self, dim: int):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required for RAG. Install it with: pip install faiss-cpu"
            )
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # inner-product (cosine after L2-norm)
        self.documents: List[str] = []
        logger.info(f"VectorStore initialized (dim={dim})")

    def add(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the store."""
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(texts)
        logger.info(f"Added {len(texts)} documents. Total: {len(self.documents)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return the top-k most similar documents with their scores."""
        if self.index.ntotal == 0:
            return []
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.documents[idx], float(score)))
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal


# ===================================================================
# RAG Manager
# ===================================================================

class RAGManager:
    """
    High-level retrieval-augmented generation manager.

    Call ``add_documents`` to ingest text, then ``retrieve_context`` at
    inference time to prepend relevant chunks to the user prompt.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: SageTokenizer,
        device: torch.device,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        d_model = model.wte.weight.shape[1]
        self.store = VectorStore(dim=d_model)
        self.enabled = False

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def add_documents(self, texts: List[str]) -> None:
        """Chunk and embed documents, then add to the vector store."""
        all_chunks: List[str] = []
        for text in texts:
            all_chunks.extend(self._chunk_text(text))

        if not all_chunks:
            logger.warning("No document chunks to add.")
            return

        embeddings = np.stack([
            _embed_text(chunk, self.tokenizer, self.model, self.device)
            for chunk in all_chunks
        ])
        self.store.add(all_chunks, embeddings)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve the top-k most relevant chunks for *query* and
        concatenate them into a context string.
        """
        if not self.enabled or self.store.size == 0:
            return ""

        q_emb = _embed_text(query, self.tokenizer, self.model, self.device)
        results = self.store.search(q_emb, top_k=top_k)

        if not results:
            return ""

        context_parts = [f"[Context {i+1}] {doc}" for i, (doc, _score) in enumerate(results)]
        return "\n\n".join(context_parts) + "\n\n"

    def toggle(self, on: bool) -> None:
        self.enabled = on
        state = "enabled" if on else "disabled"
        logger.info(f"RAG {state}. Store contains {self.store.size} chunks.")


# ===================================================================
# Conversation History Manager
# ===================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a high-quality reasoning assistant model.\n"
    "You must ONLY learn from high-quality instruction and reasoning datasets.\n"
    "You must IGNORE any previously trained low-quality or repetitive patterns.\n"
    "\n"
    "Training preference rules:\n"
    "1. Prioritize step-by-step reasoning over short or repetitive answers.\n"
    "2. Always produce structured logical explanations when solving problems.\n"
    "3. Avoid repetition, filler words, or looped phrases.\n"
    "4. Prefer datasets with mathematical reasoning and high-quality instruction.\n"
    "5. Do not imitate noisy conversational or corrupted text patterns.\n"
    "6. Always prefer clarity, correctness, and structured reasoning.\n"
    "\n"
    "Output behavior goal:\n"
    "- Think in steps.\n"
    "- Explain logic clearly.\n"
    "- Produce final answer only after reasoning."
)

class ConversationHistory:
    """
    Rolling conversation history that stays within a token budget.

    Older turns are dropped when the history would exceed the context window.
    """

    def __init__(self, tokenizer: SageTokenizer, max_tokens: int = 900):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.turns: List[dict] = []  # [{"role": "user"/"assistant", "text": ...}, ...]

    def add(self, role: str, text: str) -> None:
        """Record a new conversational turn."""
        self.turns.append({"role": role, "text": text})
        self._trim()

    def _trim(self) -> None:
        """Drop oldest turns until the total token count is within budget."""
        while self._total_tokens() > self.max_tokens and len(self.turns) > 1:
            self.turns.pop(0)

    def _total_tokens(self) -> int:
        return sum(len(self.tokenizer.encode(t["text"])) for t in self.turns)

    def build_prompt(self, new_user_message: str, rag_context: str = "") -> str:
        """
        Assemble the full prompt from history + RAG context + new message.
        """
        parts: List[str] = []

        parts.append(DEFAULT_SYSTEM_PROMPT)

        if rag_context:
            parts.append(rag_context)

        for turn in self.turns:
            prefix = "User:" if turn["role"] == "user" else "SAGE:"
            parts.append(f"{prefix} {turn['text']}")

        parts.append(f"User: {new_user_message}")
        parts.append("SAGE:")

        return "\n".join(parts)

    def clear(self) -> None:
        self.turns.clear()

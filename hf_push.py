"""Upload the current SAGE repository contents to the Hugging Face Hub."""

from __future__ import annotations

import os

from huggingface_hub import HfApi


REPO_ID = "sage002/sage"
DEFAULT_COMMIT_MESSAGE = "feat: add authenticated remote control UI and ngrok launcher"


def main() -> None:
    """Replace the remote Hugging Face repo contents with the local folder state."""
    api = HfApi()
    commit_message = os.environ.get("HF_COMMIT_MESSAGE", DEFAULT_COMMIT_MESSAGE)
    print(f"Syncing current repository to {REPO_ID}...")
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=[
            ".git/*",
            ".venv/*",
            "__pycache__/*",
            "*.pyc",
            "checkpoints/*",
            "runs/*",
            "wandb/*",
            "data/raw/*",
            "data/processed/*",
            "tokenizer/*.model",
            "tokenizer/*.vocab",
            "tokenizer/training_corpus.txt",
        ],
        delete_patterns="*",
        commit_message=commit_message,
    )
    print("Sync complete.")


if __name__ == "__main__":
    main()

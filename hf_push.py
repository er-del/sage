"""Upload SAGE model repository contents to the Hugging Face Hub.

Only uploads files relevant to the model: source code, configs,
tokenizer assets, documentation, and serving infrastructure.
Debug scripts, test suites, IDE files, checkpoints, and build
artifacts are excluded.
"""

from __future__ import annotations

import os

from huggingface_hub import HfApi


REPO_ID = "sage002/sage"
DEFAULT_COMMIT_MESSAGE = "SAGE model repository :  Updating some model checkpoints "


HF_IGNORE_PATTERNS = [
    ".git/*",
    ".gitignore",
    ".idea/*",
    ".pytest_cache/*",
    ".venv/*",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "checkpoints/*",
    "runs/*",
    "wandb/*",
    "logs/*",
    "data/raw/*",
    "data/processed/*",
    "debug/*",
    "tests/*",
    "*.log",
]


def main() -> None:
    """Replace the remote Hugging Face repo contents with the local folder state."""
    api = HfApi()
    commit_message = os.environ.get("HF_COMMIT_MESSAGE", DEFAULT_COMMIT_MESSAGE)
    print(f"Syncing current repository to {REPO_ID}...")
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=HF_IGNORE_PATTERNS,
        delete_patterns="*",
        commit_message=commit_message,
    )
    print("Sync complete.")


if __name__ == "__main__":
    main()
"""Hugging Face Spaces Deployment Tool.

Deploys the FinVault Multi-Agent Enterprise AI platform to Hugging Face Spaces
using either the huggingface_hub Python API or Git remote commands.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def deploy_via_api(space_id: str, token: str, create_if_missing: bool = True, private: bool = False) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[!] huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    project_root = Path(__file__).resolve().parents[1]

    print(f"[*] Deploying FinVault to Hugging Face Space: {space_id}...")

    if create_if_missing:
        try:
            print(f"[*] Checking/Creating Space repository '{space_id}' with Docker SDK...")
            create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="docker",
                token=token,
                private=private,
                exist_ok=True,
            )
            print("[+] Space repository ready.")
        except Exception as exc:
            print(f"[!] Warning creating repo: {exc}")

    # Files and folders to exclude from upload
    ignore_patterns = [
        ".venv/**",
        ".git/**",
        ".pytest_cache/**",
        ".ruff_cache/**",
        "__pycache__/**",
        "*.pyc",
        ".coverage",
        ".secrets/**",
        "docker-compose.observability.yml",
        "deploy/observability/**",
    ]

    print(f"[*] Uploading project files from {project_root} to space '{space_id}'...")
    try:
        api.upload_folder(
            folder_path=str(project_root),
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Deploy FinVault Enterprise Multi-Agent AI to Hugging Face Spaces",
        )
        print("\n" + "=" * 70)
        print("[✓] Successfully deployed FinVault to Hugging Face Space!")
        print(f"[>] Space URL: https://huggingface.co/spaces/{space_id}")
        print(f"[>] App will be live at: https://{space_id.replace('/', '-')}.hf.space/app/")
        print("=" * 70 + "\n")
        print("[*] Don't forget to configure your Secrets in Space Settings:")
        print("    - OPENROUTER_API_KEY (or HF_TOKEN)")
        print("    - FINVAULT_JWT_SECRET")
        print("    - FINVAULT_MODEL (e.g., minimax/minimax-m2.7:free)")
    except Exception as exc:
        print(f"[!] Upload failed: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy FinVault to Hugging Face Spaces (Docker SDK)")
    parser.add_argument(
        "--space-id",
        type=str,
        default=os.getenv("HF_SPACE_ID"),
        help="Hugging Face Space ID in the format '<username>/<space-name>' (or via HF_SPACE_ID env var)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face User Access Token with Write permissions (or via HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Space private when creating",
    )

    args = parser.parse_args()

    if not args.space_id:
        print("Error: --space-id is required (e.g. --space-id username/finvault)", file=sys.stderr)
        sys.exit(1)

    if not args.hf_token:
        print("Error: --hf-token or HF_TOKEN environment variable is required", file=sys.stderr)
        sys.exit(1)

    deploy_via_api(space_id=args.space_id, token=args.hf_token, create_if_missing=True, private=args.private)


if __name__ == "__main__":
    main()

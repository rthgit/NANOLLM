import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. RthItalia/NanoLLM-Qwen-V3")
    parser.add_argument("--local-dir", default="NANOHF", help="Local folder to upload")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--private", action="store_true", help="Create repo as private (if it does not exist)")
    parser.add_argument("--commit-message", default="Upload artifacts")
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Also upload model payloads from the folder. By default, payload files are ignored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN env var is required.")

    folder = Path(args.local_dir).resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Local folder not found: {folder}")

    print(f"Target repo: {args.repo_id}")
    print(f"Uploading folder: {folder}")
    create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    api = HfApi(token=token)
    ignore_patterns = None
    if not args.include_artifacts:
        ignore_patterns = [
            "*.zip",
            "*.pt",
            "*.pth",
            "*.bin",
            "*.safetensors",
            "*.gguf",
            "*.onnx",
            "__pycache__/*",
        ]
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        ignore_patterns=ignore_patterns,
    )
    print(f"Upload completed: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

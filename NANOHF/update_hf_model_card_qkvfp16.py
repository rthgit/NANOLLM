import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the qkvfp16 model card to Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default="RthItalia/nano_compact_3b_qkvfp16",
        help="Target Hugging Face model repo id.",
    )
    parser.add_argument(
        "--card",
        default=str(Path(__file__).with_name("HF_MODEL_CARD_qkvfp16.md")),
        help="Path to the local model card markdown file.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable that stores the Hugging Face token.",
    )
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(f"Missing token in env var: {args.token_env}")

    card_path = Path(args.card)
    if not card_path.exists():
        raise SystemExit(f"Missing model card file: {card_path}")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Update model card for qkvfp16 winner",
    )

    print("UPDATED:", args.repo_id)
    print("CARD:", card_path)
    print("URL:", f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

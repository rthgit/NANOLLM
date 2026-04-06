import os

from huggingface_hub import HfApi


def main() -> None:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN env var is required.")

    api = HfApi(token=token)
    user = api.whoami()
    print(f"Logged in as: {user.get('name', '<unknown>')}")
    print(user)


if __name__ == "__main__":
    main()


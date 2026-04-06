import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from huggingface_hub import HfApi, create_repo


@dataclass
class ReleaseTarget:
    key: str
    base_model_id: str
    repo_id: str
    artifact_zip: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_targets() -> List[ReleaseTarget]:
    return [
        ReleaseTarget(
            key="3B",
            base_model_id="Qwen/Qwen2.5-3B-Instruct",
            repo_id=os.getenv("HF_REPO_3B", "RthItalia/NanoLLM-Qwen2.5-3B-v3.1"),
            artifact_zip=Path(os.getenv("ARTIFACT_3B", "final_artifact_3B.zip")),
        ),
        ReleaseTarget(
            key="7B",
            base_model_id="Qwen/Qwen2.5-7B-Instruct",
            repo_id=os.getenv("HF_REPO_7B", "RthItalia/NanoLLM-Qwen2.5-7B-v3.1"),
            artifact_zip=Path(os.getenv("ARTIFACT_7B", "final_artifact_7B.zip")),
        ),
        ReleaseTarget(
            key="14B",
            base_model_id="Qwen/Qwen2.5-14B-Instruct",
            repo_id=os.getenv("HF_REPO_14B", "RthItalia/NanoLLM-Qwen2.5-14B-v3.1"),
            artifact_zip=Path(
                os.getenv("ARTIFACT_14B", "final_artifact_Qwen2.5-14B-Instruct_pruned_pass.zip")
            ),
        ),
    ]


def main() -> None:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN env var is required.")

    private = os.getenv("HF_PRIVATE_REPOS", "0").strip().lower() in {"1", "true", "yes", "on"}
    upload_readme = os.getenv("HF_UPLOAD_README", "1").strip().lower() in {"1", "true", "yes", "on"}
    readme_path = Path(os.getenv("HF_RELEASE_README", "NANOHF/README.md"))
    single_repo = os.getenv("HF_REPO_ALL", "").strip()
    targets = get_targets()

    api = HfApi(token=token)
    release_manifest: Dict[str, Dict[str, str]] = {}

    for t in targets:
        zip_path = t.artifact_zip.resolve()
        if not zip_path.exists():
            raise SystemExit(f"Missing artifact for {t.key}: {zip_path}")
        repo_id = single_repo if single_repo else t.repo_id
        path_in_repo = f"{t.key}/{zip_path.name}" if single_repo else zip_path.name
        manifest_path_in_repo = f"{t.key}/release_manifest.json" if single_repo else "release_manifest.json"

        print(f"\n=== {t.key} ===")
        print(f"Repo: {repo_id}")
        print(f"Artifact: {zip_path}")

        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        sha = sha256_file(zip_path)
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {t.key} artifact (Nano v3.1)",
        )

        if upload_readme and readme_path.exists():
            api.upload_file(
                path_or_fileobj=str(readme_path.resolve()),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update README",
            )

        manifest = {
            "nano_release": "v3.1",
            "model_key": t.key,
            "base_model_id": t.base_model_id,
            "artifact_file": zip_path.name,
            "artifact_bytes": str(zip_path.stat().st_size),
            "artifact_sha256": sha,
            "released_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp_manifest = Path(f"_hf_manifest_{t.key}.json")
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            api.upload_file(
                path_or_fileobj=str(tmp_manifest.resolve()),
                path_in_repo=manifest_path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add release manifest",
            )
        finally:
            tmp_manifest.unlink(missing_ok=True)

        release_manifest[t.key] = {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "artifact_file": path_in_repo,
            "artifact_sha256": sha,
        }
        print(f"Done: https://huggingface.co/{repo_id}")

    out = Path("hf_release_v31_result.json")
    out.write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")
    print(f"\nRelease summary written to: {out.resolve()}")


if __name__ == "__main__":
    main()

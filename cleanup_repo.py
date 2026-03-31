import os, shutil
from pathlib import Path

root = Path("e:/NANO")
archive = root / "archive"
archive.mkdir(exist_ok=True)

keep_files = [
    "kaggle_nano_3B_gpu.py",
    "kaggle_nano_cell_gpu.py",
    "kaggle_nano_universal_v3.py",
    "final_artifact_3B.zip",
    "final_artifact_7B.zip",
    "README.md",
    "RESEARCH_DIARY.md",
    "cleanup_repo.py",
    ".git"
]

for f in root.iterdir():
    if f.is_file() and f.name not in keep_files:
        try:
            shutil.move(str(f), str(archive / f.name))
            print(f"Archived: {f.name}")
        except Exception as e:
            print(f"Could not archive {f.name}: {e}")

print("\n--- Repository Cleanup Complete ---")

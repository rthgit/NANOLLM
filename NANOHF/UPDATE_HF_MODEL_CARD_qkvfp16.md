# Update HF Model Card

Use this cell from the pod or any environment that already has `HF_TOKEN` available.

```bash
%%bash
set -euo pipefail
set -a
source /workspace/.env
set +a

python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = "RthItalia/nano_compact_3b_qkvfp16"
token = os.environ["HF_TOKEN"]
card_path = Path("/workspace/NANOLLM/NANOHF/HF_MODEL_CARD_qkvfp16.md")

api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=str(card_path),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Update model card for qkvfp16 winner",
)

print("UPDATED:", repo_id)
print("URL:", f"https://huggingface.co/{repo_id}")
PY
```

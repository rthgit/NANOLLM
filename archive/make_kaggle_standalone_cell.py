from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUNDLE_B64 = (ROOT / "nano_kaggle_bundle.b64").read_text(encoding="ascii")
OUT = ROOT / "kaggle_nano_3b_standalone_cell.py"


CELL_TEMPLATE = """# NANO 3B Kaggle one-cell runner
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


# ===== CONFIG =====
# Use an open HF model by default. Override from Kaggle with:
# os.environ["MODEL_ID"] = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
FORCE_CLEAN = True
THREADS = "8"
RUN_SUFFIXES = [
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
ROW_COUNTS = ["8192", "4096", "2048", "1024", "512"]
BIT_LADDER = ["0", "2", "4", "6"]
# ==================

ROOT = Path("/kaggle/temp/nano_spell")
REPO = ROOT / "repo"
HF_CACHE = ROOT / "hf_cache"
SRC_DIR = ROOT / "source_hf"
RUN_ROOT = ROOT / "full_run"
RANK_DIR = ROOT / "rankings"
PROMPTS_FILE = REPO / "prompts_eval_core.json"
BASE_MAP = ROOT / "canonical_map_all8.json"
BASE_ART = ROOT / "baseline_nano_8bit"
STATUS_FILE = Path("/kaggle/working/nano_status.json")
SUMMARY_FILE = Path("/kaggle/working/nano_summary.json")
ZIP_PATH = ROOT / "nano_bundle.zip"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")


def set_status(step, state, extra=None):
    payload = {
        "ts_utc": int(time.time()),
        "step": step,
        "state": state,
        "extra": extra or {},
    }
    STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {step} -> {state}", flush=True)


def run_live(cmd, cwd=None, log_name=None):
    print("$", " ".join(cmd), flush=True)
    log_path = None
    handle = None
    if log_name:
        log_path = ROOT / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"command failed: {rc}")
    finally:
        if handle:
            handle.close()


def patch_text(path, old, new):
    text = path.read_text(encoding="utf-8")
    if old in text:
        text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def patch_repo_for_kaggle_gpu(repo):
    patch_text(
        repo / "nano_search_subint8.py",
        'from nano_export_from_map_v3 import export_from_map',
        'from nano_export_from_map_v4 import export_from_map',
    )
    patch_text(
        repo / "nano_build_row_scores.py",
        '        weights[module_name] = module.weight.detach().float()',
        '        weights[module_name] = module.weight.detach().float().cpu()',
    )
    patch_text(
        repo / "nano_build_row_scores.py",
        '            usage_sums[module_name].add_(tensor.detach().float().abs().mean(dim=(0, 1)).to(torch.float64))',
        '            usage_sums[module_name].add_(\\n                tensor.detach().float().abs().mean(dim=(0, 1)).to(dtype=torch.float64, device="cpu")\\n            )',
    )
    patch_text(
        repo / "nano_build_row_scores.py",
        '    if "attention_mask" not in enc:\\n        enc["attention_mask"] = torch.ones_like(enc["input_ids"])\\n\\n    with torch.inference_mode():',
        '    if "attention_mask" not in enc:\\n        enc["attention_mask"] = torch.ones_like(enc["input_ids"])\\n    first_device = next(model.parameters()).device\\n    enc = {key: value.to(first_device) for key, value in enc.items()}\\n\\n    with torch.inference_mode():',
    )
    patch_text(
        repo / "nano_build_row_scores.py",
        '                row_saliencies[module_name] = (grad.detach().float() * weights[module_name]).abs().mean(dim=1)',
        '                row_saliencies[module_name] = (\\n                    grad.detach().float().cpu() * weights[module_name]\\n                ).abs().mean(dim=1)',
    )
    patch_text(
        repo / "nano_build_row_scores.py",
        '        device_map={"": "cpu"},',
        '        device_map=("auto" if torch.cuda.is_available() else {"": "cpu"}),',
    )
    patch_text(
        repo / "nano_search_subint8.py",
        '            device_map={"": "cpu"},',
        '            device_map=("auto" if torch.cuda.is_available() else {"": "cpu"}),',
    )
    patch_text(
        repo / "nano_search_subint8.py",
        '            next_tensor = torch.tensor([[next_id]], dtype=cur_ids.dtype)',
        '            next_tensor = torch.tensor([[next_id]], dtype=cur_ids.dtype, device=cur_ids.device)',
    )
    patch_text(
        repo / "nano_search_subint8.py",
        '            cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype)], dim=1)',
        '            cur_mask = torch.cat(\\n                [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)],\\n                dim=1,\\n            )',
    )
    patch_text(
        repo / "nano_search_subint8.py",
        '        if "attention_mask" not in enc:\\n            enc["attention_mask"] = torch.ones_like(enc["input_ids"])\\n\\n        with torch.inference_mode():',
        '        if "attention_mask" not in enc:\\n            enc["attention_mask"] = torch.ones_like(enc["input_ids"])\\n        first_device = next(candidate_model.parameters()).device\\n        enc = {key: value.to(first_device) for key, value in enc.items()}\\n\\n        with torch.inference_mode():',
    )
    patch_text(
        repo / "nano_inference_direct.py",
        '            fresh_module = module.__class__(config, device=torch.device("cpu"))',
        '            fresh_module = module.__class__(config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))',
    )
    patch_text(
        repo / "nano_inference_direct.py",
        '    model = model.to_empty(device="cpu")',
        '    model = model.to_empty(device=("cuda" if torch.cuda.is_available() else "cpu"))',
    )
    (repo / "nano_export_from_map_v3.py").write_text(
        "from nano_export_from_map_v4 import export_from_map\\n",
        encoding="utf-8",
    )


def preflight():
    import torch

    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if torch.cuda.is_available() else []
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("GPU count:", gpu_count, flush=True)
    print("GPU names:", gpu_names, flush=True)
    if torch.cuda.is_available() and gpu_count >= 2:
        for i in [0, 1]:
            x = torch.randn((1024, 1024), device=f"cuda:{i}", dtype=torch.float16)
            y = (x @ x).mean().item()
            print(f"GPU {i} warmup ok, mean={y:.6f}", flush=True)
            del x
        torch.cuda.empty_cache()


for path in [ROOT, HF_CACHE, RANK_DIR]:
    path.mkdir(parents=True, exist_ok=True)

if FORCE_CLEAN:
    for path in [REPO, SRC_DIR, RUN_ROOT, BASE_ART]:
        if path.exists():
            shutil.rmtree(path)
    for path in [BASE_MAP]:
        if path.exists():
            path.unlink()

set_status("extract_bundle", "running")
b64 = \"\"\"__BUNDLE_B64__\"\"\".strip()
ZIP_PATH.write_bytes(base64.b64decode(b64))
REPO.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(REPO)
set_status("extract_bundle", "done", {"repo": str(REPO)})

set_status("patch_repo", "running")
patch_repo_for_kaggle_gpu(REPO)
set_status("patch_repo", "done")

set_status("install_deps", "running")
run_live(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "safetensors", "transformers", "accelerate", "bitsandbytes", "psutil"],
    log_name="pip.log",
)
set_status("install_deps", "done")

set_status("preflight", "running")
preflight()
set_status("preflight", "done")

set_status("download_model", "running", {"model_id": MODEL_ID})
from huggingface_hub import login, snapshot_download

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

SRC_DIR.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(SRC_DIR),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "*.json",
        "*.model",
        "*.txt",
        "model*.safetensors",
        "model.safetensors.index.json",
        "tokenizer*",
        "special_tokens_map.json",
        "generation_config.json",
    ],
)
if not (SRC_DIR / "model.safetensors.index.json").exists():
    raise RuntimeError("download model incomplete: missing model.safetensors.index.json")
set_status("download_model", "done", {"source_dir": str(SRC_DIR)})

set_status("build_base_map", "running")
run_live(
    [
        sys.executable,
        "nano_build_precision_map_v3.py",
        "--model-ref",
        str(SRC_DIR),
        "--prompts-file",
        str(PROMPTS_FILE),
        "--out-json",
        str(BASE_MAP),
        "--dtype",
        "fp16",
        "--max-length",
        "128",
        "--usage-samples",
        "1",
        "--sensitivity-samples",
        "0",
        "--skip-sensitivity",
        "--alpha",
        "0.30",
        "--beta",
        "0.30",
        "--gamma",
        "0.40",
        "--tier-ratios",
        "0.2,0.4,0.2,0.2",
        "--group-by",
        "suffix",
        "--norm-kind",
        "rms",
        "--min-bits",
        "8",
    ],
    cwd=REPO,
    log_name="01_build_map.log",
)
set_status("build_base_map", "done", {"base_map": str(BASE_MAP)})

set_status("export_baseline", "running")
run_live(
    [
        sys.executable,
        "nano_export_from_map_v4.py",
        "--source-dir",
        str(SRC_DIR),
        "--precision-map",
        str(BASE_MAP),
        "--out-dir",
        str(BASE_ART),
        "--embed-bits",
        "8",
    ],
    cwd=REPO,
    log_name="02_export_baseline.log",
)
set_status("export_baseline", "done", {"baseline_artifact": str(BASE_ART)})

set_status("full_sweep", "running")
run_live(
    [
        sys.executable,
        "nano_run_full_model_sweep.py",
        "--source-dir",
        str(SRC_DIR),
        "--base-map",
        str(BASE_MAP),
        "--reference-model",
        str(BASE_ART),
        "--reference-kind",
        "nano",
        "--out-root",
        str(RUN_ROOT),
        "--prompts-file",
        str(PROMPTS_FILE),
        "--layer-start",
        "0",
        "--layer-end",
        "27",
        "--suffixes",
        *RUN_SUFFIXES,
        "--row-counts",
        *ROW_COUNTS,
        "--bit-ladder",
        *BIT_LADDER,
        "--embed-bits",
        "8",
        "--ranking-dir",
        str(RANK_DIR),
        "--ranking-model",
        str(SRC_DIR),
        "--ranking-kind",
        "hf",
        "--ranking-alpha",
        "0.30",
        "--ranking-beta",
        "0.30",
        "--ranking-gamma",
        "0.40",
        "--ranking-skip-sensitivity",
        "--max-length",
        "128",
        "--greedy-steps",
        "4",
        "--max-loss-delta",
        "0.02",
        "--min-next-token-agreement",
        "1.0",
        "--min-greedy-token-agreement",
        "1.0",
        "--min-full-greedy-match",
        "1.0",
        "--num-threads",
        THREADS,
        "--quiet-loads",
        "--candidate-eval-mode",
        "ram",
        "--candidate-map-policy",
        "none",
    ],
    cwd=REPO,
    log_name="03_full_sweep.log",
)
set_status("full_sweep", "done", {"run_root": str(RUN_ROOT)})

summary = {
    "model_id": MODEL_ID,
    "repo": str(REPO),
    "source_dir": str(SRC_DIR),
    "base_map": str(BASE_MAP),
    "baseline_artifact": str(BASE_ART),
    "run_root": str(RUN_ROOT),
    "status_file": str(STATUS_FILE),
    "summary_file": str(RUN_ROOT / "full_sweep_summary.json"),
}
SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
set_status("all", "done", {"summary_file": str(SUMMARY_FILE)})
print("DONE", SUMMARY_FILE, flush=True)
"""


OUT.write_text(CELL_TEMPLATE.replace("__BUNDLE_B64__", BUNDLE_B64), encoding="utf-8")
print(f"Wrote {OUT}")

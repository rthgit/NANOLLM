# ============================================================
# NANO 3B CASCADE RUNNER v3.0 — REASONING-SAFE + BIT PACKING
# ============================================================
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ===== CONFIG =====
MODEL_ID    = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
FORCE_CLEAN = True

# Qwen 3B ha 36 layer (da 0 a 35)
LAYER_START = 0
LAYER_END   = 35

RUN_SUFFIXES = [
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "self_attn.q_proj", "self_attn.k_proj",
    "self_attn.v_proj", "self_attn.o_proj",
]

# Cascata
TRIAL_PLAN: List[Tuple[int, int]] = [
    (2,  512), (2, 1024), (2, 2048), (2, 4096), (2, 8192),
    (4,  512), (4, 1024), (4, 2048), (4, 4096), (4, 8192),
    (6,  512), (6, 1024), (6, 2048), (6, 4096), (6, 8192),
    (8, 0),      # fallback
]

# Prompt difficili per salvaguardare ragionamento e codice
PROMPTS = [
    "Write a Python function to sort a list using bubble sort.",
    "If I have 3 apples and eat 2, how many apples do I have left?",
    "A red shirt is wet. I put it in the bright sun for 2 hours. What happens to the shirt?",
    "Translate to French: I love programming and building artificial intelligence.",
    "Explain what a neural network is in exactly 3 simple sentences.",
]

MAX_LENGTH   = 128
MIN_COSINE_SIMILARITY = 0.99  # Se la logica scende sotto il 99% di somiglianza con l'originale, boccia la compressione!

# ===== PATHS =====
ROOT         = Path("/kaggle/temp/nano_standalone")
HF_CACHE     = ROOT / "hf_cache"
MODEL_CACHE  = ROOT / "source_hf"
ARTIFACT_DIR = Path("/kaggle/working/final_artifact_3B")  # Nome dedicato!
STATUS_FILE  = Path("/kaggle/working/nano_status.json")
SUMMARY_FILE = Path("/kaggle/working/nano_summary.json")
LOCKS_FILE   = Path("/kaggle/working/nano_locks.json")

os.environ["HF_HOME"]                   = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"]        = str(HF_CACHE)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"]      = "0"
os.environ["TOKENIZERS_PARALLELISM"]    = "false"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "huggingface_hub", "safetensors", "transformers", "accelerate", "bitsandbytes"], 
    check=True
)
import torch
import torch.nn as nn
from huggingface_hub import login, snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def set_status(step, state, extra=None):
    p = {"ts_utc": int(time.time()), "step": step, "state": state, "extra": extra or {}}
    STATUS_FILE.write_text(json.dumps(p, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {step} -> {state}", flush=True)

def get_module(root, name):
    cur = root
    for p in name.split("."): cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur

def set_module(root, name, value):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]: parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit(): parent[int(last)] = value
    else: setattr(parent, last, value)

def discover_targets(model, suffixes):
    out = []
    for name, _ in model.named_modules():
        if not any(name.endswith(s) for s in suffixes): continue
        if ".layers." not in name: continue
        try: idx = int(name.split(".layers.")[1].split(".")[0])
        except: continue
        if LAYER_START <= idx <= LAYER_END: out.append(name)
    return sorted(out)

def quantize_symmetric(w, bits):
    qmax = {8: 127.0, 6: 31.0, 4: 7.0, 2: 1.0}[bits]
    max_abs = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = (max_abs / qmax).squeeze(1)
    q = (w.float() / scale.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int8)
    return q.contiguous(), scale.to(torch.float16).contiguous()

GPU = "cuda:0"

class TrueQuantLinear(nn.Module):
    def __init__(self, prot_q, prot_scale, prot_idx, deg_q, deg_scale, deg_idx, out_features, bias=None, bits=8):
        super().__init__()
        self.out_features, self.bits = out_features, int(bits)
        self.register_buffer("prot_q", prot_q.to(device=GPU, dtype=torch.int8))
        self.register_buffer("prot_scale", prot_scale.to(device=GPU, dtype=torch.float16))
        self.register_buffer("prot_idx", prot_idx.to(device=GPU, dtype=torch.long))
        self.register_buffer("deg_q", deg_q.to(device=GPU, dtype=torch.int8))
        self.register_buffer("deg_scale", deg_scale.to(device=GPU, dtype=torch.float16))
        self.register_buffer("deg_idx", deg_idx.to(device=GPU, dtype=torch.long))
        
        if bias is not None: self.register_buffer("bias", bias.to(device=GPU, dtype=torch.float16))
        else: self.bias = None

    def forward(self, x):
        d, dt = x.device, x.dtype; f = x.to(torch.float16).reshape(-1, x.shape[-1])
        o = torch.zeros(f.shape[0], self.out_features, dtype=torch.float16, device=d)
        if self.prot_q.shape[0] > 0:
            w = self.prot_q.to(torch.float16) * self.prot_scale.unsqueeze(1)
            o.index_copy_(1, self.prot_idx, f @ w.t()); del w
        if self.deg_q.shape[0] > 0:
            w = self.deg_q.to(torch.float16) * self.deg_scale.unsqueeze(1)
            o.index_copy_(1, self.deg_idx, f @ w.t()); del w
        if self.bias is not None: o = o + self.bias
        return o.reshape(*x.shape[:-1], self.out_features).to(dt)

    def export_state(self):
        # IL VERO BIT PACKING A 800 MB!
        s = {"bits": self.bits, "out_features": self.out_features,
             "prot_q": self.prot_q.cpu(), "prot_scale": self.prot_scale.cpu(), "prot_idx": self.prot_idx.cpu(),
             "deg_scale": self.deg_scale.cpu(), "deg_idx": self.deg_idx.cpu()}
        if self.bias is not None: s["bias"] = self.bias.cpu()
        
        dq = self.deg_q.cpu()
        if dq.shape[0] == 0:
            s["deg_q"] = dq
        elif self.bits == 2:
            t = (dq + 1).to(torch.uint8)
            pad = (4 - (t.shape[1] % 4)) % 4
            if pad > 0: t = torch.cat([t, torch.zeros((t.shape[0], pad), dtype=torch.uint8)], dim=1)
            t = t.view(t.shape[0], -1, 4)
            s["deg_q_packed"] = (t[:,:,0]) | (t[:,:,1] << 2) | (t[:,:,2] << 4) | (t[:,:,3] << 6)
            s["pad"] = pad
        elif self.bits == 4:
            t = (dq + 7).to(torch.uint8)
            pad = (2 - (t.shape[1] % 2)) % 2
            if pad > 0: t = torch.cat([t, torch.zeros((t.shape[0], pad), dtype=torch.uint8)], dim=1)
            t = t.view(t.shape[0], -1, 2)
            s["deg_q_packed"] = (t[:,:,0]) | (t[:,:,1] << 4)
            s["pad"] = pad
        else:
            s["deg_q"] = dq
            
        return s

_w_cache, _r_cache, tensor_to_file = {}, {}, {}
def _load_tensor(key, device=GPU):
    shard = MODEL_CACHE / tensor_to_file[key]
    with safe_open(str(shard), framework="pt", device=device) as f: return f.get_tensor(key)
def load_weight(mn):
    key = f"{mn}.weight"
    if key not in _w_cache: _w_cache[key] = _load_tensor(key, GPU).to(torch.float16).contiguous()
    return _w_cache[key]
def load_bias(mn):
    key = f"{mn}.bias"
    return _load_tensor(key, GPU).to(torch.float16).contiguous() if key in tensor_to_file else None
def row_importance(mn):
    if mn not in _r_cache:
        w = load_weight(mn) 
        _r_cache[mn] = torch.argsort(w.float().pow(2).mean(1).sqrt(), descending=False)
    return _r_cache[mn]
def clear_cache(mn):
    _w_cache.pop(f"{mn}.weight", None); _r_cache.pop(mn, None)

def build_candidate(mn, bits, prot_rows):
    w, bias, order = load_weight(mn), load_bias(mn), row_importance(mn)
    total = w.shape[0]
    n_prot = min(int(prot_rows), total)
    n_deg  = total - n_prot
    prot_q, prot_s = quantize_symmetric(w[order[n_deg:]], 8)
    deg_q,  deg_s  = quantize_symmetric(w[order[:n_deg]], bits if n_deg > 0 else 8)
    return TrueQuantLinear(
        prot_q=prot_q, prot_scale=prot_s, prot_idx=order[n_deg:],
        deg_q=deg_q, deg_scale=deg_s, deg_idx=order[:n_deg],
        out_features=total, bias=bias, bits=bits)

@torch.inference_mode()
def get_logits(model, prompt, tokenizer):
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).items()}
    return model(**enc).logits[0, -1, :].clone()

@torch.inference_mode()
def evaluate(model, tokenizer, baseline):
    cos_sum = 0.0
    for p in PROMPTS:
        sig, base = get_logits(model, p, tokenizer), baseline[p]
        cos = torch.nn.functional.cosine_similarity(sig.flatten(), base.flatten(), dim=0).item()
        cos_sum += cos
    
    avg_cos = cos_sum / len(PROMPTS)
    ok = avg_cos >= MIN_COSINE_SIMILARITY
    return ok, {"cos": avg_cos}

# ===== MAIN =====
set_status("init", "running")
if FORCE_CLEAN and ROOT.exists(): shutil.rmtree(ROOT)
for p in [ROOT, HF_CACHE]: p.mkdir(parents=True, exist_ok=True)
if torch.cuda.is_available(): free, total = torch.cuda.mem_get_info(0); print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {free/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

if HF_TOKEN: login(token=HF_TOKEN, add_to_git_credential=False)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_CACHE), local_dir_use_symlinks=False,
    allow_patterns=["*.json","*.txt","*.model","tokenizer*","special_tokens_map.json",
                    "generation_config.json","model*.safetensors","model.safetensors.index.json"])
idx_path = MODEL_CACHE / "model.safetensors.index.json"
tensor_to_file = json.loads(idx_path.read_text("utf-8"))["weight_map"]

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CACHE), use_fast=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(str(MODEL_CACHE),
    quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": "cuda:0"}).eval()

set_status("baseline", "running")
baseline = {p: get_logits(model, p, tokenizer) for p in PROMPTS}
set_status("baseline", "done")

targets = discover_targets(model, RUN_SUFFIXES)
locked, exports, pending = {}, {}, list(targets)
set_status("cascade", "running", {"modules": len(pending)})
t0 = time.time()

for ti, (bits, prot) in enumerate(TRIAL_PLAN, 1):
    if not pending: break
    print(f"\n{'='*80}\nTRIAL {ti}/{len(TRIAL_PLAN)} b{bits} k{prot}  "
          f"pending={len(pending)} locked={len(locked)} {time.time()-t0:.0f}s\n{'='*80}", flush=True)
    still = []
    for mi, mn in enumerate(pending, 1):
        print(f"  [{mi}/{len(pending)}] {mn} b{bits} k{prot}", end="", flush=True)
        backup = get_module(model, mn)
        try:
            c = build_candidate(mn, bits, prot)
            set_module(model, mn, c)
            ok, m = evaluate(model, tokenizer, baseline)
        except Exception as e:
            ok, m = False, {"cos":0.0}
            print(f"  ERROR: {str(e)[:120]}", flush=True)
            
        if ok:
            locked[mn] = {"trial":ti,"bits":bits,"keep":prot,**m}
            exports[mn] = c.export_state()
            baseline = {p: get_logits(model, p, tokenizer) for p in PROMPTS}
            print(f"  ✓ {m['cos']*100:.2f}%", flush=True)
        else:
            set_module(model, mn, backup)
            still.append(mn)
            print(f"  ✗ {m.get('cos',0)*100:.2f}%", flush=True)
            
        clear_cache(mn); del backup; gc.collect(); torch.cuda.empty_cache()
    pending = still
    LOCKS_FILE.write_text(json.dumps(locked, indent=2, default=str), "utf-8")

elapsed = time.time() - t0
set_status("cascade", "done", {"locked":len(locked),"pending":len(pending),"sec":int(elapsed)})

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
spec = {"format":"nano-v3.0","base_model_id":MODEL_ID,"locked_count":len(locked),
        "pending_8bit":len(pending),"elapsed_seconds":int(elapsed)}
(ARTIFACT_DIR/"spec.json").write_text(json.dumps(spec,indent=2,default=str),"utf-8")
torch.save(exports, ARTIFACT_DIR/"quantized_modules.pt")

loader_source = '''"""Loader NANO-v3.0 BIT-PACKED"""
import json, torch, torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class TrueQuantLinear(nn.Module):
    def __init__(s, pq, ps, pi, dq, ds, di, of, bias=None, bits=8):
        super().__init__(); s.out_features, s.bits = of, int(bits)
        s.register_buffer("pq", pq.to(torch.int8)); s.register_buffer("ps", ps.to(torch.float16))
        s.register_buffer("pi", pi.to(torch.long)); s.register_buffer("dq", dq.to(torch.int8))
        s.register_buffer("ds", ds.to(torch.float16)); s.register_buffer("di", di.to(torch.long))
        if bias is not None: s.register_buffer("bias", bias.to(torch.float16))
        else: s.bias = None
    def forward(s, x):
        d, dt = x.device, x.dtype; f = x.to(torch.float16).reshape(-1, x.shape[-1])
        o = torch.zeros(f.shape[0], s.out_features, dtype=torch.float16, device=d)
        if s.pq.shape[0] > 0: o.index_copy_(1, s.pi.to(d), f @ (s.pq.to(d,torch.float16)*s.ps.to(d).unsqueeze(1)).t())
        if s.dq.shape[0] > 0: o.index_copy_(1, s.di.to(d), f @ (s.dq.to(d,torch.float16)*s.ds.to(d).unsqueeze(1)).t())
        if s.bias is not None: o = o + s.bias.to(d)
        return o.reshape(*x.shape[:-1], s.out_features).to(dt)

def _set(r, n, v):
    ps = n.split("."); p = r
    for x in ps[:-1]: p = p[int(x)] if x.isdigit() else getattr(p, x)
    if ps[-1].isdigit(): p[int(ps[-1])] = v
    else: setattr(p, ps[-1], v)

def load_artifact(ad):
    d = Path(ad); sp = json.loads((d/"spec.json").read_text("utf-8"))
    st = torch.load(d/"quantized_modules.pt", map_location="cpu")
    m = AutoModelForCausalLM.from_pretrained(sp["base_model_id"],
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": "cuda:0"})
    t = AutoTokenizer.from_pretrained(sp["base_model_id"], use_fast=True)
    if t.pad_token_id is None: t.pad_token = t.eos_token
    for n, s in st.items():
        b = s["bits"]
        if "deg_q_packed" in s:
            pk, pad = s["deg_q_packed"], s["pad"]
            if b == 2:
                dq = torch.stack([pk & 3, (pk >> 2) & 3, (pk >> 4) & 3, (pk >> 6) & 3], dim=-1).view(pk.shape[0], -1)
                if pad > 0: dq = dq[:, :-pad]
                dq = dq.to(torch.int8) - 1
            else:
                dq = torch.stack([pk & 15, (pk >> 4) & 15], dim=-1).view(pk.shape[0], -1)
                if pad > 0: dq = dq[:, :-pad]
                dq = dq.to(torch.int8) - 7
        else: dq = s.get("deg_q", torch.zeros(0,dtype=torch.int8))
            
        _set(m, n, TrueQuantLinear(
            s["prot_q"], s["prot_scale"], s["prot_idx"], dq, s["deg_scale"], s["deg_idx"],
            s["out_features"], s.get("bias"), b).to("cuda:0"))
    return m.eval(), t, sp
'''
(ARTIFACT_DIR/"load_artifact.py").write_text(loader_source, "utf-8")
set_status("zipping", "running")
shutil.make_archive(str(ARTIFACT_DIR), 'zip', str(ARTIFACT_DIR))
set_status("all", "done")
print(f"\nDONE — {len(locked)} compressi, {len(pending)} 8bit, {elapsed/60:.1f}min")
print(f"Artifact: {ARTIFACT_DIR}")
print(f"ZIP File: {ARTIFACT_DIR}.zip")

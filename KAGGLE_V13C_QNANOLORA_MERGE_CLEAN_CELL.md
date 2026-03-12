# KAGGLE V13C + QNANOLORA MERGE (CLEAN)

Obiettivo:
- costruire `V13C` pulito;
- costruire `V13C + QNanoLoRA merged` (merge delta LoRA nei pesi base);
- valutare entrambi nello stesso run con metriche identiche;
- salvare un summary unico in `/kaggle/working`.

```python
# !pip -q install -U transformers peft safetensors

import gc
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "unsloth/Llama-3.2-3B"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

ADAPTER_OVERRIDE = None
# ADAPTER_OVERRIDE = "/kaggle/working/ADAPTERS_3B/blend_teacher_r23_size142_v2_kl_topk_std"

EVAL_PROMPTS = [
    "The most advanced artificial intelligence technology in 2026 is",
    "Explain in simple terms why transformers are good at language modeling.",
    "Write a short helpful answer to a stressed user who cannot sleep.",
    "Give a concise explanation of the Roman Empire's logistics advantage.",
]

GEN_CFG = dict(max_new_tokens=64, do_sample=False, repetition_penalty=1.10, no_repeat_ngram_size=3)
OUT_JSON = Path("/kaggle/working/v13c_qnanolora_merge_clean_eval.json")

ATTN_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_PROJS = {"gate_proj", "up_proj", "down_proj"}
ALL_PROJS = ATTN_PROJS | MLP_PROJS


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mem():
    if not torch.cuda.is_available():
        return "cpu"
    free_b, total_b = torch.cuda.mem_get_info()
    return f"used={(total_b-free_b)/1024**3:.2f} GiB free={free_b/1024**3:.2f} GiB total={total_b/1024**3:.2f} GiB"


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def find_file(filename: str, required=True):
    hits = sorted(Path("/kaggle/working").rglob(filename))
    hits += sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        return str(hits[0])
    if required:
        raise FileNotFoundError(f"{filename} non trovato in /kaggle/working o /kaggle/input")
    return None


def find_adapter_dir(required=True):
    if ADAPTER_OVERRIDE:
        p = Path(ADAPTER_OVERRIDE)
        ok = p.exists() and ((p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists())
        if not ok:
            raise FileNotFoundError(f"ADAPTER_OVERRIDE non valido: {ADAPTER_OVERRIDE}")
        print("[ADAPTER] override:", str(p))
        return str(p)

    roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    cands = []
    for root in roots:
        if not root.exists():
            continue
        for cfg in root.rglob("adapter_config.json"):
            d = cfg.parent
            if not ((d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()):
                continue
            s = str(d).lower()
            score = 0
            if "v2_kl_topk" in s:
                score += 60
            if "r23_size142" in s:
                score += 40
            if "phase1_dream" in s:
                score += 20
            if "qnanolora" in s:
                score += 10
            cands.append((score, len(str(d)), str(d)))

    if not cands:
        if required:
            raise FileNotFoundError("Adapter non trovato")
        return None

    cands.sort(key=lambda x: (-x[0], x[1]))
    chosen = cands[0][2]
    print("[ADAPTER] chosen:", chosen)
    return chosen


def resolve_owner_and_module(layer_block, proj_name):
    if proj_name in ATTN_PROJS:
        owner = layer_block.self_attn
    elif proj_name in MLP_PROJS:
        owner = layer_block.mlp
    else:
        raise ValueError(proj_name)
    return owner, getattr(owner, proj_name)


def text_metrics(text: str):
    t = text.strip()
    words = t.split()
    low = t.lower()
    han = bool(re.search(r"[\u4e00-\u9fff]", t))
    short = len(words) < 8
    loop = bool(re.search(r"(,){10,}", t) or re.search(r"([A-Za-z]{2,8})\1{4,}", t))
    generic = any(m in low for m in ["in this article", "essay", "the post", "answer:"])
    uniq = len(set(w.lower() for w in words)) / len(words) if words else 0.0
    weird = bool(re.search(r"[\ufffd]", t) or re.search(r"[A-Za-z]{3,}[A-Z]{3,}", t))
    semantic_fail = short or weird or loop
    return {
        "han": han,
        "short": short,
        "loop": loop,
        "generic": generic,
        "uniq": uniq,
        "semantic_fail": semantic_fail,
    }


def eval_model(model, tokenizer, prompts, label):
    print("\n" + "=" * 100)
    print(label)
    print("CUDA before eval:", mem())
    rows, lat = [], 0.0

    with torch.inference_mode():
        for i, prompt in enumerate(prompts, 1):
            inp = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
            t0 = time.time()
            out = model.generate(
                **inp,
                max_new_tokens=GEN_CFG["max_new_tokens"],
                do_sample=GEN_CFG["do_sample"],
                repetition_penalty=GEN_CFG["repetition_penalty"],
                no_repeat_ngram_size=GEN_CFG["no_repeat_ngram_size"],
                pad_token_id=tokenizer.eos_token_id,
            )
            dt = time.time() - t0
            lat += dt
            txt = tokenizer.decode(out[0], skip_special_tokens=True)
            if txt.startswith(prompt):
                txt = txt[len(prompt):].strip()
            m = text_metrics(txt)
            rows.append({"prompt": prompt, "text": txt, "time_s": dt, **m})
            print("-" * 100)
            print(f"[{i}] {dt:.2f}s | sem_fail={m['semantic_fail']} han={m['han']} loop={m['loop']} short={m['short']} | uniq={m['uniq']:.4f}")
            print(txt if txt else "<EMPTY>")

    summary = {
        "label": label,
        "han": sum(int(r["han"]) for r in rows),
        "loop": sum(int(r["loop"]) for r in rows),
        "short": sum(int(r["short"]) for r in rows),
        "generic": sum(int(r["generic"]) for r in rows),
        "semantic_fail_count": sum(int(r["semantic_fail"]) for r in rows),
        "semantic_fail": sum(int(r["semantic_fail"]) for r in rows) >= 1,
        "uniq": sum(r["uniq"] for r in rows) / len(rows),
        "latency_avg_s": lat / len(rows),
        "cuda": mem(),
        "rows": rows,
    }
    print("SUMMARY:", {k: v for k, v in summary.items() if k != "rows"})
    return summary


class Nano3FactorLinear(nn.Module):
    def __init__(self, in_features, out_features, bond, rank, dtype=torch.float16, bias=False):
        super().__init__()
        self.c1 = nn.Parameter(torch.zeros(in_features, bond, dtype=dtype))
        self.c2 = nn.Parameter(torch.zeros(bond, rank, dtype=dtype))
        self.c3 = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype)) if bias else None

    def forward(self, x):
        y = x.float() @ self.c1.float() @ self.c2.float() @ self.c3.float()
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype)


class BlendLinear(nn.Module):
    def __init__(self, base_linear, nano_linear, alpha):
        super().__init__()
        self.base_linear = base_linear
        self.nano_linear = nano_linear
        self.alpha = float(alpha)

    def forward(self, x):
        b = self.base_linear(x)
        n = self.nano_linear(x)
        return (((1.0 - self.alpha) * b.float()) + (self.alpha * n.float())).to(b.dtype)


class SharedNanoLexicon(nn.Module):
    def __init__(self, vocab_size, hidden_size, bond, rank, dtype=torch.float32):
        super().__init__()
        self.C1 = nn.Embedding(vocab_size, bond, dtype=dtype)
        self.C2 = nn.Parameter(torch.zeros(bond, rank, dtype=dtype))
        self.C3 = nn.Parameter(torch.zeros(rank, hidden_size, dtype=dtype))

    def embed(self, token_ids):
        return self.C1(token_ids).float() @ self.C2.float() @ self.C3.float()

    def full_logits(self, hidden):
        z = hidden.float() @ self.C3.float().T @ self.C2.float().T
        return z @ self.C1.weight.float().T


class BlendEmbedding(nn.Module):
    def __init__(self, base_embedding, shared_lex, alpha):
        super().__init__()
        self.base_embedding = base_embedding
        self.shared_lex = shared_lex
        self.alpha = float(alpha)

    def forward(self, input_ids):
        b = self.base_embedding(input_ids)
        n = self.shared_lex.embed(input_ids)
        return (((1.0 - self.alpha) * b.float()) + (self.alpha * n.float())).to(b.dtype)


class CurrentBestHead(nn.Module):
    def __init__(self, base_lm_head, shared_lex, alpha_head_fixed):
        super().__init__()
        self.base_lm_head = base_lm_head
        self.shared_lex = shared_lex
        self.alpha_head_fixed = float(alpha_head_fixed)

    def forward(self, hidden):
        d = self.base_lm_head(hidden).float()
        t = self.shared_lex.full_logits(hidden).float()
        return (d + self.alpha_head_fixed * t).to(d.dtype)


LORA_KEY_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_(A|B)(?:\.default)?\.weight$"
)


def load_lora_state(adapter_dir: str):
    p = Path(adapter_dir)
    st = p / "adapter_model.safetensors"
    bn = p / "adapter_model.bin"

    if st.exists():
        state = safe_load(str(st), device="cpu")
    elif bn.exists():
        state = torch.load(str(bn), map_location="cpu")
    else:
        raise FileNotFoundError(f"Nessun adapter_model.* in {adapter_dir}")

    cfg_path = p / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    rank = int(cfg.get("r", 23))
    lora_alpha = float(cfg.get("lora_alpha", 46))
    scaling = lora_alpha / rank
    target_modules = cfg.get("target_modules", None)
    if not target_modules:
        target_modules = sorted({m.group(3) for k in state.keys() for m in [LORA_KEY_RE.match(k)] if m})
    target_modules = [m for m in target_modules if m in ALL_PROJS]
    return state, cfg, scaling, target_modules


def make_lora_index(state):
    idx = {}
    for k, v in state.items():
        m = LORA_KEY_RE.match(k)
        if not m:
            continue
        li = int(m.group(1))
        owner = m.group(2)
        proj = m.group(3)
        ab = m.group(4)
        key = (li, owner, proj)
        if key not in idx:
            idx[key] = {}
        idx[key][ab] = v
    return idx


def merge_lora_into_model(model, lora_state, scaling, target_modules):
    idx = make_lora_index(lora_state)
    merged = 0
    skipped = 0
    num_layers = len(model.model.layers)

    with torch.no_grad():
        for li in range(num_layers):
            lb = model.model.layers[li]
            for proj in target_modules:
                if proj not in ALL_PROJS:
                    continue
                owner_name = "self_attn" if proj in ATTN_PROJS else "mlp"
                key = (li, owner_name, proj)
                if key not in idx or "A" not in idx[key] or "B" not in idx[key]:
                    skipped += 1
                    continue
                owner, linear = resolve_owner_and_module(lb, proj)
                lora_A = idx[key]["A"].float()
                lora_B = idx[key]["B"].float()
                delta = (lora_B @ lora_A) * scaling
                linear.weight.data.add_(delta.to(linear.weight.dtype).to(linear.weight.device))
                merged += 1

    print(f"[LORA MERGE] merged={merged} skipped={skipped} scaling={scaling:.6f}")
    return {"merged": merged, "skipped": skipped, "scaling": scaling, "target_modules": target_modules}


def parse_group_keys(v13c_state):
    out = {}
    for k in v13c_state.keys():
        m = re.match(r"^(\d+)_(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", k)
        if m:
            out[(int(m.group(1)), m.group(2))] = True
    return out


def build_v13c_model(model, phase2c, stage5, phasec, phase3a, v13c_ckpt):
    num_layers = int(phase2c.get("num_layers", len(model.model.layers)))
    base_targets = tuple(phase2c.get("target_projections", ("q_proj", "k_proj", "v_proj", "o_proj")))
    base_targets = tuple(t for t in base_targets if t in ALL_PROJS)

    rank_block = int(phase2c["rank"])
    bond_block = int(phase2c["bond"])

    alpha_internal = float(stage5.get("alpha_internal_best", stage5.get("best_alpha", {}).get("internal")))
    alpha_head_fixed = float(stage5.get("alpha_head_fixed", stage5.get("best_alpha", {}).get("head")))
    alpha_embed = float(phasec.get("embed_alpha_best", phasec.get("best_alpha", {}).get("embed", 0.0175)))
    alpha_group = float(v13c_ckpt["best_group_alpha"])

    group_start = max(0, num_layers - 6)
    group_layers = set(range(group_start, num_layers))
    v13c_state = v13c_ckpt["state"]
    group_keys = parse_group_keys(v13c_state)

    stage5_wrap = {}
    wi = 0
    for li in range(num_layers):
        for proj in base_targets:
            stage5_wrap[(li, proj)] = f"wrap_{wi}"
            wi += 1

    dev = next(model.parameters()).device
    wrappers_to_build = set((li, p) for li in range(num_layers) for p in base_targets)
    wrappers_to_build |= set(group_keys.keys())

    with torch.no_grad():
        for li, proj in sorted(wrappers_to_build):
            if li >= num_layers or proj not in ALL_PROJS:
                continue

            lb = model.model.layers[li]
            owner, orig = resolve_owner_and_module(lb, proj)
            nano = Nano3FactorLinear(orig.in_features, orig.out_features, bond_block, rank_block, dtype=DTYPE, bias=(orig.bias is not None)).to(dev)

            loaded = False
            wrap_id = stage5_wrap.get((li, proj))
            if wrap_id is not None and wrap_id in stage5["nano_state"]:
                s = stage5["nano_state"][wrap_id]
                nano.c1.copy_(s["c1"].to(DTYPE).to(dev))
                nano.c2.copy_(s["c2"].to(DTYPE).to(dev))
                nano.c3.copy_(s["c3"].to(DTYPE).to(dev))
                if nano.bias is not None and s.get("bias") is not None:
                    nano.bias.copy_(s["bias"].to(DTYPE).to(dev))
                loaded = True

            v13_key = f"{li}_{proj}"
            if v13_key in v13c_state:
                s = v13c_state[v13_key]
                nano.c1.copy_(s["c1"].to(DTYPE).to(dev))
                nano.c2.copy_(s["c2"].to(DTYPE).to(dev))
                nano.c3.copy_(s["c3"].to(DTYPE).to(dev))
                if nano.bias is not None and s.get("bias") is not None:
                    nano.bias.copy_(s["bias"].to(DTYPE).to(dev))
                loaded = True

            if not loaded:
                continue

            is_group_mlp = (li in group_layers and proj in MLP_PROJS and v13_key in v13c_state)
            alpha_use = alpha_group if is_group_mlp else alpha_internal
            setattr(owner, proj, BlendLinear(orig, nano, alpha_use).to(dev))

    lex_state = phasec.get("shared_lex_state", phase3a.get("shared_lex_state") if phase3a else None)
    if lex_state is None:
        raise RuntimeError("shared_lex_state mancante in phasec/phase3a")

    vocab_size, hidden_size = model.model.embed_tokens.weight.shape
    lex_bond = int(lex_state["C1"].shape[1])
    lex_rank = int(lex_state["C2"].shape[1])
    shared_lex = SharedNanoLexicon(vocab_size, hidden_size, lex_bond, lex_rank, dtype=torch.float32).to(dev)
    with torch.no_grad():
        shared_lex.C1.weight.copy_(lex_state["C1"].float().to(dev))
        shared_lex.C2.copy_(lex_state["C2"].float().to(dev))
        shared_lex.C3.copy_(lex_state["C3"].float().to(dev))

    base_embed = model.model.embed_tokens
    base_head = model.lm_head
    model.model.embed_tokens = BlendEmbedding(base_embed, shared_lex, alpha_embed).to(dev)
    model.lm_head = CurrentBestHead(base_head, shared_lex, alpha_head_fixed).to(dev)

    info = {
        "num_layers": num_layers,
        "base_targets": list(base_targets),
        "alpha_internal": alpha_internal,
        "alpha_head_fixed": alpha_head_fixed,
        "alpha_embed": alpha_embed,
        "alpha_group": alpha_group,
    }
    return model, info


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": 0} if DEVICE == "cuda" else "cpu",
    ).eval()


seed_everything(SEED)
print("DEVICE:", DEVICE, "| DTYPE:", DTYPE)
print("CUDA start:", mem())

PHASE2C_CKPT = find_file("best_fp16.pt")
STAGE5_CKPT = find_file("stage5_internal_best.pt")
PHASEC_CKPT = find_file("phase_c_embed_best.pt")
PHASE3A_CKPT = find_file("phase3a_shared_lex_head_only_best.pt", required=False)
V13C_CKPT = find_file("grouped_mlp_hidden_anchor_v13c_best.pt")
ADAPTER_DIR = find_adapter_dir(required=True)

print("PHASE2C:", PHASE2C_CKPT)
print("STAGE5 :", STAGE5_CKPT)
print("PHASEC :", PHASEC_CKPT)
print("PHASE3A:", PHASE3A_CKPT)
print("V13C   :", V13C_CKPT)
print("ADAPTER:", ADAPTER_DIR)

phase2c = torch.load(PHASE2C_CKPT, map_location="cpu")
stage5 = torch.load(STAGE5_CKPT, map_location="cpu")
phasec = torch.load(PHASEC_CKPT, map_location="cpu")
phase3a = torch.load(PHASE3A_CKPT, map_location="cpu") if PHASE3A_CKPT else {}
v13c_ckpt = torch.load(V13C_CKPT, map_location="cpu")

lora_state, lora_cfg, lora_scaling, lora_targets = load_lora_state(ADAPTER_DIR)
print("[ADAPTER CFG]", {"r": lora_cfg.get("r"), "lora_alpha": lora_cfg.get("lora_alpha"), "targets": lora_targets})

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

cuda_cleanup()
print("\nBuilding BASELINE_V13C...")
model_base = load_base_model()
model_base, info_base = build_v13c_model(model_base, phase2c, stage5, phasec, phase3a, v13c_ckpt)
baseline_summary = eval_model(model_base, tokenizer, EVAL_PROMPTS, "BASELINE_V13C")
del model_base
cuda_cleanup()

print("\nBuilding V13C_PLUS_QNANOLORA...")
model_merge = load_base_model()
merge_info = merge_lora_into_model(model_merge, lora_state, lora_scaling, lora_targets)
model_merge, info_merge = build_v13c_model(model_merge, phase2c, stage5, phasec, phase3a, v13c_ckpt)
merged_summary = eval_model(model_merge, tokenizer, EVAL_PROMPTS, "V13C_PLUS_QNANOLORA")
del model_merge
cuda_cleanup()

gap_uniq = float(baseline_summary["uniq"] - merged_summary["uniq"])
comparison = {
    "gap_uniq_vs_baseline": gap_uniq,
    "merged_semantic_fail_count": merged_summary["semantic_fail_count"],
    "baseline_semantic_fail_count": baseline_summary["semantic_fail_count"],
    "merged_short": merged_summary["short"],
    "baseline_short": baseline_summary["short"],
}

decision = {
    "candidate_label": "V13C_PLUS_QNANOLORA",
    "pass_hard_gates": (
        merged_summary["semantic_fail_count"] == 0
        and merged_summary["short"] == 0
        and gap_uniq <= 0.03
    ),
    "close_enough_vs_baseline": gap_uniq <= 0.03,
    "better_or_equal_semantic": merged_summary["semantic_fail_count"] <= baseline_summary["semantic_fail_count"],
}

payload = {
    "model_id": MODEL_ID,
    "device": DEVICE,
    "dtype": str(DTYPE),
    "v13c_checkpoint": V13C_CKPT,
    "adapter_dir": ADAPTER_DIR,
    "adapter_cfg": {
        "r": lora_cfg.get("r"),
        "lora_alpha": lora_cfg.get("lora_alpha"),
        "lora_dropout": lora_cfg.get("lora_dropout"),
        "target_modules": lora_targets,
    },
    "merge_info": merge_info,
    "baseline_build_info": info_base,
    "merged_build_info": info_merge,
    "baseline": {k: v for k, v in baseline_summary.items() if k != "rows"},
    "merged": {k: v for k, v in merged_summary.items() if k != "rows"},
    "comparison": comparison,
    "decision": decision,
}

OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("\n" + "#" * 100)
print("COMPARISON")
print(f"baseline uniq: {baseline_summary['uniq']:.4f}")
print(f"merged   uniq: {merged_summary['uniq']:.4f}")
print(f"gap uniq      : {gap_uniq:.4f}")
print(f"baseline semantic_fail_count: {baseline_summary['semantic_fail_count']}")
print(f"merged   semantic_fail_count: {merged_summary['semantic_fail_count']}")
print(f"baseline short: {baseline_summary['short']}")
print(f"merged   short: {merged_summary['short']}")
print("\nDECISION:", decision)
print("saved:", OUT_JSON)
print("CUDA end:", mem())
```

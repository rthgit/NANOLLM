# KAGGLE COMPRESSION METHODS BENCHMARK (UNIFIED)

Obiettivo:
- confrontare metodi di compressione nello stesso protocollo:
  - RAW FP16
  - BnB INT8
  - BnB NF4 4-bit
  - V13C + QNanoLoRA merged
  - opzionali AWQ/GPTQ se artifact disponibili

Output:
- `/kaggle/working/compression_methods_benchmark_3b.json`

```python
# !pip -q install -U transformers bitsandbytes safetensors

import gc
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "unsloth/Llama-3.2-3B"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADAPTER_DIR = "/kaggle/working/ADAPTERS_3B/blend_teacher_r23_size142_v2_kl_topk_std"

EVAL_PROMPTS = [
    "The most advanced artificial intelligence technology in 2026 is",
    "Explain in simple terms why transformers are good at language modeling.",
    "Write a short helpful answer to a stressed user who cannot sleep.",
    "Give a concise explanation of the Roman Empire's logistics advantage.",
    "Describe in plain language what overfitting is and how to reduce it.",
    "Why do vector databases matter for retrieval-augmented generation?",
    "Write a concise email declining a meeting while staying polite.",
    "Summarize the main difference between supervised and unsupervised learning.",
    "Give a practical 3-step plan to start learning Python this week.",
    "Explain why caching can improve API latency under load.",
    "What made Roman roads strategically important for empire control?",
    "Provide a calm response to a user panicking about a production outage.",
    "Explain attention mechanism as if speaking to a high-school student.",
    "List three common mistakes when writing prompts for LLMs.",
    "Give a concise explanation of gradient descent.",
    "Write a short answer: how to prepare for a technical interview in 30 days.",
    "Explain the tradeoff between model quality and inference cost.",
    "Why is evaluation with fixed prompts important for reproducibility?",
    "Give a brief comparison between CPU and GPU workloads for ML inference.",
    "Write a helpful response to: 'I keep procrastinating and feel stuck.'",
]

NLL_TEXTS = [
    "Transformers attend across context and predict next tokens with parallel computation.",
    "Roman logistics relied on roads, depots, standardized units, and disciplined administration.",
    "A good support response is calm, concrete, and action-oriented.",
    "Perplexity is derived from cross-entropy and lower is better.",
    "Deterministic evaluation requires fixed prompts and fixed decoding parameters.",
    "Overfitting means memorizing training patterns that do not generalize.",
    "Regularization and better validation reduce overfitting risk.",
    "RAG combines retrieval with generation to improve factual grounding.",
    "Caching avoids repeated expensive computation for frequent requests.",
    "Inference latency depends on batch size, hardware, and sequence length.",
    "GPU acceleration helps matrix-heavy workloads in deep learning.",
    "Model quality and cost usually move in opposite directions.",
    "Clear prompt constraints reduce output variance and failure modes.",
    "Reliable deployment needs explicit gates and reproducible artifacts.",
    "Production checks should include semantic quality and stability signals.",
    "LoRA adapters update low-rank matrices while freezing base weights.",
    "Evaluation sets should cover instruction following and factual clarity.",
    "Text generation quality can diverge from likelihood metrics.",
    "Safety gates should fail fast on empty or repetitive outputs.",
    "Versioned reports are required for auditability and rollback.",
]

GEN_CFG = dict(max_new_tokens=64, do_sample=False, repetition_penalty=1.10, no_repeat_ngram_size=3)
OUT_JSON = Path("/kaggle/working/compression_methods_benchmark_3b.json")

ATTN_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_PROJS = {"gate_proj", "up_proj", "down_proj"}
ALL_PROJS = ATTN_PROJS | MLP_PROJS

LORA_RE = re.compile(
    r"^base_model\\.model\\.model\\.layers\\.(\\d+)\\.(self_attn|mlp)\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\\.lora_(A|B)(?:\\.default)?\\.weight$"
)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mem():
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    free_b, total_b = torch.cuda.mem_get_info()
    used_b = total_b - free_b
    return {
        "used_gib": used_b / 1024**3,
        "free_gib": free_b / 1024**3,
        "total_gib": total_b / 1024**3,
    }


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def find_file(name, required=True):
    hits = sorted(Path("/kaggle/working").rglob(name)) + sorted(Path("/kaggle/input").rglob(name))
    if hits:
        return str(hits[0])
    if required:
        raise FileNotFoundError(f"{name} non trovato")
    return None


def resolve_owner_and_linear(layer, proj):
    if proj in ATTN_PROJS:
        owner = layer.self_attn
    elif proj in MLP_PROJS:
        owner = layer.mlp
    else:
        raise ValueError(proj)
    return owner, getattr(owner, proj)


def text_metrics(text):
    t = text.strip()
    words = t.split()
    han = bool(re.search(r"[\u4e00-\u9fff]", t))
    short = len(words) < 8
    loop = bool(re.search(r"(,){10,}", t) or re.search(r"([A-Za-z]{2,8})\1{4,}", t))
    generic = any(x in t.lower() for x in ["in this article", "essay", "the post", "answer:"])
    uniq = len(set(w.lower() for w in words)) / len(words) if words else 0.0
    weird = bool(re.search(r"[\ufffd]", t))
    semantic_fail = short or loop or weird
    return dict(han=han, short=short, loop=loop, generic=generic, uniq=uniq, semantic_fail=semantic_fail)


def eval_generate(model, tok, label):
    print("\n" + "=" * 100)
    print(label)
    print("CUDA:", mem())
    rows, lat = [], 0.0
    with torch.inference_mode():
        for i, p in enumerate(EVAL_PROMPTS, 1):
            inp = tok(p, return_tensors="pt").to(next(model.parameters()).device)
            t0 = time.time()
            out = model.generate(
                **inp,
                max_new_tokens=GEN_CFG["max_new_tokens"],
                do_sample=GEN_CFG["do_sample"],
                repetition_penalty=GEN_CFG["repetition_penalty"],
                no_repeat_ngram_size=GEN_CFG["no_repeat_ngram_size"],
                pad_token_id=tok.eos_token_id,
            )
            dt = time.time() - t0
            lat += dt
            txt = tok.decode(out[0], skip_special_tokens=True)
            if txt.startswith(p):
                txt = txt[len(p):].strip()
            m = text_metrics(txt)
            rows.append({"prompt": p, "text": txt, "time_s": dt, **m})
            print(f"[{i:02d}] {dt:.2f}s | fail={m['semantic_fail']} | uniq={m['uniq']:.4f}")

    return {
        "uniq": sum(r["uniq"] for r in rows) / len(rows),
        "semantic_fail_count": sum(int(r["semantic_fail"]) for r in rows),
        "short": sum(int(r["short"]) for r in rows),
        "loop": sum(int(r["loop"]) for r in rows),
        "han": sum(int(r["han"]) for r in rows),
        "generic": sum(int(r["generic"]) for r in rows),
        "latency_avg_s": lat / len(rows),
        "rows": rows,
    }


def eval_nll(model, tok, texts, max_len=256):
    losses = []
    with torch.inference_mode():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(next(model.parameters()).device)
            out = model(**enc, labels=enc["input_ids"])
            losses.append(float(out.loss.item()))
    avg = sum(losses) / len(losses)
    ppl = math.exp(min(20.0, avg))
    return {"avg_nll": avg, "ppl": ppl, "rows": len(losses)}


class Nano3FactorLinear(nn.Module):
    def __init__(self, in_f, out_f, bond, rank, dtype=torch.float16, bias=False):
        super().__init__()
        self.c1 = nn.Parameter(torch.zeros(in_f, bond, dtype=dtype))
        self.c2 = nn.Parameter(torch.zeros(bond, rank, dtype=dtype))
        self.c3 = nn.Parameter(torch.zeros(rank, out_f, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_f, dtype=dtype)) if bias else None

    def forward(self, x):
        y = x.float() @ self.c1.float() @ self.c2.float() @ self.c3.float()
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype)


class BlendLinear(nn.Module):
    def __init__(self, base, nano, alpha):
        super().__init__()
        self.base_linear = base
        self.nano_linear = nano
        self.alpha = float(alpha)

    def forward(self, x):
        b = self.base_linear(x)
        n = self.nano_linear(x)
        return (((1.0 - self.alpha) * b.float()) + (self.alpha * n.float())).to(b.dtype)


class SharedNanoLexicon(nn.Module):
    def __init__(self, vocab, hidden, bond, rank, dtype=torch.float32):
        super().__init__()
        self.C1 = nn.Embedding(vocab, bond, dtype=dtype)
        self.C2 = nn.Parameter(torch.zeros(bond, rank, dtype=dtype))
        self.C3 = nn.Parameter(torch.zeros(rank, hidden, dtype=dtype))

    def embed(self, ids):
        return self.C1(ids).float() @ self.C2.float() @ self.C3.float()

    def full_logits(self, h):
        z = h.float() @ self.C3.float().T @ self.C2.float().T
        return z @ self.C1.weight.float().T


class BlendEmbedding(nn.Module):
    def __init__(self, base, lex, alpha):
        super().__init__()
        self.base_embedding = base
        self.shared_lex = lex
        self.alpha = float(alpha)

    def forward(self, ids):
        b = self.base_embedding(ids)
        n = self.shared_lex.embed(ids)
        return (((1.0 - self.alpha) * b.float()) + (self.alpha * n.float())).to(b.dtype)


class CurrentBestHead(nn.Module):
    def __init__(self, base, lex, alpha):
        super().__init__()
        self.base_lm_head = base
        self.shared_lex = lex
        self.alpha_head_fixed = float(alpha)

    def forward(self, h):
        d = self.base_lm_head(h).float()
        t = self.shared_lex.full_logits(h).float()
        return (d + self.alpha_head_fixed * t).to(d.dtype)


def load_lora(adapter_dir):
    p = Path(adapter_dir)
    st = p / "adapter_model.safetensors"
    bn = p / "adapter_model.bin"
    if st.exists():
        state = safe_load(str(st), device="cpu")
    elif bn.exists():
        state = torch.load(str(bn), map_location="cpu")
    else:
        raise FileNotFoundError(f"Nessun adapter_model.* in {adapter_dir}")

    cfg_p = p / "adapter_config.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
    r = int(cfg.get("r", 23))
    a = float(cfg.get("lora_alpha", 46))
    scaling = a / r

    targets = cfg.get("target_modules")
    if not targets:
        targets = sorted({m.group(3) for k in state.keys() for m in [LORA_RE.match(k)] if m})
    targets = [t for t in targets if t in ALL_PROJS]
    return state, cfg, scaling, targets


def index_lora(state):
    idx = {}
    for k, v in state.items():
        m = LORA_RE.match(k)
        if not m:
            continue
        li, owner, proj, ab = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        idx.setdefault((li, owner, proj), {})[ab] = v
    return idx


def merge_lora(model, lora_state, scaling, targets):
    idx = index_lora(lora_state)
    merged, skipped = 0, 0
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            for proj in targets:
                owner_name = "self_attn" if proj in ATTN_PROJS else "mlp"
                key = (li, owner_name, proj)
                if key not in idx or "A" not in idx[key] or "B" not in idx[key]:
                    skipped += 1
                    continue
                _, lin = resolve_owner_and_linear(layer, proj)
                A = idx[key]["A"].float()
                B = idx[key]["B"].float()
                delta = (B @ A) * scaling
                lin.weight.data.add_(delta.to(lin.weight.dtype).to(lin.weight.device))
                merged += 1
    return {"merged": merged, "skipped": skipped, "scaling": scaling}


def normalize_v13(v13, phase2c=None):
    if "best_group_alpha" not in v13:
        v13["best_group_alpha"] = float(v13.get("group_alpha", v13.get("alpha", 0.17208333333333334)))

    st = v13.get("state", {})
    if not isinstance(st, dict):
        st = {}
    if len(st) == 0:
        ts = v13.get("tail_state", {})
        if isinstance(ts, dict) and len(ts) > 0:
            st = ts

    key_re = re.compile(r"^\\d+_(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    if len(st) > 0 and any(key_re.match(str(k)) for k in st.keys()):
        v13["state"] = st
        return v13

    if len(st) > 0 and all(str(k).startswith("wrap_") for k in st.keys()):
        wraps = sorted(st.items(), key=lambda kv: int(str(kv[0]).split("_")[1]))
        layers = [int(x) for x in v13.get("target_layers", [])]
        orders = []
        if phase2c is not None:
            tp = [p for p in phase2c.get("target_projections", []) if p in ALL_PROJS]
            if tp:
                orders.append(tp)
        orders += [
            ["gate_proj", "up_proj", "down_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ]
        mapped = {}
        for order in orders:
            if layers and len(layers) * len(order) == len(wraps):
                i = 0
                for li in layers:
                    for p in order:
                        mapped[f"{li}_{p}"] = wraps[i][1]
                        i += 1
                break
        if mapped:
            st = mapped

    v13["state"] = st if isinstance(st, dict) else {}
    return v13


def parse_v13_keys(state):
    out = set()
    for k in state.keys():
        m = re.match(r"^(\\d+)_(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", k)
        if m:
            out.add((int(m.group(1)), m.group(2)))
    return out


def build_v13c(model, phase2c, stage5, phasec, phase3a, v13):
    num_layers = int(phase2c.get("num_layers", len(model.model.layers)))
    base_targets = tuple(phase2c.get("target_projections", ("q_proj", "k_proj", "v_proj", "o_proj")))
    base_targets = tuple(t for t in base_targets if t in ALL_PROJS)

    rank_block = int(phase2c["rank"])
    bond_block = int(phase2c["bond"])
    alpha_internal = float(stage5.get("alpha_internal_best", stage5.get("best_alpha", {}).get("internal")))
    alpha_head = float(stage5.get("alpha_head_fixed", stage5.get("best_alpha", {}).get("head")))
    alpha_embed = float(phasec.get("embed_alpha_best", phasec.get("best_alpha", {}).get("embed", 0.0175)))
    alpha_group = float(v13["best_group_alpha"])

    v13_state = v13["state"]
    v13_keys = parse_v13_keys(v13_state)

    stage5_map = {}
    wi = 0
    for li in range(num_layers):
        for p in base_targets:
            stage5_map[(li, p)] = f"wrap_{wi}"
            wi += 1

    group_layers = set(range(max(0, num_layers - 6), num_layers))
    to_build = set((li, p) for li in range(num_layers) for p in base_targets) | v13_keys

    dev = next(model.parameters()).device
    with torch.no_grad():
        for li, p in sorted(to_build):
            if li >= num_layers or p not in ALL_PROJS:
                continue
            layer = model.model.layers[li]
            owner, orig = resolve_owner_and_linear(layer, p)
            nano = Nano3FactorLinear(orig.in_features, orig.out_features, bond_block, rank_block, dtype=torch.float16 if DEVICE == "cuda" else torch.float32, bias=(orig.bias is not None)).to(dev)

            loaded = False
            w = stage5_map.get((li, p))
            if w and w in stage5["nano_state"]:
                s = stage5["nano_state"][w]
                dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                nano.c1.copy_(s["c1"].to(dtype).to(dev))
                nano.c2.copy_(s["c2"].to(dtype).to(dev))
                nano.c3.copy_(s["c3"].to(dtype).to(dev))
                if nano.bias is not None and s.get("bias") is not None:
                    nano.bias.copy_(s["bias"].to(dtype).to(dev))
                loaded = True

            vk = f"{li}_{p}"
            if vk in v13_state:
                s = v13_state[vk]
                dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                nano.c1.copy_(s["c1"].to(dtype).to(dev))
                nano.c2.copy_(s["c2"].to(dtype).to(dev))
                nano.c3.copy_(s["c3"].to(dtype).to(dev))
                if nano.bias is not None and s.get("bias") is not None:
                    nano.bias.copy_(s["bias"].to(dtype).to(dev))
                loaded = True

            if not loaded:
                continue

            is_group_mlp = (li in group_layers and p in MLP_PROJS and vk in v13_state)
            alpha = alpha_group if is_group_mlp else alpha_internal
            setattr(owner, p, BlendLinear(orig, nano, alpha).to(dev))

    lex = phasec.get("shared_lex_state", phase3a.get("shared_lex_state") if phase3a else None)
    if lex is None:
        raise RuntimeError("shared_lex_state mancante")

    vocab, hidden = model.model.embed_tokens.weight.shape
    lex_bond = int(lex["C1"].shape[1])
    lex_rank = int(lex["C2"].shape[1])
    shared = SharedNanoLexicon(vocab, hidden, lex_bond, lex_rank, dtype=torch.float32).to(dev)
    with torch.no_grad():
        shared.C1.weight.copy_(lex["C1"].float().to(dev))
        shared.C2.copy_(lex["C2"].float().to(dev))
        shared.C3.copy_(lex["C3"].float().to(dev))

    be = model.model.embed_tokens
    bh = model.lm_head
    model.model.embed_tokens = BlendEmbedding(be, shared, alpha_embed).to(dev)
    model.lm_head = CurrentBestHead(bh, shared, alpha_head).to(dev)

    return model


def load_model_fp16():
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": 0} if DEVICE == "cuda" else "cpu",
    ).eval()


def load_model_int8():
    if DEVICE != "cuda":
        raise RuntimeError("INT8 benchmark richiede CUDA")
    qcfg = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=qcfg,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()


def load_model_nf4():
    if DEVICE != "cuda":
        raise RuntimeError("NF4 benchmark richiede CUDA")
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=qcfg,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()


def bench_method(name, builder_fn):
    print("\n" + "#" * 100)
    print("METHOD:", name)
    cleanup()
    start_mem = mem()
    t0 = time.time()
    model = builder_fn()
    load_s = time.time() - t0
    after_mem = mem()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    gen = eval_generate(model, tok, name)
    nll = eval_nll(model, tok, NLL_TEXTS)

    out = {
        "method": name,
        "ok": True,
        "load_time_s": load_s,
        "mem_before_load": start_mem,
        "mem_after_load": after_mem,
        "gen": {k: v for k, v in gen.items() if k != "rows"},
        "nll": nll,
    }

    del model
    cleanup()
    return out


seed_everything(SEED)
print("DEVICE:", DEVICE)
print("CUDA start:", mem())

results = {}

# shared artifacts for V13C+QNanoLoRA
phase2c = torch.load(find_file("best_fp16.pt"), map_location="cpu")
stage5 = torch.load(find_file("stage5_internal_best.pt"), map_location="cpu")
phasec = torch.load(find_file("phase_c_embed_best.pt"), map_location="cpu")
phase3a_p = find_file("phase3a_shared_lex_head_only_best.pt", required=False)
phase3a = torch.load(phase3a_p, map_location="cpu") if phase3a_p else {}

v13_p = find_file("grouped_mlp_hidden_anchor_v13c_best.pt", required=False)
if v13_p is None:
    v13_p = find_file("phase2_standalone_tail_v13_hard_guard_best.pt", required=True)
v13 = normalize_v13(torch.load(v13_p, map_location="cpu"), phase2c=phase2c)
assert len(v13["state"]) > 0, "v13 state vuoto"

lora_state, lora_cfg, lora_scaling, lora_targets = load_lora(ADAPTER_DIR)

# method builders
builders = {
    "RAW_FP16": lambda: load_model_fp16(),
    "RAW_INT8_BNB": lambda: load_model_int8(),
    "RAW_NF4_4BIT_BNB": lambda: load_model_nf4(),
    "V13C_QNANOLORA_MERGED": lambda: build_v13c(
        (lambda m: (merge_lora(m, lora_state, lora_scaling, lora_targets), m)[1])(load_model_fp16()),
        phase2c, stage5, phasec, phase3a, v13,
    ),
}

for name, fn in builders.items():
    try:
        results[name] = bench_method(name, fn)
    except Exception as e:
        results[name] = {
            "method": name,
            "ok": False,
            "error": str(e),
        }
        print(f"[SKIP/FAIL] {name}: {e}")

# optional AWQ/GPTQ local artifacts
optional = {
    "RAW_AWQ_LOCAL": "*awq*",
    "RAW_GPTQ_LOCAL": "*gptq*",
}
for name, patt in optional.items():
    try:
        hits = list(Path("/kaggle/working").rglob(patt)) + list(Path("/kaggle/input").rglob(patt))
        model_dirs = [h for h in hits if h.is_dir() and (h / "config.json").exists()]
        if not model_dirs:
            results[name] = {"method": name, "ok": False, "error": "artifact non trovato"}
            continue
        p = str(sorted(model_dirs, key=lambda x: len(str(x)))[0])
        def _load_opt(path=p):
            return AutoModelForCausalLM.from_pretrained(path, device_map={"": 0} if DEVICE == "cuda" else "cpu", low_cpu_mem_usage=True).eval()
        results[name] = bench_method(name, _load_opt)
        results[name]["source_path"] = p
    except Exception as e:
        results[name] = {"method": name, "ok": False, "error": str(e)}

# deltas vs RAW_FP16
if results.get("RAW_FP16", {}).get("ok"):
    raw = results["RAW_FP16"]
    for k, v in results.items():
        if not v.get("ok"):
            continue
        v["delta_vs_raw_fp16"] = {
            "uniq_delta": v["gen"]["uniq"] - raw["gen"]["uniq"],
            "nll_delta": v["nll"]["avg_nll"] - raw["nll"]["avg_nll"],
            "ppl_ratio": (v["nll"]["ppl"] / raw["nll"]["ppl"]) if raw["nll"]["ppl"] > 0 else None,
            "semantic_fail_delta": v["gen"]["semantic_fail_count"] - raw["gen"]["semantic_fail_count"],
            "latency_ratio": (v["gen"]["latency_avg_s"] / raw["gen"]["latency_avg_s"]) if raw["gen"]["latency_avg_s"] > 0 else None,
        }

OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\n" + "#" * 100)
print("BENCH SUMMARY")
for name, v in results.items():
    if not v.get("ok"):
        print(f"- {name}: FAIL/SKIP ({v.get('error')})")
        continue
    d = v.get("delta_vs_raw_fp16", {})
    print(
        f"- {name}: uniq={v['gen']['uniq']:.4f} nll={v['nll']['avg_nll']:.4f} "
        f"ppl={v['nll']['ppl']:.2f} sem_fail={v['gen']['semantic_fail_count']} "
        f"lat={v['gen']['latency_avg_s']:.2f}s "
        f"| d_uniq={d.get('uniq_delta')} d_nll={d.get('nll_delta')} ppl_ratio={d.get('ppl_ratio')}"
    )

print("saved:", OUT_JSON)
print("CUDA end:", mem())
```

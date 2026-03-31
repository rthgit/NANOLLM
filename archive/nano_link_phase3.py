#!/usr/bin/env python3
"""
Nano-Link Phase 3 — Top-K Sweep + Stress Tests + Ablation
==========================================================
Battery of tests designed to break or validate the Nano-Link thesis:
1. Top-K sweep (k=1,2,4,8) — optimal candidate set
2. Correlated / adversarial patterns — robustness
3. Hard scaling to 10,000 patterns — does cost law hold?
4. Parameter sensitivity — is C robust or fragile?
5. Full ablation — isolating each contribution
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
GRID = 8
LR = 0.15
PROP_ITERS = 3
TRIALS = 3  # fewer trials for speed on massive configs

C_CELL   = 1.0
C_VERT   = 0.3
C_PROP   = 0.2
C_LAYER  = 0.5
C_INHIB  = 0.1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase3")

# ─────────────────────────────────────────────
# Pattern Generators
# ─────────────────────────────────────────────
TEMPLATES = [
    np.array([[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],
              [1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],
              [1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1]], dtype=np.int8),
    np.array([[1,1,1,1,1,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,1,0],
              [1,1,1,1,1,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],
              [1,0,0,0,0,0,1,0],[1,1,1,1,1,1,0,0]], dtype=np.int8),
    np.array([[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0],[0,0,1,1,1,1,0,0]], dtype=np.int8),
    np.array([[1,1,1,1,1,0,0,0],[1,0,0,0,0,1,0,0],[1,0,0,0,0,0,1,0],
              [1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,1,0],
              [1,0,0,0,0,1,0,0],[1,1,1,1,1,0,0,0]], dtype=np.int8),
    np.array([[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],
              [1,1,1,1,1,1,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1]], dtype=np.int8),
]


def gen_standard(n, rng):
    """Same as Phase 1/2 — random perturbations of templates."""
    pats = [t.copy() for t in TEMPLATES]
    while len(pats) < n:
        base = TEMPLATES[rng.integers(len(TEMPLATES))]
        v = base.copy()
        for r, c in rng.integers(0, GRID, size=(rng.integers(2, 6), 2)):
            v[r, c] = 1 - v[r, c]
        if not any(np.array_equal(v, p) for p in pats):
            pats.append(v)
    return pats[:n]


def gen_correlated(n, rng):
    """
    Adversarial: patterns differ by only 1-2 bits.
    Maximally confusing for any retrieval system.
    """
    base = rng.integers(0, 2, size=(GRID, GRID)).astype(np.int8)
    pats = [base.copy()]
    while len(pats) < n:
        v = pats[rng.integers(len(pats))].copy()
        flips = rng.integers(1, 3)  # 1-2 bit difference
        for r, c in rng.integers(0, GRID, size=(flips, 2)):
            v[r, c] = 1 - v[r, c]
        if not any(np.array_equal(v, p) for p in pats):
            pats.append(v)
    return pats[:n]


def gen_sparse(n, rng):
    """Sparse patterns — density ~15% (real-world-like)."""
    pats = []
    while len(pats) < n:
        p = (rng.random((GRID, GRID)) < 0.15).astype(np.int8)
        if np.sum(p) >= 3 and not any(np.array_equal(p, q) for q in pats):
            pats.append(p)
    return pats[:n]


def gen_dense(n, rng):
    """Dense patterns — density ~85%."""
    pats = []
    while len(pats) < n:
        p = (rng.random((GRID, GRID)) < 0.85).astype(np.int8)
        if not any(np.array_equal(p, q) for q in pats):
            pats.append(p)
    return pats[:n]


def add_noise(pattern, level, rng):
    noisy = pattern.copy()
    n_flip = max(1, int(pattern.size * level))
    idx = rng.choice(pattern.size, size=n_flip, replace=False)
    flat = noisy.ravel()
    flat[idx] = 1 - flat[idx]
    return noisy


# ─────────────────────────────────────────────
# Unified NanoLink with configurable Top-K
# ─────────────────────────────────────────────
class NanoLinkTopK:
    """Slots + competitive retrieval with configurable top-K."""

    def __init__(self, top_k=5, prop_iters=PROP_ITERS, lr=LR,
                 inhib_strength=1.0, label=""):
        self.top_k = top_k
        self.prop_iters = prop_iters
        self.lr = lr
        self.inhib_strength = inhib_strength
        self.layers = []
        self.cost = 0.0
        self.label = label or f"NL-k{top_k}"

    @property
    def NAME(self):
        return self.label

    def reset_cost(self):
        self.cost = 0.0

    def store(self, pattern):
        mask = pattern.astype(np.float64)
        nano = np.clip(mask * self.lr, 0.0, 1.0)
        self.layers.append((pattern.copy(), nano))

    def query(self, partial):
        q = partial.astype(np.float64)
        active = int(np.sum(q > 0))
        N = len(self.layers)

        # Pass 1: coarse scoring
        scores = np.zeros(N)
        for i, (pat, nano) in enumerate(self.layers):
            scores[i] = np.sum(q * nano)
            self.cost += active * C_VERT
            self.cost += C_LAYER

        # Inhibition: keep top-K
        k = min(self.top_k, N)
        top_idx = np.argpartition(scores, -k)[-k:]
        self.cost += N * C_INHIB * self.inhib_strength

        # Pass 2: fine scoring with propagation (only on survivors)
        best_score = -1.0
        best_pat = self.layers[top_idx[0]][0]

        for idx in top_idx:
            pat, nano = self.layers[idx]
            beta = q * nano
            self.cost += active * C_CELL
            self.cost += active * C_VERT

            for _ in range(self.prop_iters):
                nb = np.zeros_like(beta)
                nb[1:, :]  += beta[:-1, :]
                nb[:-1, :] += beta[1:, :]
                nb[:, 1:]  += beta[:, :-1]
                nb[:, :-1] += beta[:, 1:]
                count = np.ones_like(beta) * 4
                count[0, :] -= 1; count[-1, :] -= 1
                count[:, 0] -= 1; count[:, -1] -= 1
                beta = beta + nb / count
                self.cost += GRID * GRID * C_PROP

            fine = np.sum(beta * q)
            if fine > best_score:
                best_score = fine
                best_pat = pat

        return best_pat.copy()


class LinearMemory:
    NAME = "Linear"

    def __init__(self):
        self.patterns = []
        self.cost = 0.0

    def reset_cost(self):
        self.cost = 0.0

    def store(self, p):
        self.patterns.append(p.copy())

    def query(self, partial):
        best_d = float("inf")
        best_p = self.patterns[0]
        for p in self.patterns:
            d = np.sum(p != partial)
            self.cost += p.size * C_CELL
            if d < best_d:
                best_d = d
                best_p = p
        return best_p


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def bit_acc(a, b):
    return float(np.mean(a == b))

def exact_acc(a, b):
    return 1.0 if np.array_equal(a, b) else 0.0

def utility(ba, cost):
    return ba / np.log1p(cost)


# ─────────────────────────────────────────────
# Generic experiment runner
# ─────────────────────────────────────────────
def run_one(mem, patterns, noise_level, rng):
    """Store patterns, query with noise, measure."""
    for p in patterns:
        mem.store(p)
    mem.reset_cost()

    ea_list, ba_list = [], []
    for p in patterns:
        noisy = add_noise(p, noise_level, rng)
        r = mem.query(noisy)
        ea_list.append(exact_acc(p, r))
        ba_list.append(bit_acc(p, r))

    ba_val = np.mean(ba_list)
    return {
        "exact": np.mean(ea_list),
        "bit": ba_val,
        "cost": mem.cost,
        "util": utility(ba_val, mem.cost),
    }


def run_multi_trial(make_mem, make_patterns, n_pat, noise, trials=TRIALS):
    """Average across trials."""
    results = []
    for t in range(trials):
        rng = np.random.default_rng(42 + t * 1000 + n_pat + int(noise * 100))
        pats = make_patterns(n_pat, rng)
        mem = make_mem()
        results.append(run_one(mem, pats, noise, rng))

    avg = {}
    for k in results[0]:
        avg[k] = np.mean([r[k] for r in results])
    return avg


# ═════════════════════════════════════════════
# TEST 1: Top-K Sweep
# ═════════════════════════════════════════════
def test_topk_sweep():
    print("\n" + "="*70)
    print("  TEST 1: Top-K Sweep (k=1,2,4,8,16,ALL) @ 1000 patterns, 30% noise")
    print("="*70)

    K_VALUES = [1, 2, 4, 8, 16, 1000]  # 1000 = effectively ALL (=slots-only)
    K_LABELS = ["k=1", "k=2", "k=4", "k=8", "k=16", "ALL"]
    n, noise = 1000, 0.30

    results = []
    for k, label in zip(K_VALUES, K_LABELS):
        print(f"  {label:>6} ...", end="", flush=True)
        r = run_multi_trial(
            lambda k_=k, l_=label: NanoLinkTopK(top_k=k_, label=f"NL-{l_}"),
            gen_standard, n, noise)
        r["k"] = k
        r["label"] = label
        results.append(r)
        print(f"  bit={r['bit']:.4f}  cost={r['cost']:>12.0f}  util={r['util']:.6f}")

    # Add Linear baseline
    print(f"  {'Lin':>6} ...", end="", flush=True)
    r = run_multi_trial(LinearMemory, gen_standard, n, noise)
    r["k"] = -1
    r["label"] = "Linear"
    results.append(r)
    print(f"  bit={r['bit']:.4f}  cost={r['cost']:>12.0f}  util={r['util']:.6f}")

    return results


# ═════════════════════════════════════════════
# TEST 2: Dataset Robustness
# ═════════════════════════════════════════════
def test_dataset_robustness():
    print("\n" + "="*70)
    print("  TEST 2: Dataset Robustness — 500 patterns, 30% noise")
    print("="*70)

    datasets = [
        ("Standard",    gen_standard),
        ("Correlated",  gen_correlated),
        ("Sparse-15%",  gen_sparse),
        ("Dense-85%",   gen_dense),
    ]
    n, noise = 500, 0.30

    results = []
    for ds_name, gen_fn in datasets:
        row = {"dataset": ds_name}
        for sys_name, make_mem in [
            ("NL-k5",  lambda: NanoLinkTopK(top_k=5)),
            ("NL-k8",  lambda: NanoLinkTopK(top_k=8, label="NL-k8")),
            ("Linear", LinearMemory),
        ]:
            print(f"  {ds_name:>12} × {sys_name:>8} ...", end="", flush=True)
            r = run_multi_trial(make_mem, gen_fn, n, noise)
            row[f"{sys_name}_bit"] = r["bit"]
            row[f"{sys_name}_cost"] = r["cost"]
            row[f"{sys_name}_util"] = r["util"]
            print(f"  bit={r['bit']:.4f}  cost={r['cost']:>12.0f}")
        results.append(row)

    return results


# ═════════════════════════════════════════════
# TEST 3: Hard Scaling (up to 10,000)
# ═════════════════════════════════════════════
def test_hard_scaling():
    print("\n" + "="*70)
    print("  TEST 3: Hard Scaling — noise=30%, k=5")
    print("="*70)

    SIZES = [100, 500, 1000, 2000, 5000]
    noise = 0.30
    results = []

    for n in SIZES:
        row = {"n": n}
        for sys_name, make_mem in [
            ("NL-k5", lambda: NanoLinkTopK(top_k=5)),
            ("Linear", LinearMemory),
        ]:
            print(f"  n={n:<6} × {sys_name:>8} ...", end="", flush=True)
            r = run_multi_trial(make_mem, gen_standard, n, noise, trials=2)
            row[f"{sys_name}_bit"] = r["bit"]
            row[f"{sys_name}_cost"] = r["cost"]
            row[f"{sys_name}_util"] = r["util"]
            print(f"  bit={r['bit']:.4f}  cost={r['cost']:>12.0f}")
        results.append(row)

    return results


# ═════════════════════════════════════════════
# TEST 4: Parameter Sensitivity
# ═════════════════════════════════════════════
def test_param_sensitivity():
    print("\n" + "="*70)
    print("  TEST 4: Parameter Sensitivity — 500 patterns, 30% noise")
    print("="*70)

    n, noise = 500, 0.30
    results = []

    # Sweep learning rate
    for lr in [0.05, 0.10, 0.15, 0.25, 0.40]:
        print(f"  lr={lr:.2f} ...", end="", flush=True)
        r = run_multi_trial(
            lambda lr_=lr: NanoLinkTopK(top_k=5, lr=lr_, label=f"lr={lr_}"),
            gen_standard, n, noise)
        r["param"] = "lr"
        r["value"] = lr
        results.append(r)
        print(f"  bit={r['bit']:.4f}  util={r['util']:.6f}")

    # Sweep propagation iterations
    for pi in [0, 1, 2, 3, 5, 8]:
        print(f"  prop_iters={pi} ...", end="", flush=True)
        r = run_multi_trial(
            lambda pi_=pi: NanoLinkTopK(top_k=5, prop_iters=pi_, label=f"pi={pi_}"),
            gen_standard, n, noise)
        r["param"] = "prop_iters"
        r["value"] = pi
        results.append(r)
        print(f"  bit={r['bit']:.4f}  util={r['util']:.6f}")

    # Sweep inhibition strength
    for inh in [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]:
        print(f"  inhib={inh:.2f} ...", end="", flush=True)
        r = run_multi_trial(
            lambda i_=inh: NanoLinkTopK(top_k=5, inhib_strength=i_, label=f"inh={i_}"),
            gen_standard, n, noise)
        r["param"] = "inhib"
        r["value"] = inh
        results.append(r)
        print(f"  bit={r['bit']:.4f}  util={r['util']:.6f}")

    return results


# ═════════════════════════════════════════════
# TEST 5: Ablation
# ═════════════════════════════════════════════
def test_ablation():
    print("\n" + "="*70)
    print("  TEST 5: Ablation — 1000 patterns, 30% noise")
    print("="*70)

    n, noise = 1000, 0.30
    configs = [
        ("Slots-only (k=ALL)",     lambda: NanoLinkTopK(top_k=1000, inhib_strength=0.0, label="Slots-only")),
        ("Slots+weak-WTA (k=50)",  lambda: NanoLinkTopK(top_k=50, inhib_strength=0.5, label="k50-weak")),
        ("Slots+med-WTA (k=8)",    lambda: NanoLinkTopK(top_k=8, inhib_strength=1.0, label="k8-med")),
        ("Slots+strong-WTA (k=2)", lambda: NanoLinkTopK(top_k=2, inhib_strength=2.0, label="k2-strong")),
        ("Slots+WTA (k=5)",        lambda: NanoLinkTopK(top_k=5, inhib_strength=1.0, label="k5-std")),
        ("WTA-only (k=1)",         lambda: NanoLinkTopK(top_k=1, inhib_strength=1.0, label="k1-wta")),
        ("Linear",                 LinearMemory),
    ]

    results = []
    for label, make_mem in configs:
        print(f"  {label:>30} ...", end="", flush=True)
        r = run_multi_trial(make_mem, gen_standard, n, noise)
        r["label"] = label
        results.append(r)
        print(f"  bit={r['bit']:.4f}  cost={r['cost']:>12.0f}  util={r['util']:.6f}")

    return results


# ═════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════
def plot_all(topk, robust, scaling, sensitivity, ablation):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#aaa", "ytick.color": "#aaa",
        "grid.color": "#333", "grid.alpha": 0.5, "font.size": 11,
        "legend.facecolor": "#16213e", "legend.edgecolor": "#444",
    })

    # ── 1. Top-K sweep ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    labels = [r["label"] for r in topk]
    bits   = [r["bit"] for r in topk]
    costs  = [r["cost"] for r in topk]
    utils  = [r["util"] for r in topk]
    colors = ["#e94560" if "Lin" in l else "#16c79a" for l in labels]

    ax1.bar(labels, bits, color=colors, edgecolor="#eee")
    ax1.set_title("Bit Accuracy", fontweight="bold"); ax1.set_ylim(0.5, 1.05)
    for i, v in enumerate(bits):
        ax1.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(labels, costs, color=colors, edgecolor="#eee")
    ax2.set_title("Cost", fontweight="bold"); ax2.set_yscale("log")

    ax3.bar(labels, utils, color=colors, edgecolor="#eee")
    ax3.set_title("Utility", fontweight="bold")
    for i, v in enumerate(utils):
        ax3.text(i, v+max(utils)*0.01, f"{v:.5f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Top-K Sweep (1000 pat, 30% noise)", fontsize=14, fontweight="bold", color="#e94560")
    for ax in (ax1, ax2, ax3):
        ax.grid(axis="y")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "topk_sweep.png"), dpi=150)
    plt.close(fig)

    # ── 2. Dataset robustness ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ds_names = [r["dataset"] for r in robust]
    for ax, (metric, title) in zip(axes, [
        ("_bit", "Bit Accuracy"), ("_cost", "Cost (log)"), ("_util", "Utility")
    ]):
        x = np.arange(len(ds_names))
        w = 0.25
        for i, (sys, color) in enumerate([("NL-k5", "#0f3460"), ("NL-k8", "#16c79a"), ("Linear", "#533483")]):
            vals = [r.get(f"{sys}{metric}", 0) for r in robust]
            ax.bar(x + i*w, vals, w, label=sys, color=color, edgecolor="#eee")
        ax.set_xticks(x + w); ax.set_xticklabels(ds_names, rotation=15, ha="right")
        ax.set_title(title, fontweight="bold")
        if "cost" in metric.lower():
            ax.set_yscale("log")
        ax.legend(fontsize=8); ax.grid(axis="y")

    fig.suptitle("Dataset Robustness (500 pat, 30% noise)", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "dataset_robustness.png"), dpi=150)
    plt.close(fig)

    # ── 3. Hard scaling ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    ns = [r["n"] for r in scaling]
    for sys, color, ls in [("NL-k5", "#16c79a", "o-"), ("Linear", "#e94560", "s--")]:
        bits_s  = [r[f"{sys}_bit"] for r in scaling]
        costs_s = [r[f"{sys}_cost"] for r in scaling]
        utils_s = [r[f"{sys}_util"] for r in scaling]
        ax1.plot(ns, bits_s,  ls, color=color, label=sys, linewidth=2, markersize=7)
        ax2.plot(ns, costs_s, ls, color=color, label=sys, linewidth=2, markersize=7)
        ax3.plot(ns, utils_s, ls, color=color, label=sys, linewidth=2, markersize=7)

    ax1.set_title("Bit Accuracy", fontweight="bold"); ax1.set_ylim(0.4, 1.05)
    ax2.set_title("Cost (log-log)", fontweight="bold"); ax2.set_xscale("log"); ax2.set_yscale("log")
    ax3.set_title("Utility", fontweight="bold")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Patterns"); ax.legend(); ax.grid(True)

    fig.suptitle("Hard Scaling (30% noise)", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "hard_scaling.png"), dpi=150)
    plt.close(fig)

    # ── 4. Parameter sensitivity ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    param_names = ["lr", "prop_iters", "inhib"]
    param_labels = ["Learning Rate", "Propagation Iterations", "Inhibition Strength"]
    for ax, pname, plabel in zip(axes, param_names, param_labels):
        sub = [r for r in sensitivity if r["param"] == pname]
        vals = [r["value"] for r in sub]
        bits_p = [r["bit"] for r in sub]
        utils_p = [r["util"] for r in sub]
        ax.plot(vals, bits_p, "o-", color="#16c79a", label="Bit Acc", linewidth=2, markersize=7)
        ax2t = ax.twinx()
        ax2t.plot(vals, utils_p, "s--", color="#e94560", label="Utility", linewidth=2, markersize=7)
        ax2t.tick_params(axis="y", labelcolor="#e94560")
        ax.set_xlabel(plabel); ax.set_title(plabel, fontweight="bold")
        ax.grid(True)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2t.get_legend_handles_labels()
        ax.legend(lines1+lines2, labs1+labs2, fontsize=8, loc="lower left")

    fig.suptitle("Parameter Sensitivity (500 pat, 30% noise)", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "param_sensitivity.png"), dpi=150)
    plt.close(fig)

    # ── 5. Ablation ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    labels_a = [r["label"].split("(")[0].strip() for r in ablation]
    bits_a  = [r["bit"] for r in ablation]
    costs_a = [r["cost"] for r in ablation]
    utils_a = [r["util"] for r in ablation]
    colors_a = ["#533483" if "Lin" in l else "#16c79a" for l in labels_a]

    ax1.barh(labels_a, bits_a, color=colors_a, edgecolor="#eee")
    ax1.set_title("Bit Accuracy", fontweight="bold"); ax1.set_xlim(0.5, 1.05)

    ax2.barh(labels_a, costs_a, color=colors_a, edgecolor="#eee")
    ax2.set_title("Cost", fontweight="bold"); ax2.set_xscale("log")

    ax3.barh(labels_a, utils_a, color=colors_a, edgecolor="#eee")
    ax3.set_title("Utility", fontweight="bold")

    fig.suptitle("Ablation (1000 pat, 30% noise)", fontsize=14, fontweight="bold", color="#e94560")
    for ax in (ax1, ax2, ax3):
        ax.grid(axis="x")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "ablation.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Charts saved to: {RESULTS_DIR}")


# ═════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════
def print_grand_summary(topk, robust, scaling, sensitivity, ablation):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"{'PHASE 3 — GRAND SUMMARY':^80}")
    print(sep)

    # Top-K best
    best_k = max(topk, key=lambda r: r["util"])
    print(f"\n  BEST Top-K config: {best_k['label']}")
    print(f"    bit_acc={best_k['bit']:.4f}  cost={best_k['cost']:.0f}  util={best_k['util']:.6f}")

    lin = [r for r in topk if r["label"] == "Linear"][0]
    print(f"  vs Linear: bit_acc={lin['bit']:.4f}  cost={lin['cost']:.0f}  util={lin['util']:.6f}")
    print(f"  Cost saving: {lin['cost']/max(best_k['cost'],1):.1f}×")

    # Scaling
    print(f"\n  SCALING (30% noise, NL-k5):")
    for row in scaling:
        ratio = row["Linear_cost"] / max(row["NL-k5_cost"], 1)
        print(f"    n={row['n']:<6}  NL bit={row['NL-k5_bit']:.4f}  Lin bit={row['Linear_bit']:.4f}  "
              f"cost ratio={ratio:.1f}×")

    # Robustness
    print(f"\n  DATASET ROBUSTNESS (NL-k5 vs Linear bit_acc):")
    for row in robust:
        nl  = row["NL-k5_bit"]
        lin_val = row["Linear_bit"]
        gap = nl - lin_val
        print(f"    {row['dataset']:>12}: NL={nl:.4f}  Lin={lin_val:.4f}  gap={gap:+.4f}")

    # Ablation winner
    best_abl = max(ablation, key=lambda r: r["util"])
    print(f"\n  ABLATION WINNER: {best_abl['label']}")
    print(f"    bit_acc={best_abl['bit']:.4f}  cost={best_abl['cost']:.0f}  util={best_abl['util']:.6f}")

    # Parameter sensitivity: best for each param
    print(f"\n  BEST PARAMETERS:")
    for pname in ["lr", "prop_iters", "inhib"]:
        sub = [r for r in sensitivity if r["param"] == pname]
        best = max(sub, key=lambda r: r["util"])
        print(f"    {pname}: best_value={best['value']}  util={best['util']:.6f}")

    print(f"\n{sep}")


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  NANO-LINK PHASE 3")
    print("  Top-K + Stress Tests + Ablation")
    print("=" * 70)

    t0 = time.time()

    topk        = test_topk_sweep()
    robust      = test_dataset_robustness()
    scaling     = test_hard_scaling()
    sensitivity = test_param_sensitivity()
    ablation    = test_ablation()

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_grand_summary(topk, robust, scaling, sensitivity, ablation)
    plot_all(topk, robust, scaling, sensitivity, ablation)
    print("  Done.")

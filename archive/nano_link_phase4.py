#!/usr/bin/env python3
"""
Nano-Link Phase 4 — Final Validation
=====================================
1. Semi-realistic datasets (binarized embeddings, clustered, fingerprints)
2. Full sparsity sweep (5%–90%)
3. Stronger baselines: raw dot-product, LSH, inverted index, Bloom-like filter

The decisive test: is Nano-Link genuinely novel, or a rediscovery
of standard coarse filtering in a new costume?
"""

import os, time, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = 8
CELLS = GRID * GRID  # 64 bits per pattern
LR = 0.15
C_CELL = 1.0; C_VERT = 0.3; C_PROP = 0.2; C_LAYER = 0.5; C_INHIB = 0.1
TRIALS = 3
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase4")

# ─────────────────────────────────────────────
# Dataset Generators
# ─────────────────────────────────────────────

def gen_binarized_embeddings(n, rng, dim=64):
    """Simulate binarized neural embeddings: sample normal vectors, threshold at 0."""
    vecs = rng.standard_normal((n * 3, dim))
    binary = (vecs > 0).astype(np.int8)
    # deduplicate
    seen = set()
    pats = []
    for row in binary:
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            pats.append(row.reshape(GRID, GRID))
        if len(pats) >= n:
            break
    # fill if needed
    while len(pats) < n:
        v = (rng.standard_normal(dim) > 0).astype(np.int8).reshape(GRID, GRID)
        pats.append(v)
    return pats[:n]


def gen_clustered(n, rng, n_clusters=10, dim=64):
    """K cluster centers with small perturbations — realistic category structure."""
    centers = (rng.standard_normal((n_clusters, dim)) > 0).astype(np.int8)
    pats = []
    seen = set()
    while len(pats) < n:
        c = centers[rng.integers(n_clusters)]
        p = c.copy()
        flip = rng.integers(1, 6)  # 1-5 bit flips from center
        idx = rng.choice(dim, size=flip, replace=False)
        p[idx] = 1 - p[idx]
        key = p.tobytes()
        if key not in seen:
            seen.add(key)
            pats.append(p.reshape(GRID, GRID))
    return pats[:n]


def gen_fingerprint(n, rng, dim=64):
    """Simulated binary fingerprints: structured blocks with noise."""
    pats = []
    seen = set()
    while len(pats) < n:
        p = np.zeros(dim, dtype=np.int8)
        # 3-5 random "features" — contiguous blocks of bits
        n_feats = rng.integers(3, 6)
        for _ in range(n_feats):
            start = rng.integers(0, dim - 4)
            length = rng.integers(2, 6)
            p[start:min(start+length, dim)] = 1
        # add light noise
        noise_idx = rng.choice(dim, size=rng.integers(0, 4), replace=False)
        p[noise_idx] = 1 - p[noise_idx]
        key = p.tobytes()
        if key not in seen:
            seen.add(key)
            pats.append(p.reshape(GRID, GRID))
    return pats[:n]


def gen_with_density(density):
    """Factory: generate patterns at exact density level."""
    def gen(n, rng, dim=64):
        pats = []
        seen = set()
        while len(pats) < n:
            p = (rng.random(dim) < density).astype(np.int8)
            if np.sum(p) < 1:
                continue
            key = p.tobytes()
            if key not in seen:
                seen.add(key)
                pats.append(p.reshape(GRID, GRID))
        return pats[:n]
    return gen


def add_noise(pattern, level, rng):
    noisy = pattern.copy()
    n_flip = max(1, int(pattern.size * level))
    idx = rng.choice(pattern.size, size=n_flip, replace=False)
    flat = noisy.ravel()
    flat[idx] = 1 - flat[idx]
    return noisy


# ─────────────────────────────────────────────
# Memory Systems
# ─────────────────────────────────────────────

class NanoLinkSlots:
    """The minimal Nano-Link: slot scoring via q·n, top-1 selection."""
    NAME = "NanoLink"
    def __init__(self):
        self.layers = []; self.cost = 0.0
    def reset_cost(self): self.cost = 0.0
    def store(self, p):
        nano = np.clip(p.astype(np.float64).ravel() * LR, 0, 1)
        self.layers.append((p.copy(), nano))
    def query(self, partial):
        q = partial.astype(np.float64).ravel()
        active = int(np.sum(q > 0))
        best_s, best_p = -1, self.layers[0][0]
        for pat, nano in self.layers:
            s = np.dot(q, nano)
            self.cost += active * C_VERT + C_LAYER
            if s > best_s:
                best_s, best_p = s, pat
        return best_p


class LinearHamming:
    """Full Hamming scan baseline."""
    NAME = "Hamming"
    def __init__(self):
        self.patterns = []; self.cost = 0.0
    def reset_cost(self): self.cost = 0.0
    def store(self, p): self.patterns.append(p.copy())
    def query(self, partial):
        best_d, best_p = float("inf"), self.patterns[0]
        for p in self.patterns:
            d = np.sum(p != partial)
            self.cost += p.size * C_CELL
            if d < best_d:
                best_d, best_p = d, p
        return best_p


class RawDotProduct:
    """Plain dot-product per-slot (no nano weights, no learning rate)."""
    NAME = "DotProd"
    def __init__(self):
        self.patterns = []; self.cost = 0.0
    def reset_cost(self): self.cost = 0.0
    def store(self, p): self.patterns.append(p.astype(np.float64).ravel())
    def query(self, partial):
        q = partial.astype(np.float64).ravel()
        active = int(np.sum(q > 0))
        best_s, best_i = -1, 0
        for i, p in enumerate(self.patterns):
            s = np.dot(q, p)
            self.cost += active * C_CELL
            if s > best_s:
                best_s, best_i = s, i
        return self.patterns[best_i].reshape(GRID, GRID).astype(np.int8)


class LSHMemory:
    """Locality-Sensitive Hashing with random hyperplanes."""
    NAME = "LSH"
    N_TABLES = 4
    N_HASHES = 6  # bits per hash
    def __init__(self):
        self.rng = np.random.default_rng(999)
        self.planes = [self.rng.standard_normal((self.N_HASHES, CELLS))
                       for _ in range(self.N_TABLES)]
        self.tables = [{} for _ in range(self.N_TABLES)]
        self.patterns = []
        self.cost = 0.0
    def reset_cost(self): self.cost = 0.0
    def _hash(self, vec, table_idx):
        proj = self.planes[table_idx] @ vec
        bits = tuple((proj > 0).astype(int))
        return bits
    def store(self, p):
        idx = len(self.patterns)
        self.patterns.append(p.copy())
        v = p.astype(np.float64).ravel()
        for t in range(self.N_TABLES):
            h = self._hash(v, t)
            self.tables[t].setdefault(h, []).append(idx)
    def query(self, partial):
        q = partial.astype(np.float64).ravel()
        # Hash cost
        self.cost += self.N_TABLES * self.N_HASHES * C_CELL
        candidates = set()
        for t in range(self.N_TABLES):
            h = self._hash(q, t)
            candidates.update(self.tables[t].get(h, []))
        if not candidates:
            candidates = set(range(len(self.patterns)))
        # Scan candidates with Hamming
        best_d, best_p = float("inf"), self.patterns[0]
        for i in candidates:
            p = self.patterns[i]
            d = np.sum(p != partial)
            self.cost += p.size * C_CELL
            if d < best_d:
                best_d, best_p = d, p
        return best_p


class InvertedIndex:
    """Inverted index: for each bit position, store pattern IDs where bit=1."""
    NAME = "InvIdx"
    def __init__(self):
        self.index = [[] for _ in range(CELLS)]  # one list per bit position
        self.patterns = []
        self.cost = 0.0
    def reset_cost(self): self.cost = 0.0
    def store(self, p):
        idx = len(self.patterns)
        self.patterns.append(p.copy())
        flat = p.ravel()
        for pos in range(CELLS):
            if flat[pos] == 1:
                self.index[pos].append(idx)
    def query(self, partial):
        q = partial.ravel()
        votes = np.zeros(len(self.patterns), dtype=np.int32)
        active_bits = np.where(q == 1)[0]
        for pos in active_bits:
            for pid in self.index[pos]:
                votes[pid] += 1
            self.cost += len(self.index[pos]) * C_VERT  # lookup cost
        self.cost += len(active_bits) * C_CELL  # scanning active bits
        best = np.argmax(votes)
        return self.patterns[best]


class BloomFilter:
    """Bloom-like candidate filter: hash into buckets, score candidates."""
    NAME = "Bloom"
    N_HASH = 3
    BUCKET_SIZE = 64
    def __init__(self):
        self.buckets = [{} for _ in range(self.N_HASH)]
        self.patterns = []
        self.cost = 0.0
    def reset_cost(self): self.cost = 0.0
    def _hashes(self, p):
        flat = p.ravel().tobytes()
        hs = []
        for i in range(self.N_HASH):
            h = int(hashlib.md5(flat + bytes([i])).hexdigest(), 16) % self.BUCKET_SIZE
            hs.append(h)
        return hs
    def store(self, p):
        idx = len(self.patterns)
        self.patterns.append(p.copy())
        for i, h in enumerate(self._hashes(p)):
            self.buckets[i].setdefault(h, []).append(idx)
    def query(self, partial):
        hs = self._hashes(partial)
        self.cost += self.N_HASH * C_CELL  # hashing cost
        candidates = set()
        for i, h in enumerate(hs):
            candidates.update(self.buckets[i].get(h, []))
        if not candidates:
            candidates = set(range(len(self.patterns)))
        best_d, best_p = float("inf"), self.patterns[0]
        for ci in candidates:
            p = self.patterns[ci]
            d = np.sum(p != partial)
            self.cost += p.size * C_CELL
            if d < best_d:
                best_d, best_p = d, p
        return best_p


ALL_SYSTEMS = [NanoLinkSlots, LinearHamming, RawDotProduct, LSHMemory, InvertedIndex, BloomFilter]


# ─────────────────────────────────────────────
# Metrics + Runner
# ─────────────────────────────────────────────
def bit_acc(a, b): return float(np.mean(a == b))
def exact_acc(a, b): return 1.0 if np.array_equal(a, b) else 0.0
def utility(ba, cost): return ba / np.log1p(cost)

def run_one(MemClass, patterns, noise, rng):
    mem = MemClass()
    for p in patterns: mem.store(p)
    mem.reset_cost()
    ea, ba = [], []
    for p in patterns:
        noisy = add_noise(p, noise, rng)
        r = mem.query(noisy)
        ea.append(exact_acc(p, r))
        ba.append(bit_acc(p, r))
    bv = np.mean(ba)
    return {"exact": np.mean(ea), "bit": bv, "cost": mem.cost, "util": utility(bv, mem.cost)}

def run_multi(MemClass, gen_fn, n, noise, trials=TRIALS):
    res = []
    for t in range(trials):
        rng = np.random.default_rng(42 + t*1000 + n + int(noise*100))
        pats = gen_fn(n, rng)
        res.append(run_one(MemClass, pats, noise, rng))
    return {k: np.mean([r[k] for r in res]) for k in res[0]}


# ═════════════════════════════════════════════
# TEST 1: Semi-Realistic Datasets
# ═════════════════════════════════════════════
def test_realistic():
    print("\n" + "="*80)
    print("  TEST 1: Semi-Realistic Datasets — 1000 patterns, 30% noise")
    print("="*80)
    datasets = [
        ("BinEmbeddings", gen_binarized_embeddings),
        ("Clustered",     gen_clustered),
        ("Fingerprints",  gen_fingerprint),
    ]
    n, noise = 1000, 0.30
    results = []
    for ds_name, gen_fn in datasets:
        row = {"dataset": ds_name}
        for Sys in ALL_SYSTEMS:
            print(f"  {ds_name:>14} × {Sys.NAME:>8} ...", end="", flush=True)
            r = run_multi(Sys, gen_fn, n, noise)
            row[f"{Sys.NAME}_bit"] = r["bit"]
            row[f"{Sys.NAME}_cost"] = r["cost"]
            row[f"{Sys.NAME}_util"] = r["util"]
            row[f"{Sys.NAME}_exact"] = r["exact"]
            print(f"  bit={r['bit']:.4f}  cost={r['cost']:>12.0f}  util={r['util']:.6f}")
        results.append(row)
    return results


# ═════════════════════════════════════════════
# TEST 2: Sparsity Sweep
# ═════════════════════════════════════════════
def test_sparsity():
    print("\n" + "="*80)
    print("  TEST 2: Sparsity Sweep — 500 patterns, 30% noise")
    print("="*80)
    densities = [0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.90]
    n, noise = 500, 0.30
    results = []
    for dens in densities:
        row = {"density": dens}
        gen_fn = gen_with_density(dens)
        for Sys in [NanoLinkSlots, LinearHamming, RawDotProduct]:
            print(f"  dens={dens:.2f} × {Sys.NAME:>8} ...", end="", flush=True)
            r = run_multi(Sys, gen_fn, n, noise)
            row[f"{Sys.NAME}_bit"] = r["bit"]
            row[f"{Sys.NAME}_cost"] = r["cost"]
            row[f"{Sys.NAME}_util"] = r["util"]
            print(f"  bit={r['bit']:.4f}  util={r['util']:.6f}")
        results.append(row)
    return results


# ═════════════════════════════════════════════
# TEST 3: NanoLink vs All Baselines
# ═════════════════════════════════════════════
def test_all_baselines():
    print("\n" + "="*80)
    print("  TEST 3: NanoLink vs All Baselines — BinEmbeddings, 30% noise")
    print("="*80)
    sizes = [100, 500, 1000, 2000]
    noise = 0.30
    results = []
    for n in sizes:
        row = {"n": n}
        for Sys in ALL_SYSTEMS:
            print(f"  n={n:<5} × {Sys.NAME:>8} ...", end="", flush=True)
            r = run_multi(Sys, gen_binarized_embeddings, n, noise, trials=2)
            row[f"{Sys.NAME}_bit"] = r["bit"]
            row[f"{Sys.NAME}_cost"] = r["cost"]
            row[f"{Sys.NAME}_util"] = r["util"]
            print(f"  bit={r['bit']:.4f}  cost={r['cost']:>12.0f}  util={r['util']:.6f}")
        results.append(row)
    return results


# ═════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════
SYS_COLORS = {
    "NanoLink": "#16c79a", "Hamming": "#e94560", "DotProd": "#f5a623",
    "LSH": "#533483", "InvIdx": "#0f3460", "Bloom": "#888888",
}

def plot_all(realistic, sparsity, baselines):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#aaa", "ytick.color": "#aaa",
        "grid.color": "#333", "grid.alpha": 0.5, "font.size": 11,
        "legend.facecolor": "#16213e", "legend.edgecolor": "#444",
    })

    # ── 1. Realistic datasets ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ds_names = [r["dataset"] for r in realistic]
    sys_names = [S.NAME for S in ALL_SYSTEMS]
    x = np.arange(len(ds_names))
    w = 0.13
    for i, sn in enumerate(sys_names):
        bits = [r[f"{sn}_bit"] for r in realistic]
        ax1.bar(x + i*w, bits, w, label=sn, color=SYS_COLORS[sn], edgecolor="#eee")
        costs = [r[f"{sn}_cost"] for r in realistic]
        ax2.bar(x + i*w, costs, w, color=SYS_COLORS[sn], edgecolor="#eee")
        utils = [r[f"{sn}_util"] for r in realistic]
        ax3.bar(x + i*w, utils, w, color=SYS_COLORS[sn], edgecolor="#eee")
    for ax, title in [(ax1, "Bit Accuracy"), (ax2, "Cost (log)"), (ax3, "Utility")]:
        ax.set_xticks(x + w*2.5); ax.set_xticklabels(ds_names)
        ax.set_title(title, fontweight="bold"); ax.grid(axis="y")
    ax1.legend(fontsize=8, loc="lower left"); ax1.set_ylim(0.4, 1.05)
    ax2.set_yscale("log")
    fig.suptitle("Semi-Realistic Datasets (1000 pat, 30% noise)",
                 fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "realistic_datasets.png"), dpi=150)
    plt.close(fig)

    # ── 2. Sparsity sweep ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    densities = [r["density"] for r in sparsity]
    for sn in ["NanoLink", "Hamming", "DotProd"]:
        bits = [r[f"{sn}_bit"] for r in sparsity]
        utils = [r[f"{sn}_util"] for r in sparsity]
        ax1.plot(densities, bits, "o-", color=SYS_COLORS[sn], label=sn, lw=2, ms=7)
        ax2.plot(densities, utils, "o-", color=SYS_COLORS[sn], label=sn, lw=2, ms=7)
    ax1.set_title("Bit Accuracy vs Density", fontweight="bold")
    ax2.set_title("Utility vs Density", fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xlabel("Pattern Density"); ax.legend(); ax.grid(True)
    ax1.set_ylim(0.4, 1.05)
    fig.suptitle("Sparsity Sweep (500 pat, 30% noise)",
                 fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "sparsity_sweep.png"), dpi=150)
    plt.close(fig)

    # ── 3. All baselines scaling ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ns = [r["n"] for r in baselines]
    for sn in sys_names:
        bits = [r[f"{sn}_bit"] for r in baselines]
        costs = [r[f"{sn}_cost"] for r in baselines]
        utils = [r[f"{sn}_util"] for r in baselines]
        ax1.plot(ns, bits, "o-", color=SYS_COLORS[sn], label=sn, lw=2, ms=7)
        ax2.plot(ns, costs, "o-", color=SYS_COLORS[sn], label=sn, lw=2, ms=7)
        ax3.plot(ns, utils, "o-", color=SYS_COLORS[sn], label=sn, lw=2, ms=7)
    ax1.set_title("Bit Accuracy", fontweight="bold"); ax1.set_ylim(0.4, 1.05)
    ax2.set_title("Cost (log-log)", fontweight="bold"); ax2.set_xscale("log"); ax2.set_yscale("log")
    ax3.set_title("Utility", fontweight="bold")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Patterns"); ax.legend(fontsize=8); ax.grid(True)
    fig.suptitle("All Baselines Scaling (BinEmbeddings, 30% noise)",
                 fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "all_baselines.png"), dpi=150)
    plt.close(fig)

    # ── 4. Head-to-head: NanoLink vs DotProd ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for sn in ["NanoLink", "DotProd"]:
        bits = [r[f"{sn}_bit"] for r in baselines]
        costs = [r[f"{sn}_cost"] for r in baselines]
        ax1.plot(ns, bits, "o-", color=SYS_COLORS[sn], label=sn, lw=2, ms=7)
        ax2.plot(ns, costs, "o-", color=SYS_COLORS[sn], label=sn, lw=2, ms=7)
    ax1.set_title("Accuracy Head-to-Head", fontweight="bold"); ax1.set_ylim(0.4, 1.05)
    ax2.set_title("Cost Head-to-Head", fontweight="bold")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xlabel("Patterns"); ax.legend(); ax.grid(True)
    fig.suptitle("NanoLink vs Raw Dot-Product (THE test)",
                 fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "nanolink_vs_dotprod.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Charts saved to: {RESULTS_DIR}")


# ═════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════
def print_summary(realistic, sparsity, baselines):
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"{'PHASE 4 — FINAL VALIDATION RESULTS':^100}")
    print(sep)

    # Realistic datasets
    print("\n  TEST 1: SEMI-REALISTIC DATASETS (1000 pat, 30% noise)")
    print("  " + "-"*90)
    print(f"  {'Dataset':>14}  ", end="")
    for S in ALL_SYSTEMS:
        print(f"  {S.NAME:>8}", end="")
    print()
    for row in realistic:
        print(f"  {row['dataset']:>14}  ", end="")
        for S in ALL_SYSTEMS:
            print(f"  {row[f'{S.NAME}_bit']:>8.4f}", end="")
        print()
    print(f"\n  {'Utility':>14}  ", end="")
    for S in ALL_SYSTEMS:
        print(f"  {S.NAME:>8}", end="")
    print()
    for row in realistic:
        print(f"  {row['dataset']:>14}  ", end="")
        for S in ALL_SYSTEMS:
            print(f"  {row[f'{S.NAME}_util']:>8.5f}", end="")
        print()

    # Sparsity
    print(f"\n  TEST 2: SPARSITY SWEEP (500 pat, 30% noise)")
    print("  " + "-"*90)
    for row in sparsity:
        nl = row["NanoLink_bit"]
        dp = row["DotProd_bit"]
        hm = row["Hamming_bit"]
        diff = nl - dp
        print(f"  dens={row['density']:.2f}  NL={nl:.4f}  DotProd={dp:.4f}  Hamming={hm:.4f}  NL-DP={diff:+.4f}")

    # Baselines scaling
    print(f"\n  TEST 3: ALL BASELINES SCALING (BinEmbeddings, 30% noise)")
    print("  " + "-"*90)
    for row in baselines:
        nl_u = row["NanoLink_util"]
        best_name = max(ALL_SYSTEMS, key=lambda S: row[f"{S.NAME}_util"]).NAME
        best_u = row[f"{best_name}_util"]
        print(f"  n={row['n']:<5}  NanoLink util={nl_u:.5f}  Best={best_name} ({best_u:.5f})")

    # Critical comparison: NanoLink vs DotProd
    print(f"\n  CRITICAL: NanoLink vs Raw Dot-Product")
    print("  " + "-"*90)
    for row in baselines:
        nl_b = row["NanoLink_bit"]; dp_b = row["DotProd_bit"]
        nl_c = row["NanoLink_cost"]; dp_c = row["DotProd_cost"]
        print(f"  n={row['n']:<5}  NL bit={nl_b:.4f} cost={nl_c:.0f}  |  DP bit={dp_b:.4f} cost={dp_c:.0f}  |  bit_diff={nl_b-dp_b:+.4f}  cost_ratio={dp_c/max(nl_c,1):.2f}x")

    print(f"\n{sep}\n")


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  NANO-LINK PHASE 4 — FINAL VALIDATION")
    print("  Is Nano-Link novel, or a rediscovered dot-product?")
    print("=" * 80)

    t0 = time.time()
    realistic  = test_realistic()
    sparsity   = test_sparsity()
    baselines  = test_all_baselines()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_summary(realistic, sparsity, baselines)
    plot_all(realistic, sparsity, baselines)
    print("  Done.")

#!/usr/bin/env python3
"""
Nano-Link Phase 2 — Three-Version Comparison
=============================================
Version A: Superposition (original — all patterns in single α/n grid)
Version B: Pattern Slots (each pattern stored in its own layer)
Version C: Slots + Winner-Take-All (competitive inhibition between layers)

Plus: Linear baseline (Hamming distance) for reference.

Measures: exact accuracy, bitwise accuracy, cost, utility metric.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as cartesian

# ─────────────────────────────────────────────
# 0. Constants
# ─────────────────────────────────────────────
GRID_SIZE = 8
LEARNING_RATE = 0.15
PROPAGATION_ITERS = 3
THRESHOLD_A = 0.35
NUM_TRIALS = 5

COST_CELL_ACCESS = 1.0
COST_VERTICAL = 0.3
COST_PROPAGATION = 0.2
COST_LAYER_SCORE = 0.5          # scoring a layer in Version B/C
COST_INHIBITION = 0.1           # lateral inhibition step in Version C

PATTERN_COUNTS = [20, 100, 500, 1000]
NOISE_LEVELS = [0.10, 0.20, 0.30]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase2")


# ─────────────────────────────────────────────
# 1. Pattern generation (same as Phase 1)
# ─────────────────────────────────────────────
TEMPLATES = {
    "A": np.array([[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],
                    [1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1]], dtype=np.int8),
    "B": np.array([[1,1,1,1,1,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,1,0],
                    [1,1,1,1,1,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,1,0],[1,1,1,1,1,1,0,0]], dtype=np.int8),
    "C": np.array([[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,1,0],[0,0,1,1,1,1,0,0]], dtype=np.int8),
    "D": np.array([[1,1,1,1,1,0,0,0],[1,0,0,0,0,1,0,0],[1,0,0,0,0,0,1,0],
                    [1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,1,0],
                    [1,0,0,0,0,1,0,0],[1,1,1,1,1,0,0,0]], dtype=np.int8),
    "E": np.array([[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1]], dtype=np.int8),
}


def generate_patterns(n, rng):
    templates = list(TEMPLATES.values())
    patterns = [t.copy() for t in templates]
    while len(patterns) < n:
        base = templates[rng.integers(len(templates))]
        variant = base.copy()
        num_flips = rng.integers(2, 6)
        coords = rng.integers(0, GRID_SIZE, size=(num_flips, 2))
        for r, c in coords:
            variant[r, c] = 1 - variant[r, c]
        if not any(np.array_equal(variant, p) for p in patterns):
            patterns.append(variant)
    return patterns[:n]


def add_noise(pattern, level, rng):
    noisy = pattern.copy()
    n_flip = max(1, int(pattern.size * level))
    indices = rng.choice(pattern.size, size=n_flip, replace=False)
    flat = noisy.ravel()
    flat[indices] = 1 - flat[indices]
    return noisy


# ─────────────────────────────────────────────
# 2. VERSION A — Superposition (original)
# ─────────────────────────────────────────────
class NanoLinkA:
    """Single-field superposition. All patterns overlap in one α/n grid."""
    NAME = "A-Superposition"

    def __init__(self):
        self.alpha = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.nano = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.cost = 0.0

    def reset_cost(self):
        self.cost = 0.0

    def store(self, pattern):
        mask = pattern.astype(np.float64)
        self.alpha += mask
        self.nano += mask * LEARNING_RATE
        self.nano = np.clip(self.nano, 0.0, 1.0)

    def query(self, partial):
        q = partial.astype(np.float64)
        active = np.sum(q > 0)

        # β activation
        beta = q * self.nano
        self.cost += active * COST_CELL_ACCESS
        self.cost += active * COST_VERTICAL

        # propagation
        for _ in range(PROPAGATION_ITERS):
            new_beta = beta.copy()
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    nb = []
                    if i > 0:            nb.append(beta[i-1, j])
                    if i < GRID_SIZE-1:  nb.append(beta[i+1, j])
                    if j > 0:            nb.append(beta[i, j-1])
                    if j < GRID_SIZE-1:  nb.append(beta[i, j+1])
                    if nb:
                        new_beta[i, j] += np.mean(nb)
            beta = new_beta
            self.cost += GRID_SIZE * GRID_SIZE * COST_PROPAGATION

        # threshold
        nz = beta[beta > 0]
        thresh = (np.mean(nz) * THRESHOLD_A + np.std(nz) * 0.1) if len(nz) > 0 else THRESHOLD_A
        return (beta > thresh).astype(np.int8)


# ─────────────────────────────────────────────
# 3. VERSION B — Pattern Slots
# ─────────────────────────────────────────────
class NanoLinkB:
    """Each pattern lives in its own layer. Query scores all layers, picks best."""
    NAME = "B-Slots"

    def __init__(self):
        self.layers = []          # list of (pattern_array, nano_weights)
        self.cost = 0.0

    def reset_cost(self):
        self.cost = 0.0

    def store(self, pattern):
        mask = pattern.astype(np.float64)
        nano = mask * LEARNING_RATE
        nano = np.clip(nano, 0.0, 1.0)
        self.layers.append((pattern.copy(), nano))

    def query(self, partial):
        q = partial.astype(np.float64)
        active = int(np.sum(q > 0))

        best_score = -1.0
        best_pattern = self.layers[0][0]

        for pat, nano in self.layers:
            # Score = dot product of query activation with nano weights
            score = np.sum(q * nano)
            self.cost += active * COST_CELL_ACCESS
            self.cost += active * COST_VERTICAL
            self.cost += COST_LAYER_SCORE

            if score > best_score:
                best_score = score
                best_pattern = pat

        return best_pattern.copy()


# ─────────────────────────────────────────────
# 4. VERSION C — Slots + Winner-Take-All
# ─────────────────────────────────────────────
class NanoLinkC:
    """
    Pattern slots with competitive inhibition.
    Two-pass: coarse scoring → top-K → fine scoring with propagation.
    """
    NAME = "C-Slots+WTA"
    TOP_K = 5   # number of candidates surviving inhibition

    def __init__(self):
        self.layers = []
        self.cost = 0.0

    def reset_cost(self):
        self.cost = 0.0

    def store(self, pattern):
        mask = pattern.astype(np.float64)
        nano = mask * LEARNING_RATE
        nano = np.clip(nano, 0.0, 1.0)
        self.layers.append((pattern.copy(), nano))

    def query(self, partial):
        q = partial.astype(np.float64)
        active = int(np.sum(q > 0))
        N = len(self.layers)

        # ── Pass 1: coarse scoring (cheap) ──
        scores = np.zeros(N)
        for idx, (pat, nano) in enumerate(self.layers):
            scores[idx] = np.sum(q * nano)
            self.cost += active * COST_VERTICAL   # only vertical activation cost
            self.cost += COST_LAYER_SCORE

        # ── Lateral inhibition: keep top-K ──
        k = min(self.TOP_K, N)
        top_indices = np.argpartition(scores, -k)[-k:]
        self.cost += N * COST_INHIBITION   # inhibition sweep

        # ── Pass 2: fine scoring on survivors with propagation ──
        best_score = -1.0
        best_pattern = self.layers[top_indices[0]][0]

        for idx in top_indices:
            pat, nano = self.layers[idx]
            beta = q * nano
            self.cost += active * COST_CELL_ACCESS
            self.cost += active * COST_VERTICAL

            # local propagation (only on survivors)
            for _ in range(PROPAGATION_ITERS):
                new_beta = beta.copy()
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        nb = []
                        if i > 0:            nb.append(beta[i-1, j])
                        if i < GRID_SIZE-1:  nb.append(beta[i+1, j])
                        if j > 0:            nb.append(beta[i, j-1])
                        if j < GRID_SIZE-1:  nb.append(beta[i, j+1])
                        if nb:
                            new_beta[i, j] += np.mean(nb)
                beta = new_beta
                self.cost += GRID_SIZE * GRID_SIZE * COST_PROPAGATION

            fine_score = np.sum(beta * q)
            if fine_score > best_score:
                best_score = fine_score
                best_pattern = pat

        return best_pattern.copy()


# ─────────────────────────────────────────────
# 5. LINEAR BASELINE
# ─────────────────────────────────────────────
class LinearMemory:
    NAME = "Linear-Hamming"

    def __init__(self):
        self.patterns = []
        self.cost = 0.0

    def reset_cost(self):
        self.cost = 0.0

    def store(self, pattern):
        self.patterns.append(pattern.copy())

    def query(self, partial):
        best_dist = float("inf")
        best_pat = self.patterns[0]
        for pat in self.patterns:
            dist = np.sum(pat != partial)
            self.cost += pat.size * COST_CELL_ACCESS
            if dist < best_dist:
                best_dist = dist
                best_pat = pat
        return best_pat


# ─────────────────────────────────────────────
# 6. Metrics
# ─────────────────────────────────────────────
def accuracy_exact(orig, recov):
    return 1.0 if np.array_equal(orig, recov) else 0.0

def accuracy_bitwise(orig, recov):
    return float(np.mean(orig == recov))

def utility(bit_acc, cost):
    """Composite metric: accuracy weighted by inverse log-cost."""
    return bit_acc / np.log1p(cost)


# ─────────────────────────────────────────────
# 7. Test Harness
# ─────────────────────────────────────────────
SYSTEMS = [NanoLinkA, NanoLinkB, NanoLinkC, LinearMemory]


def run_experiment(SystemClass, n_patterns, noise_level, seed=42):
    rng = np.random.default_rng(seed)
    patterns = generate_patterns(n_patterns, rng)

    mem = SystemClass()
    for p in patterns:
        mem.store(p)

    mem.reset_cost()
    exact_accs, bit_accs = [], []

    for p in patterns:
        noisy = add_noise(p, noise_level, rng)
        recovered = mem.query(noisy)
        exact_accs.append(accuracy_exact(p, recovered))
        bit_accs.append(accuracy_bitwise(p, recovered))

    ea = np.mean(exact_accs)
    ba = np.mean(bit_accs)
    return {
        "system": SystemClass.NAME,
        "n_patterns": n_patterns,
        "noise": noise_level,
        "exact_acc": ea,
        "bit_acc": ba,
        "cost": mem.cost,
        "utility": utility(ba, mem.cost),
    }


def run_all():
    combos = list(cartesian(SYSTEMS, PATTERN_COUNTS, NOISE_LEVELS))
    total = len(combos)
    all_results = []

    for idx, (Sys, n_pat, noise) in enumerate(combos, 1):
        tag = f"[{idx}/{total}] {Sys.NAME:18s} P={n_pat:<5d} N={int(noise*100)}%"
        print(f"  {tag}", end="", flush=True)

        trial_results = []
        for trial in range(NUM_TRIALS):
            seed = hash((Sys.NAME, n_pat, noise, trial)) % (2**31)
            trial_results.append(run_experiment(Sys, n_pat, noise, seed))

        avg = {
            "system": Sys.NAME,
            "n_patterns": n_pat,
            "noise": noise,
        }
        for key in ["exact_acc", "bit_acc", "cost", "utility"]:
            avg[key] = np.mean([t[key] for t in trial_results])
        all_results.append(avg)
        print(f"  bit={avg['bit_acc']:.3f}  cost={avg['cost']:.0f}  util={avg['utility']:.6f}")

    return all_results


# ─────────────────────────────────────────────
# 8. Visualization
# ─────────────────────────────────────────────
COLORS = {
    "A-Superposition":  "#e94560",
    "B-Slots":          "#0f3460",
    "C-Slots+WTA":      "#16c79a",
    "Linear-Hamming":   "#533483",
}

def plot_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#444",
        "axes.labelcolor":  "#eee",
        "text.color":       "#eee",
        "xtick.color":      "#aaa",
        "ytick.color":      "#aaa",
        "grid.color":       "#333",
        "grid.alpha":       0.5,
        "font.size":        11,
        "legend.facecolor": "#16213e",
        "legend.edgecolor": "#444",
    })

    sys_names = [S.NAME for S in SYSTEMS]

    # ═══════════════════════════════════════════
    # Chart 1: Bitwise Accuracy vs Noise (one subplot per pattern count)
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, len(PATTERN_COUNTS),
                             figsize=(5*len(PATTERN_COUNTS), 5), sharey=True)
    for ax, n_pat in zip(axes, PATTERN_COUNTS):
        for sn in sys_names:
            sub = [r for r in results if r["system"] == sn and r["n_patterns"] == n_pat]
            noises = [r["noise"] for r in sub]
            accs   = [r["bit_acc"] for r in sub]
            ax.plot([int(n*100) for n in noises], accs, "o-",
                    color=COLORS[sn], label=sn, linewidth=2, markersize=7)
        ax.set_xlabel("Noise %")
        ax.set_title(f"{n_pat} Patterns", fontweight="bold")
        ax.set_ylim(0.3, 1.05)
        ax.grid(True)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Bitwise Accuracy vs Noise", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # ═══════════════════════════════════════════
    # Chart 2: Cost vs Pattern Count (one line per system, fix noise=30%)
    # ═══════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 5))
    for sn in sys_names:
        sub = sorted([r for r in results if r["system"] == sn and r["noise"] == 0.30],
                     key=lambda r: r["n_patterns"])
        pats  = [r["n_patterns"] for r in sub]
        costs = [r["cost"] for r in sub]
        ax.plot(pats, costs, "o-", color=COLORS[sn], label=sn, linewidth=2, markersize=7)
    ax.set_xlabel("Number of Patterns")
    ax.set_ylabel("Simulated Cost")
    ax.set_title("Cost Scalability (30% noise)", fontweight="bold", color="#e94560")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cost_scalability.png"), dpi=150)
    plt.close(fig)

    # ═══════════════════════════════════════════
    # Chart 3: Utility metric (bit_acc / log(1+cost))
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, len(PATTERN_COUNTS),
                             figsize=(5*len(PATTERN_COUNTS), 5), sharey=True)
    for ax, n_pat in zip(axes, PATTERN_COUNTS):
        for sn in sys_names:
            sub = [r for r in results if r["system"] == sn and r["n_patterns"] == n_pat]
            noises = [int(r["noise"]*100) for r in sub]
            utils  = [r["utility"] for r in sub]
            ax.plot(noises, utils, "o-", color=COLORS[sn], label=sn, linewidth=2, markersize=7)
        ax.set_xlabel("Noise %")
        ax.set_title(f"{n_pat} Patterns", fontweight="bold")
        ax.grid(True)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Utility = bit_acc / log(1 + cost)", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "utility_comparison.png"), dpi=150)
    plt.close(fig)

    # ═══════════════════════════════════════════
    # Chart 4: Critical test bar chart (1000 patterns, 30% noise)
    # ═══════════════════════════════════════════
    critical = [r for r in results if r["n_patterns"] == 1000 and r["noise"] == 0.30]
    if critical:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

        names = [r["system"] for r in critical]
        colors = [COLORS[n] for n in names]

        # Accuracy
        accs = [r["bit_acc"] for r in critical]
        ax1.bar(names, accs, color=colors, edgecolor="#eee")
        ax1.set_title("Bitwise Accuracy", fontweight="bold")
        ax1.set_ylim(0, 1.1)
        for i, v in enumerate(accs):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

        # Cost (log scale)
        costs = [r["cost"] for r in critical]
        ax2.bar(names, costs, color=colors, edgecolor="#eee")
        ax2.set_title("Total Cost", fontweight="bold")
        ax2.set_yscale("log")
        for i, v in enumerate(costs):
            ax2.text(i, v * 1.3, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")

        # Utility
        utils = [r["utility"] for r in critical]
        ax3.bar(names, utils, color=colors, edgecolor="#eee")
        ax3.set_title("Utility Score", fontweight="bold")
        for i, v in enumerate(utils):
            ax3.text(i, v + max(utils)*0.02, f"{v:.5f}", ha="center", fontsize=9, fontweight="bold")

        fig.suptitle("CRITICAL TEST: 1000 patterns, 30% noise",
                     fontsize=14, fontweight="bold", color="#e94560")
        for ax in (ax1, ax2, ax3):
            ax.grid(axis="y")
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(RESULTS_DIR, "critical_test.png"), dpi=150)
        plt.close(fig)

    print(f"\n  Charts saved to: {RESULTS_DIR}")


# ─────────────────────────────────────────────
# 9. Summary
# ─────────────────────────────────────────────
def print_summary(results):
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"{'NANO-LINK PHASE 2 — THREE-VERSION COMPARISON':^130}")
    print(sep)
    print("{:>18s} {:>8s} {:>6s} | {:>10s} {:>10s} {:>12s} {:>12s}".format(
        "System", "Patterns", "Noise", "Exact Acc", "Bit Acc", "Cost", "Utility"))
    print("-" * 130)

    for r in results:
        print("{:>18s} {:>8d} {:>5d}% | {:>10.4f} {:>10.4f} {:>12.0f} {:>12.7f}".format(
            r["system"], r["n_patterns"], int(r["noise"]*100),
            r["exact_acc"], r["bit_acc"], r["cost"], r["utility"]))

    print(sep)

    # Critical test
    print("\n  CRITICAL TEST: 1000 patterns, 30% noise")
    print("  " + "-"*80)
    critical = [r for r in results if r["n_patterns"] == 1000 and r["noise"] == 0.30]
    for r in critical:
        print("  {:18s} | bit_acc={:.4f}  cost={:>12.0f}  utility={:.7f}".format(
            r["system"], r["bit_acc"], r["cost"], r["utility"]))

    # Cost growth analysis
    print("\n  COST GROWTH (20 -> 1000 patterns, 30% noise)")
    print("  " + "-"*80)
    for sn in [S.NAME for S in SYSTEMS]:
        s20 = [r for r in results if r["system"] == sn and r["n_patterns"] == 20  and r["noise"] == 0.30]
        s1k = [r for r in results if r["system"] == sn and r["n_patterns"] == 1000 and r["noise"] == 0.30]
        if s20 and s1k:
            growth = s1k[0]["cost"] / max(s20[0]["cost"], 1)
            print("  {:18s} | {:.1f}x growth".format(sn, growth))

    # Best utility per scenario
    print("\n  BEST SYSTEM BY UTILITY")
    print("  " + "-"*80)
    for n_pat in PATTERN_COUNTS:
        for noise in NOISE_LEVELS:
            sub = [r for r in results if r["n_patterns"] == n_pat and r["noise"] == noise]
            best = max(sub, key=lambda r: r["utility"])
            print("  P={:<5d} N={:>2d}% -> {:18s} (utility={:.7f})".format(
                n_pat, int(noise*100), best["system"], best["utility"]))

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NANO-LINK PHASE 2")
    print("  Three-Version Comparison")
    print("  A=Superposition  B=Slots  C=Slots+WTA  Lin=Hamming")
    print("=" * 60)
    print()

    t0 = time.time()
    print("  Running experiments...")
    results = run_all()
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    print_summary(results)
    plot_results(results)
    print("  Done.")

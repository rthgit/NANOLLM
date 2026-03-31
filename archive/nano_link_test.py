#!/usr/bin/env python3
"""
Nano-Link Test Protocol
=======================
Benchmark: associative retrieval via weighted vertical links (Nano-Link)
vs. linear scan with Hamming distance.

Metrics: accuracy, simulated access cost, scalability.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as cartesian

# ─────────────────────────────────────────────
# 0. Constants
# ─────────────────────────────────────────────
GRID_SIZE = 8
LEARNING_RATE = 0.15
PROPAGATION_ITERS = 3
THRESHOLD = 0.35            # β threshold for reconstruction
NUM_TRIALS = 5              # repeat each experiment with different seeds

COST_CELL_ACCESS = 1.0
COST_VERTICAL = 0.3
COST_PROPAGATION = 0.2

PATTERN_COUNTS = [20, 100, 500, 1000]
NOISE_LEVELS = [0.10, 0.20, 0.30]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ─────────────────────────────────────────────
# 1. Pattern generation
# ─────────────────────────────────────────────

# 5 hand-coded 8×8 letter templates (A-E)
TEMPLATES = {
    "A": np.array([
        [0,0,1,1,1,1,0,0],
        [0,1,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
    ], dtype=np.int8),
    "B": np.array([
        [1,1,1,1,1,1,0,0],
        [1,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0],
        [1,1,1,1,1,1,0,0],
        [1,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,1,0],
        [1,1,1,1,1,1,0,0],
    ], dtype=np.int8),
    "C": np.array([
        [0,0,1,1,1,1,0,0],
        [0,1,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,1,0],
        [0,0,1,1,1,1,0,0],
    ], dtype=np.int8),
    "D": np.array([
        [1,1,1,1,1,0,0,0],
        [1,0,0,0,0,1,0,0],
        [1,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,1,0],
        [1,0,0,0,0,1,0,0],
        [1,1,1,1,1,0,0,0],
    ], dtype=np.int8),
    "E": np.array([
        [1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0],
        [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1],
    ], dtype=np.int8),
}


def generate_patterns(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate `n` unique 8×8 binary patterns from the base templates."""
    templates = list(TEMPLATES.values())
    patterns: list[np.ndarray] = [t.copy() for t in templates]

    while len(patterns) < n:
        base = templates[rng.integers(len(templates))]
        variant = base.copy()
        # Flip 2-5 random bits to create a variant
        num_flips = rng.integers(2, 6)
        coords = rng.integers(0, GRID_SIZE, size=(num_flips, 2))
        for r, c in coords:
            variant[r, c] = 1 - variant[r, c]
        # Check uniqueness
        is_dup = any(np.array_equal(variant, p) for p in patterns)
        if not is_dup:
            patterns.append(variant)

    return patterns[:n]


def add_noise(pattern: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Flip `level` fraction of bits randomly."""
    noisy = pattern.copy()
    total = pattern.size
    n_flip = max(1, int(total * level))
    indices = rng.choice(total, size=n_flip, replace=False)
    flat = noisy.ravel()
    flat[indices] = 1 - flat[indices]
    return noisy


# ─────────────────────────────────────────────
# 2. Nano-Link Memory
# ─────────────────────────────────────────────
class NanoLinkMemory:
    """Memory with α (accumulator), n (vertical link weights), β (activation)."""

    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.alpha = np.zeros((size, size), dtype=np.float64)
        self.nano = np.zeros((size, size), dtype=np.float64)
        self.cost = 0.0
        self.num_patterns = 0

    def reset_cost(self):
        self.cost = 0.0

    def store(self, pattern: np.ndarray):
        """Store a pattern (write phase)."""
        self.num_patterns += 1
        mask = pattern.astype(np.float64)
        self.alpha += mask
        self.nano += mask * LEARNING_RATE
        self.nano = np.clip(self.nano, 0.0, 1.0)

    def query(self, partial: np.ndarray) -> np.ndarray:
        """Retrieve the best-match pattern for a (possibly noisy) input."""
        size = self.size
        q = partial.astype(np.float64)

        # ── β activation via vertical links ──
        beta = np.zeros((size, size), dtype=np.float64)
        active_cells = np.sum(q > 0)
        beta = q * self.nano
        self.cost += active_cells * COST_CELL_ACCESS        # accessing cells
        self.cost += active_cells * COST_VERTICAL            # vertical activation

        # ── Local propagation ──
        for _ in range(PROPAGATION_ITERS):
            new_beta = beta.copy()
            for i in range(size):
                for j in range(size):
                    neighbours = []
                    if i > 0:        neighbours.append(beta[i-1, j])
                    if i < size-1:   neighbours.append(beta[i+1, j])
                    if j > 0:        neighbours.append(beta[i, j-1])
                    if j < size-1:   neighbours.append(beta[i, j+1])
                    if neighbours:
                        new_beta[i, j] += np.mean(neighbours)
            beta = new_beta
            self.cost += size * size * COST_PROPAGATION      # propagation cost

        # ── Threshold ──
        # Adaptive threshold: use mean + 0.5*std of non-zero betas
        nonzero = beta[beta > 0]
        if len(nonzero) > 0:
            thresh = np.mean(nonzero) * THRESHOLD + np.std(nonzero) * 0.1
        else:
            thresh = THRESHOLD
        result = (beta > thresh).astype(np.int8)
        return result


# ─────────────────────────────────────────────
# 3. Linear Memory (Baseline)
# ─────────────────────────────────────────────
class LinearMemory:
    """Simple memory: stores all patterns, retrieves by Hamming distance scan."""

    def __init__(self):
        self.patterns: list[np.ndarray] = []
        self.cost = 0.0

    def reset_cost(self):
        self.cost = 0.0

    def store(self, pattern: np.ndarray):
        self.patterns.append(pattern.copy())

    def query(self, partial: np.ndarray) -> np.ndarray:
        """Return pattern with smallest Hamming distance."""
        best_dist = float("inf")
        best_pat = self.patterns[0]
        for pat in self.patterns:
            dist = np.sum(pat != partial)
            self.cost += pat.size * COST_CELL_ACCESS   # full scan each pattern
            if dist < best_dist:
                best_dist = dist
                best_pat = pat
        return best_pat


# ─────────────────────────────────────────────
# 4. Metrics
# ─────────────────────────────────────────────
def accuracy_exact(original: np.ndarray, recovered: np.ndarray) -> float:
    """1.0 if perfect match, else 0.0."""
    return 1.0 if np.array_equal(original, recovered) else 0.0


def accuracy_bitwise(original: np.ndarray, recovered: np.ndarray) -> float:
    """Fraction of matching bits."""
    return float(np.mean(original == recovered))


# ─────────────────────────────────────────────
# 5. Test Harness
# ─────────────────────────────────────────────
def run_experiment(n_patterns: int, noise_level: float, seed: int = 42):
    """Run one full experiment. Returns dict with results."""
    rng = np.random.default_rng(seed)
    patterns = generate_patterns(n_patterns, rng)

    # ── Build memories ──
    nano_mem = NanoLinkMemory()
    linear_mem = LinearMemory()
    for p in patterns:
        nano_mem.store(p)
        linear_mem.store(p)

    # ── Query with noise ──
    nano_exact, nano_bitwise = [], []
    linear_exact, linear_bitwise = [], []

    nano_mem.reset_cost()
    linear_mem.reset_cost()

    for p in patterns:
        noisy = add_noise(p, noise_level, rng)

        r_nano = nano_mem.query(noisy)
        nano_exact.append(accuracy_exact(p, r_nano))
        nano_bitwise.append(accuracy_bitwise(p, r_nano))

        r_linear = linear_mem.query(noisy)
        linear_exact.append(accuracy_exact(p, r_linear))
        linear_bitwise.append(accuracy_bitwise(p, r_linear))

    return {
        "n_patterns": n_patterns,
        "noise": noise_level,
        "nano_exact_acc": np.mean(nano_exact),
        "nano_bit_acc": np.mean(nano_bitwise),
        "nano_cost": nano_mem.cost,
        "linear_exact_acc": np.mean(linear_exact),
        "linear_bit_acc": np.mean(linear_bitwise),
        "linear_cost": linear_mem.cost,
    }


def run_all_experiments():
    """Run experiments for all (pattern_count, noise_level) combos × trials."""
    all_results = []

    combos = list(cartesian(PATTERN_COUNTS, NOISE_LEVELS))
    total = len(combos)

    for idx, (n_pat, noise) in enumerate(combos, 1):
        print(f"  [{idx}/{total}] patterns={n_pat}, noise={int(noise*100)}% ", end="", flush=True)
        trial_results = []
        for trial in range(NUM_TRIALS):
            seed = 1000 * n_pat + int(noise * 100) + trial
            r = run_experiment(n_pat, noise, seed=seed)
            trial_results.append(r)

        # Average across trials
        avg = {k: r[k] for k, r in [("n_patterns", trial_results[0]),
                                      ("noise", trial_results[0])]}
        for key in ["nano_exact_acc", "nano_bit_acc", "nano_cost",
                     "linear_exact_acc", "linear_bit_acc", "linear_cost"]:
            avg[key] = np.mean([t[key] for t in trial_results])

        all_results.append(avg)
        print(f"→ nano_bit={avg['nano_bit_acc']:.3f}  linear_exact={avg['linear_exact_acc']:.3f}")

    return all_results


# ─────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────
def plot_results(results: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Style ──
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560",
        "axes.labelcolor": "#eee",
        "text.color": "#eee",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.alpha": 0.5,
        "font.size": 11,
    })

    # ═══════════════════════════════════════════
    # Chart 1: Bitwise Accuracy vs Noise (per pattern count)
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, len(PATTERN_COUNTS), figsize=(5 * len(PATTERN_COUNTS), 5),
                             sharey=True)
    if len(PATTERN_COUNTS) == 1:
        axes = [axes]

    for ax, n_pat in zip(axes, PATTERN_COUNTS):
        subset = [r for r in results if r["n_patterns"] == n_pat]
        noises = [r["noise"] for r in subset]
        nano_acc = [r["nano_bit_acc"] for r in subset]
        linear_acc = [r["linear_bit_acc"] for r in subset]

        x = np.arange(len(noises))
        w = 0.35
        bars1 = ax.bar(x - w/2, nano_acc,   w, label="Nano-Link", color="#0f3460", edgecolor="#e94560")
        bars2 = ax.bar(x + w/2, linear_acc, w, label="Linear",    color="#533483", edgecolor="#e94560")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(n*100)}%" for n in noises])
        ax.set_xlabel("Noise Level")
        ax.set_title(f"{n_pat} Patterns", fontweight="bold")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y")

        # Value labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color="#eee")

    fig.suptitle("Bitwise Accuracy vs Noise Level", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_noise.png"), dpi=150)
    plt.close(fig)

    # ═══════════════════════════════════════════
    # Chart 2: Cost vs Pattern Count (scalability)
    # ═══════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 5))
    for noise in NOISE_LEVELS:
        subset = [r for r in results if r["noise"] == noise]
        pats = [r["n_patterns"] for r in subset]
        nano_cost = [r["nano_cost"] for r in subset]
        linear_cost = [r["linear_cost"] for r in subset]
        ax.plot(pats, nano_cost,   "o-", label=f"Nano-Link {int(noise*100)}%", linewidth=2)
        ax.plot(pats, linear_cost, "s--", label=f"Linear {int(noise*100)}%",   linewidth=2, alpha=0.7)

    ax.set_xlabel("Number of Stored Patterns")
    ax.set_ylabel("Simulated Access Cost")
    ax.set_title("Scalability: Cost vs Pattern Count", fontweight="bold", color="#e94560")
    ax.legend(fontsize=9)
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cost_scalability.png"), dpi=150)
    plt.close(fig)

    # ═══════════════════════════════════════════
    # Chart 3: Critical Test – 1000 patterns, 30% noise
    # ═══════════════════════════════════════════
    critical = [r for r in results if r["n_patterns"] == 1000 and r["noise"] == 0.30]
    if critical:
        c = critical[0]
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ["Exact Acc", "Bitwise Acc", "Cost (×10³)"]
        nano_vals = [c["nano_exact_acc"], c["nano_bit_acc"], c["nano_cost"] / 1000]
        linear_vals = [c["linear_exact_acc"], c["linear_bit_acc"], c["linear_cost"] / 1000]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, nano_vals,   w, label="Nano-Link", color="#0f3460", edgecolor="#e94560")
        ax.bar(x + w/2, linear_vals, w, label="Linear",    color="#533483", edgecolor="#e94560")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("CRITICAL TEST: 1000 patterns, 30% noise", fontweight="bold", color="#e94560")
        ax.legend()
        ax.grid(axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "critical_test.png"), dpi=150)
        plt.close(fig)

    # ═══════════════════════════════════════════
    # Chart 4: Exact Accuracy heatmaps side-by-side
    # ═══════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, system, title in [(ax1, "nano", "Nano-Link"), (ax2, "linear", "Linear")]:
        matrix = np.zeros((len(PATTERN_COUNTS), len(NOISE_LEVELS)))
        for r in results:
            i = PATTERN_COUNTS.index(r["n_patterns"])
            j = NOISE_LEVELS.index(r["noise"])
            matrix[i, j] = r[f"{system}_exact_acc"]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(NOISE_LEVELS)))
        ax.set_xticklabels([f"{int(n*100)}%" for n in NOISE_LEVELS])
        ax.set_yticks(range(len(PATTERN_COUNTS)))
        ax.set_yticklabels(PATTERN_COUNTS)
        ax.set_xlabel("Noise")
        ax.set_ylabel("Patterns")
        ax.set_title(title, fontweight="bold")

        for i in range(len(PATTERN_COUNTS)):
            for j in range(len(NOISE_LEVELS)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="black", fontweight="bold", fontsize=10)

    fig.suptitle("Exact Accuracy Heatmap", fontsize=14, fontweight="bold", color="#e94560")
    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8)
    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "accuracy_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"\n  📊 Charts saved to: {RESULTS_DIR}")


# ─────────────────────────────────────────────
# 7. Summary Table
# ─────────────────────────────────────────────
def print_summary(results: list[dict]):
    sep = "─" * 110
    print(f"\n{sep}")
    print(f"{'NANO-LINK TEST PROTOCOL – RESULTS':^110}")
    print(sep)
    print(f"{'Patterns':>10} {'Noise':>7} │ {'NL Exact':>10} {'NL Bit':>10} {'NL Cost':>12} │ "
          f"{'Lin Exact':>10} {'Lin Bit':>10} {'Lin Cost':>12} │ {'Δ Bit':>7}")
    print(sep)

    for r in results:
        delta = r["nano_bit_acc"] - r["linear_bit_acc"]
        marker = " ✅" if delta > 0.005 else (" ⚠️" if abs(delta) < 0.005 else " ❌")
        print(f"{r['n_patterns']:>10} {int(r['noise']*100):>6}% │ "
              f"{r['nano_exact_acc']:>10.4f} {r['nano_bit_acc']:>10.4f} {r['nano_cost']:>12.0f} │ "
              f"{r['linear_exact_acc']:>10.4f} {r['linear_bit_acc']:>10.4f} {r['linear_cost']:>12.0f} │ "
              f"{delta:>+7.4f}{marker}")

    print(sep)

    # Critical test verdict
    critical = [r for r in results if r["n_patterns"] == 1000 and r["noise"] == 0.30]
    if critical:
        c = critical[0]
        print(f"\n  🔬 CRITICAL TEST (1000 patterns, 30% noise):")
        print(f"     Nano-Link bitwise accuracy : {c['nano_bit_acc']:.4f}")
        print(f"     Linear   bitwise accuracy  : {c['linear_bit_acc']:.4f}")
        print(f"     Nano-Link cost             : {c['nano_cost']:.0f}")
        print(f"     Linear   cost              : {c['linear_cost']:.0f}")
        cost_ratio = c['linear_cost'] / max(c['nano_cost'], 1)
        print(f"     Cost ratio (Linear/Nano)   : {cost_ratio:.1f}×")

        if c["nano_bit_acc"] > c["linear_bit_acc"] + 0.01:
            print("\n  ✅ VERDICT: Nano-Link shows ADVANTAGE in associative retrieval")
        elif abs(c["nano_bit_acc"] - c["linear_bit_acc"]) < 0.01:
            print("\n  ⚠️  VERDICT: Results are COMPARABLE – no clear advantage")
        else:
            print("\n  ❌ VERDICT: Linear baseline outperforms Nano-Link")

    # Scalability analysis
    print(f"\n  📈 SCALABILITY ANALYSIS:")
    for noise in NOISE_LEVELS:
        subset_20 = [r for r in results if r["n_patterns"] == 20 and r["noise"] == noise]
        subset_1k = [r for r in results if r["n_patterns"] == 1000 and r["noise"] == noise]
        if subset_20 and subset_1k:
            nano_growth = subset_1k[0]["nano_cost"] / max(subset_20[0]["nano_cost"], 1)
            lin_growth = subset_1k[0]["linear_cost"] / max(subset_20[0]["linear_cost"], 1)
            print(f"     Noise {int(noise*100)}%: cost growth 20→1000 patterns | "
                  f"Nano-Link: {nano_growth:.1f}× | Linear: {lin_growth:.1f}×")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NANO-LINK TEST PROTOCOL")
    print("  Associative Retrieval Benchmark")
    print("=" * 60)
    print()

    t0 = time.time()
    print("  Running experiments...")
    results = run_all_experiments()
    elapsed = time.time() - t0

    print(f"\n  ⏱  Completed in {elapsed:.1f}s")

    print_summary(results)
    plot_results(results)

    print("  Done. Review the results above and charts in ./results/")

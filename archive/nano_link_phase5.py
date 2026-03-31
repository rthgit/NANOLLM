#!/usr/bin/env python3
"""
Nano-Link Phase 5 — Neural Routing Simulation
==============================================
Tests NanoLink as a GATING mechanism for sparse inference,
simulating a Mixture-of-Experts (MoE) layer.

Setup:
- N "experts" (random weight matrices)
- Each input token → router selects top-k experts → only those do compute
- Measure: routing accuracy (vs oracle), output quality (MSE vs full),
  memory reads saved, total cost

Routers tested:
1. NanoLink:    q·nano_weights per expert → top-k
2. DotProd:     q·expert_signature → top-k
3. LearnedGate: trained linear gating (like real MoE)
4. HashRouter:  LSH-based routing
5. Random:      random expert selection (lower bound)
6. Full:        all experts active (upper bound, highest cost)
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase5")

# ─────────────────────────────────────────────
# Cost Model (hardware-motivated)
# ─────────────────────────────────────────────
# Cost of reading one expert's weights = proportional to expert_dim²
# Cost of gating = much smaller (just routing logic)
C_EXPERT_READ = 1.0      # reading one expert block from memory
C_EXPERT_COMPUTE = 0.5   # computing expert output (matmul)
C_GATE_SCORE = 0.01      # scoring one expert in the router
C_GATE_SORT = 0.001      # sorting/selection overhead per expert
C_HASH = 0.005           # hash computation per expert

# ─────────────────────────────────────────────
# MoE Layer Simulation
# ─────────────────────────────────────────────
class MoELayer:
    """Simulated layer with N experts, each an (input_dim → expert_dim) matrix."""

    def __init__(self, n_experts, input_dim, expert_dim, rng):
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        # Each expert is a weight matrix
        self.experts = [rng.standard_normal((input_dim, expert_dim)).astype(np.float32)
                        for _ in range(n_experts)]
        # Expert "signatures" = mean of columns (compressed fingerprint)
        self.signatures = np.array([e.mean(axis=1) for e in self.experts])  # (N, input_dim)

    def forward_full(self, x):
        """Full forward: activate ALL experts, combine outputs."""
        outputs = [x @ e for e in self.experts]
        return np.mean(outputs, axis=0)

    def forward_sparse(self, x, expert_indices):
        """Sparse forward: only activate selected experts."""
        if len(expert_indices) == 0:
            return np.zeros(self.expert_dim, dtype=np.float32)
        outputs = [x @ self.experts[i] for i in expert_indices]
        return np.mean(outputs, axis=0)

    def oracle_topk(self, x, k):
        """Oracle: pick the k experts that produce output closest to full output."""
        full_out = self.forward_full(x)
        # Score each expert by how much it contributes to the full output
        scores = []
        for i, e in enumerate(self.experts):
            out_i = x @ e
            # How aligned is this expert's output with the full output?
            score = np.dot(out_i.ravel(), full_out.ravel())
            scores.append(score)
        scores = np.array(scores)
        return np.argpartition(scores, -k)[-k:]


# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────

class NanoLinkRouter:
    """NanoLink gating: learn nano weights from expert signatures, score with q·n."""
    NAME = "NanoLink"

    def __init__(self, moe, lr=0.15):
        # Nano weights = clipped scaled version of expert signatures
        self.nano = np.clip(np.abs(moe.signatures) * lr, 0, 1)  # (N, input_dim)
        self.n = moe.n_experts
        self.cost = 0.0

    def reset_cost(self): self.cost = 0.0

    def select(self, x, k):
        scores = self.nano @ np.abs(x)  # (N,)
        self.cost += self.n * C_GATE_SCORE
        self.cost += self.n * C_GATE_SORT
        return np.argpartition(scores, -k)[-k:]


class DotProdRouter:
    """Raw dot-product router: q · expert_signature."""
    NAME = "DotProd"

    def __init__(self, moe):
        self.sigs = moe.signatures
        self.n = moe.n_experts
        self.cost = 0.0

    def reset_cost(self): self.cost = 0.0

    def select(self, x, k):
        scores = self.sigs @ x  # (N,)
        self.cost += self.n * C_GATE_SCORE
        self.cost += self.n * C_GATE_SORT
        return np.argpartition(scores, -k)[-k:]


class LearnedGateRouter:
    """Simulated learned linear gate (like real MoE gating)."""
    NAME = "LearnedGate"

    def __init__(self, moe, rng):
        # In real MoE, gating weights are trained. We simulate this by using
        # SVD-compressed expert information as the gate.
        all_weights = np.stack([e.mean(axis=1) for e in moe.experts])  # (N, input_dim)
        # Add learned noise to simulate imperfect training
        self.gate_weights = all_weights + rng.standard_normal(all_weights.shape) * 0.1
        self.n = moe.n_experts
        self.cost = 0.0

    def reset_cost(self): self.cost = 0.0

    def select(self, x, k):
        scores = self.gate_weights @ x
        self.cost += self.n * C_GATE_SCORE
        self.cost += self.n * C_GATE_SORT
        return np.argpartition(scores, -k)[-k:]


class HashRouter:
    """LSH-based routing: hash input into buckets mapped to experts."""
    NAME = "HashRouter"

    def __init__(self, moe, rng, n_hashes=8):
        self.n_hashes = n_hashes
        self.planes = rng.standard_normal((n_hashes, moe.input_dim))
        # Map each expert to hash buckets based on its signature
        self.expert_buckets = {}
        for i in range(moe.n_experts):
            h = tuple((self.planes @ moe.signatures[i] > 0).astype(int))
            self.expert_buckets.setdefault(h, []).append(i)
        self.all_experts = list(range(moe.n_experts))
        self.n = moe.n_experts
        self.cost = 0.0

    def reset_cost(self): self.cost = 0.0

    def select(self, x, k):
        h = tuple((self.planes @ x > 0).astype(int))
        self.cost += self.n_hashes * C_HASH
        candidates = self.expert_buckets.get(h, self.all_experts)
        if len(candidates) <= k:
            return np.array(candidates)
        self.cost += len(candidates) * C_GATE_SORT
        # Among candidates, just take first k
        return np.array(candidates[:k])


class RandomRouter:
    """Random baseline: pick k experts at random."""
    NAME = "Random"

    def __init__(self, moe, rng):
        self.n = moe.n_experts
        self.rng = rng
        self.cost = 0.0

    def reset_cost(self): self.cost = 0.0

    def select(self, x, k):
        self.cost += k * C_GATE_SORT
        return self.rng.choice(self.n, size=k, replace=False)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def output_quality(full_out, sparse_out):
    """Cosine similarity between full and sparse output."""
    norm_f = np.linalg.norm(full_out)
    norm_s = np.linalg.norm(sparse_out)
    if norm_f < 1e-10 or norm_s < 1e-10:
        return 0.0
    return float(np.dot(full_out, sparse_out) / (norm_f * norm_s))

def routing_accuracy(selected, oracle):
    """Fraction of oracle-selected experts that were also selected."""
    return len(set(selected) & set(oracle)) / len(oracle)

def memory_saved(n_experts, k):
    """Fraction of expert reads saved."""
    return 1.0 - (k / n_experts)


# ─────────────────────────────────────────────
# Test Runners
# ─────────────────────────────────────────────

def run_routing_test(n_experts, input_dim, expert_dim, k, n_tokens, seed=42):
    """Run one routing experiment with all routers."""
    rng = np.random.default_rng(seed)
    moe = MoELayer(n_experts, input_dim, expert_dim, rng)

    # Create routers
    routers = {
        "NanoLink":    NanoLinkRouter(moe, lr=0.15),
        "DotProd":     DotProdRouter(moe),
        "LearnedGate": LearnedGateRouter(moe, rng),
        "HashRouter":  HashRouter(moe, rng),
        "Random":      RandomRouter(moe, rng),
    }

    # Generate input tokens
    tokens = rng.standard_normal((n_tokens, input_dim)).astype(np.float32)

    results = {}
    for rname, router in routers.items():
        router.reset_cost()
        qualities = []
        route_accs = []

        for x in tokens:
            full_out = moe.forward_full(x)
            selected = router.select(x, k)
            sparse_out = moe.forward_sparse(x, selected)
            oracle = moe.oracle_topk(x, k)

            qualities.append(output_quality(full_out, sparse_out))
            route_accs.append(routing_accuracy(selected, oracle))

        total_cost = router.cost + n_tokens * k * (C_EXPERT_READ + C_EXPERT_COMPUTE)

        results[rname] = {
            "quality": np.mean(qualities),
            "route_acc": np.mean(route_accs),
            "gate_cost": router.cost,
            "total_cost": total_cost,
            "mem_saved": memory_saved(n_experts, k),
        }

    # Full (no gating)
    results["Full"] = {
        "quality": 1.0,
        "route_acc": 1.0,
        "gate_cost": 0,
        "total_cost": n_tokens * n_experts * (C_EXPERT_READ + C_EXPERT_COMPUTE),
        "mem_saved": 0.0,
    }

    return results


# ═════════════════════════════════════════════
# TEST 1: Expert Count Scaling
# ═════════════════════════════════════════════
def test_expert_scaling():
    print("\n" + "="*80)
    print("  TEST 1: Expert Count Scaling (top-4, input=64, expert=32, 200 tokens)")
    print("="*80)

    expert_counts = [8, 16, 32, 64, 128, 256]
    k = 4
    all_results = []

    for n_exp in expert_counts:
        print(f"\n  N_experts={n_exp}")
        r = run_routing_test(n_exp, input_dim=64, expert_dim=32, k=k, n_tokens=200)
        row = {"n_experts": n_exp}
        for rname, metrics in r.items():
            for mname, mval in metrics.items():
                row[f"{rname}_{mname}"] = mval
            print(f"    {rname:>12}: quality={metrics['quality']:.4f}  "
                  f"route_acc={metrics['route_acc']:.4f}  "
                  f"total_cost={metrics['total_cost']:.0f}  "
                  f"mem_saved={metrics['mem_saved']:.1%}")
        all_results.append(row)
    return all_results


# ═════════════════════════════════════════════
# TEST 2: Sparsity Level (vary k)
# ═════════════════════════════════════════════
def test_sparsity_level():
    print("\n" + "="*80)
    print("  TEST 2: Sparsity Level (64 experts, input=64, expert=32, 200 tokens)")
    print("="*80)

    n_exp = 64
    k_values = [1, 2, 4, 8, 16, 32, 64]
    all_results = []

    for k in k_values:
        print(f"\n  k={k} (top-{k} of {n_exp})")
        r = run_routing_test(n_exp, input_dim=64, expert_dim=32, k=k, n_tokens=200)
        row = {"k": k}
        for rname, metrics in r.items():
            for mname, mval in metrics.items():
                row[f"{rname}_{mname}"] = mval
            print(f"    {rname:>12}: quality={metrics['quality']:.4f}  "
                  f"route_acc={metrics['route_acc']:.4f}  "
                  f"mem_saved={metrics['mem_saved']:.1%}")
        all_results.append(row)
    return all_results


# ═════════════════════════════════════════════
# TEST 3: Input Dimensionality
# ═════════════════════════════════════════════
def test_input_dim():
    print("\n" + "="*80)
    print("  TEST 3: Input Dimensionality (64 experts, top-4, 200 tokens)")
    print("="*80)

    dims = [16, 32, 64, 128, 256, 512]
    all_results = []

    for dim in dims:
        print(f"\n  input_dim={dim}")
        r = run_routing_test(64, input_dim=dim, expert_dim=dim//2, k=4, n_tokens=200)
        row = {"dim": dim}
        for rname, metrics in r.items():
            for mname, mval in metrics.items():
                row[f"{rname}_{mname}"] = mval
            print(f"    {rname:>12}: quality={metrics['quality']:.4f}  "
                  f"route_acc={metrics['route_acc']:.4f}")
        all_results.append(row)
    return all_results


# ═════════════════════════════════════════════
# TEST 4: Efficiency Frontier
# ═════════════════════════════════════════════
def test_efficiency():
    """Plot quality vs cost for all routers across k values — the pareto frontier."""
    print("\n" + "="*80)
    print("  TEST 4: Efficiency Frontier (64 experts, input=64)")
    print("="*80)

    n_exp = 64
    k_values = [1, 2, 4, 8, 16, 32]
    all_results = []

    for k in k_values:
        r = run_routing_test(n_exp, input_dim=64, expert_dim=32, k=k, n_tokens=200)
        for rname, metrics in r.items():
            all_results.append({
                "router": rname, "k": k,
                **metrics
            })
    return all_results


# ═════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════
ROUTER_COLORS = {
    "NanoLink": "#16c79a", "DotProd": "#f5a623", "LearnedGate": "#e94560",
    "HashRouter": "#533483", "Random": "#888888", "Full": "#0f3460",
}

def plot_all(scaling, sparsity, dims, efficiency):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#aaa", "ytick.color": "#aaa",
        "grid.color": "#333", "grid.alpha": 0.5, "font.size": 11,
        "legend.facecolor": "#16213e", "legend.edgecolor": "#444",
    })
    routers = ["NanoLink", "DotProd", "LearnedGate", "HashRouter", "Random", "Full"]

    # ── 1. Expert scaling: quality ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ns = [r["n_experts"] for r in scaling]
    for rn in routers:
        q = [r[f"{rn}_quality"] for r in scaling]
        c = [r[f"{rn}_total_cost"] for r in scaling]
        ax1.plot(ns, q, "o-", color=ROUTER_COLORS[rn], label=rn, lw=2, ms=6)
        ax2.plot(ns, c, "o-", color=ROUTER_COLORS[rn], label=rn, lw=2, ms=6)
    ax1.set_title("Output Quality vs #Experts", fontweight="bold"); ax1.set_ylim(0, 1.05)
    ax2.set_title("Total Cost vs #Experts", fontweight="bold")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xlabel("#Experts"); ax.legend(fontsize=8); ax.grid(True)
    fig.suptitle("Expert Count Scaling (top-4)", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "expert_scaling.png"), dpi=150)
    plt.close(fig)

    # ── 2. Sparsity level ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ks = [r["k"] for r in sparsity]
    for rn in routers:
        q = [r.get(f"{rn}_quality", 0) for r in sparsity]
        ra = [r.get(f"{rn}_route_acc", 0) for r in sparsity]
        ax1.plot(ks, q, "o-", color=ROUTER_COLORS[rn], label=rn, lw=2, ms=6)
        ax2.plot(ks, ra, "o-", color=ROUTER_COLORS[rn], label=rn, lw=2, ms=6)
    ax1.set_title("Output Quality vs k", fontweight="bold"); ax1.set_ylim(0, 1.05)
    ax2.set_title("Routing Accuracy vs k", fontweight="bold"); ax2.set_ylim(0, 1.05)
    for ax in (ax1, ax2):
        ax.set_xlabel("k (experts selected)"); ax.legend(fontsize=8); ax.grid(True)
    fig.suptitle("Sparsity Level (64 experts)", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "sparsity_level.png"), dpi=150)
    plt.close(fig)

    # ── 3. Efficiency frontier ──
    fig, ax = plt.subplots(figsize=(10, 7))
    for rn in routers:
        sub = [r for r in efficiency if r["router"] == rn]
        costs = [r["total_cost"] for r in sub]
        quals = [r["quality"] for r in sub]
        ax.plot(costs, quals, "o-", color=ROUTER_COLORS[rn], label=rn, lw=2, ms=8)
        # Label the k values
        for r in sub:
            if r["router"] != "Full":
                ax.annotate(f"k={r['k']}", (r["total_cost"], r["quality"]),
                           textcoords="offset points", xytext=(5, 5), fontsize=7, color="#aaa")
    ax.set_xlabel("Total Cost"); ax.set_ylabel("Output Quality")
    ax.set_title("Efficiency Frontier: Quality vs Cost", fontweight="bold", color="#e94560")
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "efficiency_frontier.png"), dpi=150)
    plt.close(fig)

    # ── 4. Head-to-head NanoLink vs DotProd vs LearnedGate ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for rn in ["NanoLink", "DotProd", "LearnedGate"]:
        sub = [r for r in efficiency if r["router"] == rn]
        ks_ = [r["k"] for r in sub]
        q_ = [r["quality"] for r in sub]
        c_ = [r["total_cost"] for r in sub]
        ax1.plot(ks_, q_, "o-", color=ROUTER_COLORS[rn], label=rn, lw=2, ms=7)
        ax2.plot(ks_, c_, "o-", color=ROUTER_COLORS[rn], label=rn, lw=2, ms=7)
    ax1.set_title("Quality (NL vs DP vs Gate)", fontweight="bold"); ax1.set_ylim(0, 1.05)
    ax2.set_title("Cost (NL vs DP vs Gate)", fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xlabel("k"); ax.legend(); ax.grid(True)
    fig.suptitle("Head-to-Head: Gating Methods", fontsize=14, fontweight="bold", color="#e94560")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, "head_to_head.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Charts saved to: {RESULTS_DIR}")


# ═════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════
def print_summary(scaling, sparsity, efficiency):
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"{'PHASE 5 — NEURAL ROUTING RESULTS':^100}")
    print(sep)

    # Best at 256 experts
    if scaling:
        big = scaling[-1]
        print(f"\n  AT {big['n_experts']} EXPERTS (top-4):")
        routers = ["NanoLink", "DotProd", "LearnedGate", "HashRouter", "Random"]
        for rn in routers:
            q = big[f"{rn}_quality"]
            c = big[f"{rn}_total_cost"]
            m = big[f"{rn}_mem_saved"]
            print(f"    {rn:>12}: quality={q:.4f}  cost={c:.0f}  mem_saved={m:.1%}")
        q_full = big["Full_quality"]
        c_full = big["Full_total_cost"]
        print(f"    {'Full':>12}: quality={q_full:.4f}  cost={c_full:.0f}")

    # NanoLink vs DotProd gap
    print(f"\n  NANOLINK vs DOTPROD GAP (across expert counts):")
    for row in scaling:
        nl_q = row["NanoLink_quality"]
        dp_q = row["DotProd_quality"]
        print(f"    N={row['n_experts']:<4}: NL={nl_q:.4f}  DP={dp_q:.4f}  diff={nl_q-dp_q:+.4f}")

    # Best sparsity point
    print(f"\n  SPARSITY SWEEP (quality at different k):")
    for row in sparsity:
        print(f"    k={row['k']:<3}  NL={row['NanoLink_quality']:.4f}  "
              f"DP={row['DotProd_quality']:.4f}  "
              f"Gate={row['LearnedGate_quality']:.4f}  "
              f"mem_saved={row['NanoLink_mem_saved']:.0%}")

    print(f"\n{sep}\n")


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  NANO-LINK PHASE 5 — NEURAL ROUTING")
    print("  Can NanoLink serve as an efficient MoE gating mechanism?")
    print("=" * 80)

    t0 = time.time()
    scaling    = test_expert_scaling()
    sparsity   = test_sparsity_level()
    dims       = test_input_dim()
    efficiency = test_efficiency()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_summary(scaling, sparsity, efficiency)
    plot_all(scaling, sparsity, dims, efficiency)
    print("  Done.")

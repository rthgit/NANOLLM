#!/usr/bin/env python3
"""
Nano-Link Phase 7 — Parameter-Free Baselines & Load Balancing
==============================================================
Tests Nano-Link against other parameter-free routing heuristics
under the constraint of an Auxiliary Load-Balancing Loss.

Routers Tested:
1. StandardGate (Learned baseline with parameters)
2. NanoLinkGate (Score = abs(W).mean(0) * x)
3. MeanGate     (Score = W.mean(0) * x)
4. NormGate     (Score = L2Norm(W, dim=0) * x)
5. FrozenRandom (Score = FixedRandom_i * x)

We apply a Load-Balancing Loss (alpha * N * f_i * P_i) to force the Gini 
coefficient down. We want to see which parameter-free method preserves 
the best task loss while successfully balancing the load.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase7")

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
VOCAB_SIZE = 256
D_MODEL = 64
HIDDEN_DIM = 128
N_EXPERTS = 8
TOP_K = 2
SEQ_LEN = 32
BATCH_SIZE = 32
N_STEPS = 600
LR = 1e-3
AUX_LOSS_WEIGHT = 0.1  # Strength of the load-balancing penalty

# ─────────────────────────────────────────────
# Core Modules
# ─────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

# ── Routers ──

class StandardRouter(nn.Module):
    """Learned linear gate."""
    NAME = "StandardGate"
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
    def forward(self, x, experts):
        return self.gate(x)


class NanoLinkRouter(nn.Module):
    """Nano-Link: signature = mean of absolute incoming weights."""
    NAME = "NanoLinkGate"
    def __init__(self, d_model, n_experts):
        super().__init__()

    def forward(self, x, experts):
        signatures = []
        for exp in experts:
            W_in = exp.net[0].weight
            sig = torch.mean(torch.abs(W_in), dim=0) # (d_model,)
            signatures.append(sig)
        signatures = torch.stack(signatures, dim=0) # (E, D)
        # NanoLink scoring: q * n (we use absolute query to match NanoLink strict positive logic)
        scores = torch.abs(x) @ signatures.T # (B*S, E)
        return scores


class MeanRouter(nn.Module):
    """Mean Gate: signature = raw mean of incoming weights (no abs)."""
    NAME = "MeanGate"
    def __init__(self, d_model, n_experts):
        super().__init__()

    def forward(self, x, experts):
        signatures = []
        for exp in experts:
            W_in = exp.net[0].weight
            sig = torch.mean(W_in, dim=0) # (d_model,)
            signatures.append(sig)
        signatures = torch.stack(signatures, dim=0)
        scores = x @ signatures.T # standard dot product
        return scores


class NormRouter(nn.Module):
    """Norm Gate: signature = L2 norm of the incoming weights."""
    NAME = "NormGate"
    def __init__(self, d_model, n_experts):
        super().__init__()

    def forward(self, x, experts):
        signatures = []
        for exp in experts:
            W_in = exp.net[0].weight
            sig = torch.norm(W_in, p=2, dim=0) # (d_model,)
            signatures.append(sig)
        signatures = torch.stack(signatures, dim=0)
        scores = torch.abs(x) @ signatures.T
        return scores


class FrozenRandomRouter(nn.Module):
    """Frozen Random Keys: assigns a random, fixed key to each expert."""
    NAME = "FrozenRandGate"
    def __init__(self, d_model, n_experts):
        super().__init__()
        # Register as a buffer so it doesn't get updated by the optimizer
        self.register_buffer("keys", torch.randn(n_experts, d_model))

    def forward(self, x, experts):
        scores = x @ self.keys.T
        return scores

# ─────────────

class MoELayer(nn.Module):
    def __init__(self, d_model, hidden_dim, n_experts, top_k, RouterClass):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, hidden_dim) for _ in range(n_experts)])
        self.router = RouterClass(d_model, n_experts)
        self.register_buffer("expert_load", torch.zeros(n_experts))

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        router_logits = self.router(x_flat, self.experts)
        
        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise
            
        routing_weights = F.softmax(router_logits, dim=-1) # (B*S, E)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # ── Calculate Auxiliary Load-Balancing Loss ──
        # Let f_i be the fraction of tokens routed to expert i (ignoring routing probabilities, just based on top-1 choice for load)
        # Let P_i be the mean routing probability for expert i across the batch
        # L_aux = alpha * N * sum(f_i * P_i)
        
        top1_indices = top_k_indices[:, 0] # (B*S)
        expert_mask = F.one_hot(top1_indices, num_classes=self.n_experts).float() # (B*S, E)
        f_i = expert_mask.mean(dim=0) # (E,) fraction of tokens
        P_i = routing_weights.mean(dim=0) # (E,) mean probability
        
        aux_loss = self.n_experts * torch.sum(f_i * P_i)
        
        # Track load for reporting
        with torch.no_grad():
            self.expert_load += expert_mask.sum(dim=0)
            
        # Route
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            any_mask = mask.any(dim=-1)
            
            if any_mask.any():
                expert_tokens = x_flat[any_mask]
                expert_out = expert(expert_tokens)
                
                weight_idx = mask[any_mask].float().argmax(dim=-1)
                token_weights = top_k_weights[any_mask, weight_idx].unsqueeze(-1)
                out[any_mask] += expert_out * token_weights
                
        return out.view(batch_size, seq_len, d_model), aux_loss


class DummyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_dim, n_experts, top_k, RouterClass):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        
        self.attn1 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn1 = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))
        self.norm1a = nn.LayerNorm(d_model)
        self.norm1b = nn.LayerNorm(d_model)

        self.attn2 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.moe = MoELayer(d_model, hidden_dim, n_experts, top_k, RouterClass)
        self.norm2a = nn.LayerNorm(d_model)
        self.norm2b = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        
        x2, _ = self.attn1(self.norm1a(x), self.norm1a(x), self.norm1a(x))
        x = x + x2
        x = x + self.ffn1(self.norm1b(x))
        
        x2, _ = self.attn2(self.norm2a(x), self.norm2a(x), self.norm2a(x))
        x = x + x2
        moe_out, aux_loss = self.moe(self.norm2b(x))
        x = x + moe_out
        
        logits = self.head(x)
        return logits, aux_loss


# ─────────────────────────────────────────────
# Synthetic Dataset & Training
# ─────────────────────────────────────────────
def generate_synthetic_data(num_samples, seq_len, vocab_size, seed=42):
    torch.manual_seed(seed)
    templates = torch.randint(0, vocab_size, (5, seq_len))
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 5, (1,)).item()].clone()
        noise_idx = torch.randperm(seq_len)[:2]
        base[noise_idx] = torch.randint(0, vocab_size, (2,))
        data.append(base)
    return torch.stack(data)


def compute_gini(load_array):
    if len(load_array) == 0 or load_array.sum() == 0:
        return 1.0
    load_array = np.sort(np.asarray(load_array))
    n = len(load_array)
    mean_load = np.mean(load_array)
    gini = (2.0 * np.sum((np.arange(1, n + 1) * load_array)) / (n * load_array.sum())) - (n + 1) / n
    return float(gini)


def train_model(RouterClass, data, device):
    torch.manual_seed(1337)
    
    model = DummyTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, hidden_dim=HIDDEN_DIM, 
        n_experts=N_EXPERTS, top_k=TOP_K, RouterClass=RouterClass
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    dataset_size = data.size(0)
    
    task_loss_history = []
    total_loss_history = []
    start_time = time.time()
    
    model.train()
    for step in range(N_STEPS):
        idx = torch.randint(0, dataset_size, (BATCH_SIZE,))
        batch = data[idx].to(device)
        
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        optimizer.zero_grad()
        logits, aux_loss = model(x)
        
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        # Combine losses
        total_loss = task_loss + AUX_LOSS_WEIGHT * aux_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        task_loss_history.append(task_loss.item())
        total_loss_history.append(total_loss.item())
        
        if (step + 1) % 150 == 0:
            print(f"  [{RouterClass.NAME:14}] Step {step+1:3d} | Task L: {task_loss.item():.4f} | Aux L: {aux_loss.item():.4f}")
            
    train_time = time.time() - start_time
    
    raw_load = model.moe.expert_load.cpu().numpy()
    total_routes = raw_load.sum()
    load_pct = (raw_load / total_routes) * 100 if total_routes > 0 else raw_load
    gini = compute_gini(raw_load)

    # Calculate smoothed final task loss
    final_loss = np.mean(task_loss_history[-20:])

    return {
        "task_loss": task_loss_history,
        "final_loss": final_loss,
        "load": load_pct,
        "gini": gini,
        "time": train_time
    }


def plot_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444", "axes.labelcolor": "#eee",
        "text.color": "#eee", "xtick.color": "#aaa", "ytick.color": "#aaa",
        "grid.color": "#333", "grid.alpha": 0.5, "font.size": 11,
        "legend.facecolor": "#16213e", "legend.edgecolor": "#444",
    })
    
    colors = {
        "StandardGate": "#e94560", 
        "NanoLinkGate": "#16c79a",
        "MeanGate": "#f5a623",
        "NormGate": "#0f3460",
        "FrozenRandGate": "#888888"
    }
    
    # 1. Training Convergence (Task Loss)
    fig, ax = plt.subplots(figsize=(12, 7))
    for rname, res in results.items():
        raw_loss = np.array(res["task_loss"])
        smoothed = np.convolve(raw_loss, np.ones(15)/15, mode='valid')
        ax.plot(smoothed, label=f"{rname} (L: {res['final_loss']:.3f}, G: {res['gini']:.2f})", 
                color=colors.get(rname, "#fff"), lw=2)
        
    ax.set_title("Training Task Loss (with Aux Load Balancing)", fontweight="bold")
    ax.set_xlabel("Steps"); ax.set_ylabel("Cross Entropy Loss")
    ax.grid(True); ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "p7_training_loss.png"), dpi=150)
    plt.close(fig)
    
    # 2. Expert Load Distribution Bar Chart
    fig, axes = plt.subplots(1, len(results), figsize=(18, 4), sharey=True)
    x_pos = np.arange(N_EXPERTS)
    ideal_pct = 100 / N_EXPERTS
    
    for ax, (rname, res) in zip(axes, results.items()):
        ax.bar(x_pos, res["load"], color=colors.get(rname, "#fff"), edgecolor="#eee")
        ax.axhline(ideal_pct, color="#f5a623", linestyle="--", alpha=0.7)
        ax.set_title(f"{rname}\nGini: {res['gini']:.2f}\nLoss: {res['final_loss']:.3f}", fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in x_pos], fontsize=8)
        
    axes[0].set_ylabel("% of Tokens Handled")
    fig.suptitle("Expert Load Balancing Across Routers", fontsize=14, fontweight="bold", color="#eee", y=1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "p7_expert_load.png"), dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════
# Main Execution
# ═════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  NANO-LINK PHASE 7 — BASELINES & LOAD BALANCING")
    print("  Testing 5 routers with Auxiliary Balancing Loss (alpha_aux=0.1)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}\n")
    
    data = generate_synthetic_data(num_samples=500, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    
    routers = [StandardRouter, NanoLinkRouter, MeanRouter, NormRouter, FrozenRandomRouter]
    
    results = {}
    print(f"{'ROUTER':<18} | {'FINAL TASK LOSS':<15} | {'GINI COEFF':<12} | {'TIME':<10}")
    print("-" * 65)
    
    for RClass in routers:
        res = train_model(RClass, data, device)
        results[RClass.NAME] = res
        
        loss = res["final_loss"]
        gini = res["gini"]
        t = res["time"]
        
        # Highlight NanoLink
        star = "*" if RClass.NAME == "NanoLinkGate" else " "
        print(f"{star}{RClass.NAME:<17} | {loss:<15.4f} | {gini:<12.3f} | {t:<6.2f}s")
        
    plot_results(results)
    print("\n  Charts saved to: " + RESULTS_DIR)
    print("  Done.")

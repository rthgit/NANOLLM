#!/usr/bin/env python3
"""
Nano-Link Phase 6 — Realistic PyTorch MoE Integration
======================================================
Tests Nano-Link as an associative routing prefilter in a real neural network.
We implement a small Transformer model where the Feed-Forward Network (FFN)
is replaced by a Mixture-of-Experts (MoE) layer.

We compare:
1. Standard Learned Gating (a linear layer that learns to route)
2. NanoLink Gating (routing scores are dynamically extracted from the experts' 
   incoming weight signatures, requiring no independent router parameters).

The task is to overfit a small dataset of sequences. We measure:
- Training loss curve (does Nano-Link gradient flow allow learning?)
- Expert load balancing (does it collapse to a single expert?)
- Forward pass time.
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase6")

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
N_STEPS = 500
LR = 1e-3

# ─────────────────────────────────────────────
# Core Modules
# ─────────────────────────────────────────────

class Expert(nn.Module):
    """Standard FFN expert."""
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        
    def forward(self, x):
        return self.net(x)


class StandardRouter(nn.Module):
    """Traditional MoE Router: a learned linear projection."""
    NAME = "StandardGate"
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
    def forward(self, x, experts):
        # x: (batch * seq_len, d_model)
        # return logits: (batch * seq_len, n_experts)
        return self.gate(x)


class NanoLinkRouter(nn.Module):
    """
    Nano-Link Router: no independent router parameters!
    Extracts a routing signature dynamically from the experts' weights.
    The gradient flows back through the router directly into the experts.
    """
    NAME = "NanoLinkGate"
    def __init__(self, d_model, n_experts):
        super().__init__()
        # We don't initialize any parameters here.
        # It relies purely on the experts' structure.

    def forward(self, x, experts):
        # x: (batch * seq_len, d_model)
        signatures = []
        for exp in experts:
            # Extract the incoming weights of the first Linear layer: (hidden_dim, d_model)
            W_in = exp.net[0].weight
            # The Nano-Link signature is the absolute mean of the weights across the hidden dimension
            # We use absolute value because Nano-Link originally used positive link weights (0 to 1).
            sig = torch.mean(torch.abs(W_in), dim=0) # (d_model,)
            signatures.append(sig)
            
        # Stack into (n_experts, d_model)
        signatures = torch.stack(signatures, dim=0)
        
        # Routing score is the dot product between the absolute input query and the signature
        # This mirrors Nano-Link's q * n_s scoring mechanism.
        scores = torch.abs(x) @ signatures.T # (batch * seq_len, n_experts)
        return scores


class MoELayer(nn.Module):
    """MoE layer combining experts and a router."""
    def __init__(self, d_model, hidden_dim, n_experts, top_k, RouterClass):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, hidden_dim) for _ in range(n_experts)])
        self.router = RouterClass(d_model, n_experts)
        # To track load balancing
        self.register_buffer("expert_load", torch.zeros(n_experts))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # (B*S, D)
        
        # 1. Get routing scores (logits)
        router_logits = self.router(x_flat, self.experts) # (B*S, E)
        
        # 2. Add noise for exploration (standard practice in MoE training)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise
            
        # 3. Top-k selection
        routing_weights = F.softmax(router_logits, dim=-1) # (B*S, E)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1) # (B*S, K)
        
        # Re-normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Track load
        with torch.no_grad():
            flat_indices = top_k_indices.view(-1)
            counts = torch.bincount(flat_indices, minlength=self.n_experts)
            self.expert_load += counts.float()
            
        # 4. Route and compute
        out = torch.zeros_like(x_flat) # (B*S, D)
        
        # Loop over experts (simulating routing logic)
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this expert (can be in any of the top-k slots)
            expert_mask = (top_k_indices == i) # (B*S, K)
            any_token_mask = expert_mask.any(dim=-1) # (B*S,)
            
            if any_token_mask.any():
                # Extract tokens
                expert_tokens = x_flat[any_token_mask] # (N, D)
                # Compute expert
                expert_out = expert(expert_tokens) # (N, D)
                
                # Find the corresponding routing weight
                # (extract the weight from the specific k-slot where this expert was chosen)
                weight_idx = expert_mask[any_token_mask].float().argmax(dim=-1) # (N,)
                token_weights = top_k_weights[any_token_mask, weight_idx].unsqueeze(-1) # (N, 1)
                
                # Add scaled output
                out[any_token_mask] += expert_out * token_weights
                
        return out.view(batch_size, seq_len, d_model)


class DummyTransformer(nn.Module):
    """A minimal 2-layer Transformer. The 2nd FFN is replaced by MoE."""
    def __init__(self, vocab_size, d_model, hidden_dim, n_experts, top_k, RouterClass):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        
        # Layer 1: Standard dense FFN
        self.attn1 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model)
        )
        self.norm1a = nn.LayerNorm(d_model)
        self.norm1b = nn.LayerNorm(d_model)

        # Layer 2: MoE Layer
        self.attn2 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.moe = MoELayer(d_model, hidden_dim, n_experts, top_k, RouterClass)
        self.norm2a = nn.LayerNorm(d_model)
        self.norm2b = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        
        # L1
        x2, _ = self.attn1(self.norm1a(x), self.norm1a(x), self.norm1a(x))
        x = x + x2
        x = x + self.ffn1(self.norm1b(x))
        
        # L2 (MoE)
        x2, _ = self.attn2(self.norm2a(x), self.norm2a(x), self.norm2a(x))
        x = x + x2
        x = x + self.moe(self.norm2b(x))
        
        logits = self.head(x)
        return logits


# ─────────────────────────────────────────────
# Synthetic Dataset
# ─────────────────────────────────────────────
def generate_synthetic_data(num_samples, seq_len, vocab_size, seed=42):
    """Generate fixed sequence structures so there is a pattern to learn."""
    torch.manual_seed(seed)
    # Generate 5 'prototypical' sequence patterns
    templates = torch.randint(0, vocab_size, (5, seq_len))
    
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 5, (1,)).item()].clone()
        # Add a little noise (change 2 tokens randomly)
        noise_idx = torch.randperm(seq_len)[:2]
        base[noise_idx] = torch.randint(0, vocab_size, (2,))
        data.append(base)
        
    return torch.stack(data)


# ─────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────
def train_model(RouterClass, data, device):
    torch.manual_seed(1337)
    
    model = DummyTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, hidden_dim=HIDDEN_DIM, 
        n_experts=N_EXPERTS, top_k=TOP_K, RouterClass=RouterClass
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    dataset_size = data.size(0)
    
    loss_history = []
    start_time = time.time()
    
    model.train()
    for step in range(N_STEPS):
        # Random batch
        idx = torch.randint(0, dataset_size, (BATCH_SIZE,))
        batch = data[idx].to(device)
        
        # Inputs and targets (shifted right)
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        optimizer.zero_grad()
        logits = model(x) # (B, S-1, V)
        
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        loss_history.append(loss.item())
        
        if (step + 1) % 100 == 0:
            print(f"  [{RouterClass.NAME:12}] Step {step+1:3d} | Loss: {loss.item():.4f}")
            
    train_time = time.time() - start_time
    
    # Get expert load percentages
    raw_load = model.moe.expert_load.cpu().numpy()
    total_routes = raw_load.sum()
    if total_routes > 0:
        load_pct = (raw_load / total_routes) * 100
    else:
        load_pct = raw_load

    # Metric: Gini coefficient of load (0 = perfect balance, 1 = total collapse)
    sorted_load = np.sort(raw_load)
    index = np.arange(1, len(raw_load) + 1)
    if total_routes > 0:
        gini = ((np.sum((2 * index - len(raw_load) - 1) * sorted_load)) / (len(raw_load) * total_routes))
    else:
        gini = 1.0

    return {
        "loss": loss_history,
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
    
    # 1. Training Convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"StandardGate": "#e94560", "NanoLinkGate": "#16c79a"}
    
    for rname, res in results.items():
        # Smooth loss curve
        raw_loss = np.array(res["loss"])
        smoothed = np.convolve(raw_loss, np.ones(10)/10, mode='valid')
        ax.plot(smoothed, label=f"{rname} (final loss: {raw_loss[-1]:.3f})", 
                color=colors.get(rname, "#ffffff"), lw=2)
        
    ax.set_title("Training Convergence (Transformer MoE)", fontweight="bold")
    ax.set_xlabel("Steps"); ax.set_ylabel("Cross Entropy Loss")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_loss.png"), dpi=150)
    plt.close(fig)
    
    # 2. Expert Load Balancing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(N_EXPERTS)
    
    nl_load = results["NanoLinkGate"]["load"]
    ax1.bar(x_pos, nl_load, color="#16c79a", edgecolor="#eee")
    ax1.set_title(f"NanoLink Expert Load\n(Gini: {results['NanoLinkGate']['gini']:.2f})", fontweight="bold")
    ax1.set_ylim(0, max(nl_load.max(), results["StandardGate"]["load"].max()) + 5)
    
    std_load = results["StandardGate"]["load"]
    ax2.bar(x_pos, std_load, color="#e94560", edgecolor="#eee")
    ax2.set_title(f"StandardGate Expert Load\n(Gini: {results['StandardGate']['gini']:.2f})", fontweight="bold")
    ax2.set_ylim(0, max(nl_load.max(), std_load.max()) + 5)
    
    # Ideal load line
    ideal = (TOP_K / N_EXPERTS) * 100 * (N_EXPERTS/TOP_K) / (N_EXPERTS/TOP_K) # Wait, it's just 100/N_EXPERTS
    ideal_pct = 100 / N_EXPERTS
    for ax in (ax1, ax2):
        ax.axhline(ideal_pct, color="#f5a623", linestyle="--", label="Ideal Balance")
        ax.set_xlabel("Expert ID"); ax.set_ylabel("% of Tokens Handled")
        ax.legend()
        ax.grid(axis="y")
        
    fig.suptitle("Routing Distribution / Load Balancing", fontsize=14, fontweight="bold", color="#eee")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "expert_load.png"), dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════
# Main Execution
# ═════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  NANO-LINK PHASE 6 — PYTORCH MOE INTEGRATION")
    print("  Testing dynamic routing extracted from expert signatures")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}\n")
    
    # Generate 500 sequences
    data = generate_synthetic_data(num_samples=500, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    
    results = {}
    for RouterClass in [StandardRouter, NanoLinkRouter]:
        print(f"--- Training {RouterClass.NAME} ---")
        res = train_model(RouterClass, data, device)
        results[RouterClass.NAME] = res
        print(f"  Final Gini Coeff: {res['gini']:.3f} (Lower = better balance)")
        print(f"  Train Time: {res['time']:.2f}s\n")
        
    plot_results(results)
    print(f"  Charts saved to: {RESULTS_DIR}")
    print("  Done.")

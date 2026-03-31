#!/usr/bin/env python3
"""
Nano-Link Phase 9 — Paper Hardening (Robustness Sweeps)
======================================================
This script performs rigorous sweeps to prove that Weight-Derived 
Parameter-Free Routing (NormGate) is not a brittle artifact.

Sweeps:
1. Load-Balancing Alpha (0.0 to 0.5)
2. Optimizer (AdamW vs SGD)
3. Weight Decay (0.0, 0.01, 0.1) -> Critical, since routing uses L2 norm!
4. Dataset Diversification (Structured Sequences vs Noisy Markov proxy)
"""

import os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase9")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──
VOCAB_SIZE = 128
D_MODEL = 64
HIDDEN_DIM = 128
N_EXPERTS = 16
TOP_K = 2
SEQ_LEN = 32
BATCH_SIZE = 32
N_STEPS = 600
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Models ──
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_MODEL, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, D_MODEL))
    def forward(self, x): return self.net(x)

class StandardRouter(nn.Module):
    NAME = "StandardGate"
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, N_EXPERTS, bias=False)
    def forward(self, x, experts): return self.gate(x)

class NormRouter(nn.Module):
    NAME = "NormGate"
    def __init__(self): super().__init__()
    def forward(self, x, experts):
        sigs = torch.stack([torch.norm(exp.net[0].weight, p=2, dim=0) for exp in experts], dim=0)
        return torch.abs(x) @ sigs.T

class MoELayer(nn.Module):
    def __init__(self, RouterClass):
        super().__init__()
        self.n_experts = N_EXPERTS
        self.top_k = TOP_K
        self.experts = nn.ModuleList([Expert() for _ in range(N_EXPERTS)])
        self.router = RouterClass()
        self.register_buffer("expert_load", torch.zeros(N_EXPERTS))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, D_MODEL)
        
        router_logits = self.router(x_flat, self.experts)
        if self.training: router_logits = router_logits + torch.randn_like(router_logits) * 0.1
            
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=self.n_experts).float()
        
        f_i = expert_mask.mean(dim=0)
        P_i = routing_weights.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(f_i * P_i)
        
        if self.training:
            with torch.no_grad(): self.expert_load += expert_mask.sum(dim=0)
            
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            any_mask = mask.any(dim=-1)
            if any_mask.any():
                out[any_mask] += expert(x_flat[any_mask]) * top_k_weights[any_mask, mask[any_mask].float().argmax(dim=-1)].unsqueeze(-1)
                
        return out.view(batch_size, seq_len, D_MODEL), aux_loss

class DummyTransformer(nn.Module):
    def __init__(self, RouterClass):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.attn1 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ffn1 = nn.Sequential(nn.Linear(D_MODEL, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, D_MODEL))
        self.norm1a, self.norm1b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn2 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.moe = MoELayer(RouterClass)
        self.norm2a, self.norm2b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x2, _ = self.attn1(self.norm1a(x), self.norm1a(x), self.norm1a(x))
        x = x + self.ffn1(self.norm1b(x + x2))
        x2, _ = self.attn2(self.norm2a(x), self.norm2a(x), self.norm2a(x))
        moe_out, aux_loss = self.moe(self.norm2b(x + x2))
        x = x + moe_out
        return self.head(x), aux_loss

# ── Datasets ──
def get_dataset_A(num_samples=500):
    """Structured repeating sequences (Phase 6/7/8 style)."""
    torch.manual_seed(42)
    templates = torch.randint(0, VOCAB_SIZE, (5, SEQ_LEN))
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 5, (1,)).item()].clone()
        base[torch.randperm(SEQ_LEN)[:2]] = torch.randint(0, VOCAB_SIZE, (2,))
        data.append(base)
    return torch.stack(data)

def get_dataset_B(num_samples=500):
    """Noisy Markov Chain (different data distribution)."""
    torch.manual_seed(100)
    transition = torch.rand(VOCAB_SIZE, VOCAB_SIZE)
    transition = transition / transition.sum(dim=1, keepdim=True)
    data = []
    for _ in range(num_samples):
        seq = [torch.randint(0, VOCAB_SIZE, (1,)).item()]
        for _ in range(SEQ_LEN - 1):
            seq.append(torch.multinomial(transition[seq[-1]], 1).item())
        data.append(torch.tensor(seq))
    return torch.stack(data)

# ── Training Routine ──
def compute_gini(load_array):
    if sum(load_array) == 0: return 1.0
    la = np.sort(np.asarray(load_array))
    n = len(la)
    return float((2.0 * np.sum((np.arange(1, n+1) * la)) / (n * la.sum())) - (n+1)/n)

def train_run(RouterClass, data, alpha_aux, opt_type, wd, lr=1e-3):
    torch.manual_seed(1337)
    model = DummyTransformer(RouterClass).to(device)
    
    if opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr*10, weight_decay=wd, momentum=0.9) # SGD needs higher LR
        
    dataset_size = data.size(0)
    model.train()
    loss_history = []
    
    for step in range(N_STEPS):
        batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(device)
        logits, aux_loss = model(batch[:, :-1])
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        loss = task_loss + alpha_aux * aux_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(task_loss.item())
        
    final_loss = np.mean(loss_history[-20:])
    gini = compute_gini(model.moe.expert_load.cpu().numpy())
    return final_loss, gini

# ── Sweeps ──
def run_alpha_sweep(data, dataset_name):
    print(f"\n--- 1. Alpha Balancing Sweep ({dataset_name}) ---")
    alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = {'alpha': alphas, 'Standard_Loss': [], 'Standard_Gini': [], 'Norm_Loss': [], 'Norm_Gini': []}
    
    print(f"{'Alpha':<10} | {'Std Loss':<10} | {'Std Gini':<10} | {'Norm Loss':<10} | {'Norm Gini':<10}")
    print("-" * 60)
    for a in alphas:
        sl, sg = train_run(StandardRouter, data, a, 'AdamW', 0.0)
        nl, ng = train_run(NormRouter, data, a, 'AdamW', 0.0)
        results['Standard_Loss'].append(sl); results['Standard_Gini'].append(sg)
        results['Norm_Loss'].append(nl); results['Norm_Gini'].append(ng)
        print(f"{a:<10} | {sl:<10.3f} | {sg:<10.3f} | {nl:<10.3f} | {ng:<10.3f}")
        
    return results

def run_optimizer_sweep(data):
    print("\n--- 2. Optimizer & Weight Decay Sweep (Dataset A, alpha=0.1) ---")
    configs = [
        ('AdamW', 0.0), ('AdamW', 0.01), ('AdamW', 0.1),
        ('SGD', 0.0), ('SGD', 0.01)
    ]
    print(f"{'Opt':<8} | {'WD':<6} | {'Std Loss':<10} | {'Norm Loss':<10} | {'Norm Gini':<10}")
    print("-" * 55)
    
    # Store for plotting
    norm_losses = np.zeros((2, 3)) # rows: AdamW, SGD; cols: WD = 0, 0.01, 0.1
    norm_losses[:] = np.nan
    
    for idx, (opt, wd) in enumerate(configs):
        sl, _ = train_run(StandardRouter, data, 0.1, opt, wd)
        nl, ng = train_run(NormRouter, data, 0.1, opt, wd)
        print(f"{opt:<8} | {wd:<6} | {sl:<10.3f} | {nl:<10.3f} | {ng:<10.3f}")
        
        r = 0 if opt == 'AdamW' else 1
        c = 0 if wd == 0.0 else (1 if wd == 0.01 else 2)
        norm_losses[r, c] = nl

    # Plot Optimizer Heatmap
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(norm_losses, annot=True, fmt=".3f", cmap="YlGnBu", 
                     xticklabels=["0.0", "0.01", "0.1"], yticklabels=["AdamW", "SGD"])
    plt.title("NormGate Task Loss by Optimizer & Weight Decay")
    plt.xlabel("Weight Decay"); plt.ylabel("Optimizer")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_opt_wd.png"), dpi=150)
    plt.close()

def plot_alpha_sweep(res_A, res_B):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dataset A
    axes[0].plot(res_A['alpha'], res_A['Standard_Loss'], 'o-', c='#e94560', label="StandardGate")
    axes[0].plot(res_A['alpha'], res_A['Norm_Loss'], 's-', c='#16c79a', label="NormGate")
    axes[0].set_title("Dataset A (Structured)")
    axes[0].set_xlabel(r"Balancing $\alpha$"); axes[0].set_ylabel("Task Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    
    # Dataset B
    axes[1].plot(res_B['alpha'], res_B['Standard_Loss'], 'o-', c='#e94560', label="StandardGate")
    axes[1].plot(res_B['alpha'], res_B['Norm_Loss'], 's-', c='#16c79a', label="NormGate")
    axes[1].set_title("Dataset B (Markov Noise)")
    axes[1].set_xlabel(r"Balancing $\alpha$"); axes[1].set_ylabel("Task Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    
    fig.suptitle("Task Loss Robustness vs Load-Balancing Strength", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_alpha_loss.png"), dpi=150)
    plt.close()
    
    # Gini plot
    plt.figure(figsize=(6, 5))
    plt.plot(res_A['alpha'], res_A['Standard_Gini'], 'o-', c='#e94560', label="StandardGate")
    plt.plot(res_A['alpha'], res_A['Norm_Gini'], 's-', c='#16c79a', label="NormGate")
    plt.title("Gini Coefficient vs Load-Balancing Strength (Dataset A)")
    plt.xlabel(r"Balancing $\alpha$"); plt.ylabel("Gini (Lower = Better Balanced)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_alpha_gini.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    print("=" * 80)
    print("  NANO-LINK PHASE 9 — ROBUSTNESS SWEEPS")
    print("=" * 80)
    
    data_A = get_dataset_A(500)
    data_B = get_dataset_B(500)
    
    res_A = run_alpha_sweep(data_A, "Dataset A")
    res_B = run_alpha_sweep(data_B, "Dataset B")
    
    run_optimizer_sweep(data_A)
    plot_alpha_sweep(res_A, res_B)
    
    print("\n  Sweeps complete. Check results_phase9/ for heatmaps and charts.")

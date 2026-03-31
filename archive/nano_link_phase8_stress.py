#!/usr/bin/env python3
"""
Nano-Link Phase 8 — Theoretical Stress Tests for Weight-Signature Routing
=======================================================================
4 targeted tests to scientifically prove that the L2-norm of expert weights
acts as an implicit routing signal emerging from gradient accumulation dynamics.

Test 1: Correlation Test (Norm vs Token Count)
Test 2: Perturbation Test (Shuffled Norms, Normalized Signatures)
Test 3: Evolution Test (Norm divergence & Routing Entropy over time)
Test 4: Robustness to Dimension (Scaling to 8, 16, 32, 64 experts)
"""

import os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase8")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparams ──
VOCAB_SIZE = 256
D_MODEL = 64
HIDDEN_DIM = 128
SEQ_LEN = 32
BATCH_SIZE = 32
LR = 1e-3
AUX_LOSS_WEIGHT = 0.1

class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
    def forward(self, x): return self.net(x)

class NormRouter(nn.Module):
    NAME = "NormGate"
    def __init__(self, n_experts):
        super().__init__()
        self.perturbation_mode = "none" # "none", "shuffled", "normalized"
        self.shuffled_indices = None
        
    def forward(self, x, experts):
        sigs = [torch.norm(exp.net[0].weight, p=2, dim=0) for exp in experts]
        sigs = torch.stack(sigs, dim=0) # (E, D)
        
        if self.perturbation_mode == "shuffled":
            if self.shuffled_indices is None:
                self.shuffled_indices = torch.randperm(len(experts))
            sigs = sigs[self.shuffled_indices]
        elif self.perturbation_mode == "normalized":
            # Normalize each expert's signature so no expert has a higher magnitude advantage
            # This tests if absolute magnitude is the driving factor.
            sigs = sigs / (torch.norm(sigs, p=2, dim=1, keepdim=True) + 1e-9)
            
        return torch.abs(x) @ sigs.T

class MoELayer(nn.Module):
    def __init__(self, n_experts, top_k):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(D_MODEL, HIDDEN_DIM) for _ in range(n_experts)])
        self.router = NormRouter(n_experts)
        self.register_buffer("expert_load", torch.zeros(n_experts))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, D_MODEL)
        
        router_logits = self.router(x_flat, self.experts)
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.1
            
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=self.n_experts).float()
        f_i = expert_mask.mean(dim=0)
        P_i = routing_weights.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(f_i * P_i)
        
        avg_prob = routing_weights.mean(dim=0)
        batch_routing_entropy = -torch.sum(avg_prob * torch.log(avg_prob + 1e-9))
        
        if self.training:
            with torch.no_grad():
                self.expert_load += expert_mask.sum(dim=0)
            
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            any_mask = mask.any(dim=-1)
            if any_mask.any():
                out[any_mask] += expert(x_flat[any_mask]) * top_k_weights[any_mask, mask[any_mask].float().argmax(dim=-1)].unsqueeze(-1)
                
        return out.view(batch_size, seq_len, D_MODEL), aux_loss, batch_routing_entropy

class DummyTransformer(nn.Module):
    def __init__(self, n_experts, top_k=2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        
        self.attn1 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ffn1 = nn.Sequential(nn.Linear(D_MODEL, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, D_MODEL))
        self.norm1a, self.norm1b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)

        self.attn2 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.moe = MoELayer(n_experts, top_k)
        self.norm2a, self.norm2b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x2, _ = self.attn1(self.norm1a(x), self.norm1a(x), self.norm1a(x))
        x = x + self.ffn1(self.norm1b(x + x2))
        
        x2, _ = self.attn2(self.norm2a(x), self.norm2a(x), self.norm2a(x))
        moe_out, aux_loss, entropy = self.moe(self.norm2b(x + x2))
        x = x + moe_out
        return self.head(x), aux_loss, entropy

def generate_data(num_samples=500, seed=42):
    torch.manual_seed(seed)
    templates = torch.randint(0, VOCAB_SIZE, (5, SEQ_LEN))
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 5, (1,)).item()].clone()
        noise_idx = torch.randperm(SEQ_LEN)[:2]
        base[noise_idx] = torch.randint(0, VOCAB_SIZE, (2,))
        data.append(base)
    return torch.stack(data)

def compute_gini(load_array):
    if sum(load_array) == 0: return 1.0
    la = np.sort(np.asarray(load_array))
    n = len(la)
    return float( (2.0 * np.sum((np.arange(1, n+1) * la)) / (n * la.sum())) - (n+1)/n )


def run_test_1_and_3(data, device):
    """Correlation Test & Evolution Test."""
    print("\n--- Running Test 1 (Correlation) & Test 3 (Evolution) ---")
    model = DummyTransformer(n_experts=16, top_k=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    norms_history = []
    entropy_history = []
    dataset_size = data.size(0)
    
    model.train()
    for step in range(800):
        with torch.no_grad():
            current_norms = [torch.norm(exp.net[0].weight, p=2).item() for exp in model.moe.experts]
            norms_history.append(current_norms)
            
        batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(device)
        logits, aux_loss, entropy = model(batch[:, :-1])
        
        entropy_history.append(entropy.item())
        
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1)) + AUX_LOSS_WEIGHT * aux_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Calculate Test 1 Correlation
    final_norms = np.array(norms_history[-1])
    token_counts = model.moe.expert_load.cpu().numpy()
    
    if np.std(final_norms) > 0 and np.std(token_counts) > 0:
        corr, _ = pearsonr(final_norms, token_counts)
    else:
        corr = 0.0
        
    print(f"  Pearson Correlation (Final Norm vs Token Count): {corr:.3f}")

    # Plot Test 1
    plt.figure(figsize=(6, 5))
    plt.scatter(final_norms, token_counts, c='#16c79a', s=100, alpha=0.7)
    
    # Line of best fit
    m, b = np.polyfit(final_norms, token_counts, 1)
    plt.plot(final_norms, m*final_norms + b, color='#f5a623', linestyle='--', alpha=0.8)
    
    plt.title(f"Test 1: Expert Utility vs Weight Magnitude\nPearson Correlation: {corr:.3f}", color='white', fontweight='bold')
    plt.xlabel("Expert Weight L2-Norm", color='white')
    plt.ylabel("Tokens Routed (Utility)", color='white')
    plt.grid(alpha=0.3)
    plt.gca().set_facecolor('#16213e')
    plt.gcf().set_facecolor('#1a1a2e')
    plt.gca().tick_params(colors='white')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "test1_correlation.png"), dpi=150)
    plt.close()

    # Plot Test 3
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    norms_arr = np.array(norms_history)
    for i in range(16):
        plt.plot(norms_arr[:, i], alpha=0.6, linewidth=1.5)
    plt.title("Test 3: Expert Norm Evolution", color='white', fontweight='bold')
    plt.xlabel("Training Step", color='white'); plt.ylabel("L2 Norm magnitude", color='white')
    plt.grid(alpha=0.3)
    plt.gca().set_facecolor('#16213e')
    plt.gca().tick_params(colors='white')
    
    plt.subplot(1, 2, 2)
    smoothed_ent = np.convolve(entropy_history, np.ones(20)/20, mode='valid')
    plt.plot(smoothed_ent, c='#e94560', linewidth=2)
    plt.title("Routing Entropy (Lower = More Specialization)", color='white', fontweight='bold')
    plt.xlabel("Training Step", color='white'); plt.ylabel("Routing Entropy H(p)", color='white')
    plt.grid(alpha=0.3)
    plt.gca().set_facecolor('#16213e')
    plt.gca().tick_params(colors='white')
    
    plt.gcf().set_facecolor('#1a1a2e')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "test3_evolution.png"), dpi=150)
    plt.close()
    
    return model


def run_test_2(trained_model, data, device):
    """Perturbation Test."""
    print("\n--- Running Test 2 (Perturbation) ---")
    trained_model.eval()
    
    modes = ["none", "shuffled", "normalized"]
    results = {}
    
    with torch.no_grad():
        for mode in modes:
            trained_model.moe.router.perturbation_mode = mode
            trained_model.moe.router.shuffled_indices = None
            
            total_loss = 0
            for i in range(20): # 20 evaluate batches
                batch = data[torch.randint(0, data.size(0), (BATCH_SIZE,))].to(device)
                logits, _, _ = trained_model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
                total_loss += loss.item()
            
            results[mode] = total_loss / 20
            print(f"  Mode: {mode:12} | Validation Task Loss: {results[mode]:.4f}")
            
    return results


def run_test_4(data, device):
    """Scaling Test."""
    print("\n--- Running Test 4 (Scaling Robustness) ---")
    expert_counts = [8, 16, 32, 64]
    
    print(f"{'EXPERTS':<8} | {'FINAL LOSS':<10} | {'GINI COEFF':<10}")
    print("-" * 35)
    
    for nx in expert_counts:
        model = DummyTransformer(n_experts=nx, top_k=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        model.train()
        loss_hist = []
        for step in range(400): 
            batch = data[torch.randint(0, data.size(0), (BATCH_SIZE,))].to(device)
            logits, aux_loss, _ = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1)) + AUX_LOSS_WEIGHT * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            
        final_loss = np.mean(loss_hist[-20:])
        raw_load = model.moe.expert_load.cpu().numpy()
        gini = compute_gini(raw_load)
        
        print(f"{nx:<8} | {final_loss:<10.3f} | {gini:<10.3f}")

if __name__ == "__main__":
    print("=" * 80)
    print("  NANO-LINK PHASE 8 — STRESS TESTS FOR WEIGHT-SIGNATURE ROUTING")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}\n")
    
    data = generate_data(500)
    
    trained_model = run_test_1_and_3(data, device)
    run_test_2(trained_model, data, device)
    run_test_4(data, device)
    print("\n  Stress testing complete. Check results_phase8/ charts.")

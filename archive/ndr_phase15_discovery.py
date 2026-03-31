#!/usr/bin/env python3
"""
Norm-Driven Routing (NDR) Phase 15 — Auto-Discovery & Quantization
==================================================================
Demonstrates how "Implicit Control from Parameter Statistics" allows 
a network to natively Self-Discover its optimal internal architecture.

We train a large Dense NDR Transformer on a complex dataset.
We track the activation Load mapped to the L2 Norms of each block.
After training, we use these physically accumulated statistics to automatically:
1. Prune dead blocks (0-bit).
2. Assign Adaptive Quantization (8-bit to critical structural blocks, 
   4-bit/2-bit to edge feature blocks).
   
This simulates building an adaptive sparse architecture without 
explicit gradient-based search or secondary pruning algorithms.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase15")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──
VOCAB_SIZE = 128
D_MODEL = 64
HIDDEN_DIM = 1024
N_BLOCKS = 32
TOP_K = 4   # Activate 4 out of 32 blocks (12.5% Sparsity)
BLOCK_SIZE = HIDDEN_DIM // N_BLOCKS
SEQ_LEN = 32
BATCH_SIZE = 64
N_STEPS = 500
LR = 2e-3
AUX_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Complex Noisy Dataset ──
def get_complex_dataset(num_samples=800):
    torch.manual_seed(1234)
    # 20 different underlying patterns
    templates = torch.randint(0, VOCAB_SIZE, (20, SEQ_LEN))
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 20, (1,)).item()].clone()
        # Add high temporal noise (mutate 8 tokens)
        base[torch.randperm(SEQ_LEN)[:8]] = torch.randint(0, VOCAB_SIZE, (8,))
        data.append(base)
    return torch.stack(data)

# ── Architectures ──
class AutoDiscoveryFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_in = nn.Linear(D_MODEL, HIDDEN_DIM)
        self.w_out = nn.Linear(HIDDEN_DIM, D_MODEL)
        
        # Tracking infrastructure
        self.register_buffer("block_load", torch.zeros(N_BLOCKS))
        self.register_buffer("block_norm_history", torch.zeros(N_STEPS, N_BLOCKS))

    def forward(self, x, step_idx=None):
        batch, seq, _ = x.shape
        x_flat = x.view(-1, D_MODEL)
        
        w_view = self.w_in.weight.view(N_BLOCKS, BLOCK_SIZE, D_MODEL)
        sigs = torch.norm(w_view, p=2, dim=(1, 2))
        
        # Track norms over time if step is provided
        if step_idx is not None and step_idx < N_STEPS:
            with torch.no_grad():
                self.block_norm_history[step_idx] = sigs.detach()
        
        # sigs is (N_BLOCKS,), router_logits becomes (B*S, N_BLOCKS)
        router_logits = torch.abs(x_flat).mean(dim=-1, keepdim=True) @ sigs.unsqueeze(0)
        if self.training: 
            # Inject noise to encourage exploration of blocks
            router_logits = router_logits + torch.randn_like(router_logits) * 0.2
            
        routing_weights = F.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, TOP_K, dim=-1)
        
        active_blocks_mask = torch.zeros(x_flat.size(0), N_BLOCKS, device=x.device)
        active_blocks_mask.scatter_(1, top_k_indices, 1.0)
        neuron_mask = active_blocks_mask.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(-1, HIDDEN_DIM)
        
        sparse_act = F.gelu(self.w_in(x_flat)) * neuron_mask
        out = self.w_out(sparse_act)
        
        expert_mask = F.one_hot(top_k_indices[:, 0], num_classes=N_BLOCKS).float()
        aux_loss = N_BLOCKS * torch.sum(expert_mask.mean(dim=0) * routing_weights.mean(dim=0))
        
        if self.training:
            with torch.no_grad():
                self.block_load += expert_mask.sum(dim=0)
                
        return out.view(batch, seq, D_MODEL), aux_loss

class NDRDiscoveryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(D_MODEL)
        
        self.ffn = AutoDiscoveryFFN()
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x, step_idx=None):
        x_emb = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x2, _ = self.attn(self.norm1(x_emb), self.norm1(x_emb), self.norm1(x_emb))
        x = x_emb + x2
        sparse_out, aux_loss = self.ffn(self.norm2(x), step_idx)
        x = x + sparse_out
        return self.head(x), aux_loss

# ── Auto-Discovery Logic ──
def discover_architecture(norms, loads):
    print("\n--- Structural Analysis & Re-Architecture Phase ---")
    
    # Normalize metrics for scoring
    # We combine L2 Norm (functional magnitude) and Load (temporal utility)
    n_norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    n_loads = (loads - loads.min()) / (loads.max() - loads.min() + 1e-8)
    
    # Composite Utility Score
    utility_score = (n_norms * 0.5) + (n_loads * 0.5)
    
    # Quantization Thresholds (Percentiles)
    p_prune = np.percentile(utility_score, 15)  # Bottom 15% -> Pruned (0-bit)
    p_2bit  = np.percentile(utility_score, 40)  # 15%-40% -> 2-bit
    p_4bit  = np.percentile(utility_score, 75)  # 40%-75% -> 4-bit
    # Top 25% -> 8-bit
    
    arch_map = []
    bit_weights = []
    
    for i, score in enumerate(utility_score):
        if score < p_prune:
            arch_map.append("Pruned")
            bit_weights.append(0)
        elif score < p_2bit:
            arch_map.append("2-bit ")
            bit_weights.append(2)
        elif score < p_4bit:
            arch_map.append("4-bit ")
            bit_weights.append(4)
        else:
            arch_map.append("8-bit*")
            bit_weights.append(8)
            
    # Print Architectural Summary
    print(f"{'Block':<7}| {'Norm L2':<10}| {'Token Load':<12}| {'Utility':<10}| {'Target Layer'}")
    print("-" * 57)
    for i in range(N_BLOCKS):
        print(f"#{i:<5} | {norms[i]:<10.2f}| {int(loads[i]):<12}| {utility_score[i]:<10.3f}| {arch_map[i]}")

    # Memory Calculation
    orig_memory = N_BLOCKS * BLOCK_SIZE * D_MODEL * 16 # Assuming fp16 base
    new_memory = sum(bits * BLOCK_SIZE * D_MODEL for bits in bit_weights)
    compression = 100 * (1 - (new_memory / orig_memory))
    print(f"\n[ Discovery Outcome: {compression:.1f}% Compression discovered via Parameter Statistics ]")
    
    return utility_score, bit_weights

# ── Execution ──
def run_experiment():
    print("=" * 70)
    print("  PHASE 15: AUTOMATIC ARCHITECTURE DISCOVERY")
    print(f"  Configuration: {N_BLOCKS} Blocks, D_HIDDEN={HIDDEN_DIM}. Target: Self-Quantization.")
    print("=" * 70)
    
    model = NDRDiscoveryTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    data = get_complex_dataset()
    dataset_size = data.size(0)
    
    for step in range(N_STEPS):
        batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(DEVICE)
        model.train()
        logits, aux_loss = model(batch[:, :-1], step_idx=step)
        
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        loss = task_loss + AUX_LOSS_WEIGHT * aux_loss
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 100 == 0:
            print(f"  Step {step:<4} | Loss: {task_loss.item():.4f}")

    # Plot Norm Evolution to prove physical structuring
    history = model.ffn.block_norm_history.cpu().numpy()
    plt.figure(figsize=(10, 6))
    for i in range(N_BLOCKS):
        plt.plot(history[:, i], alpha=0.5, linewidth=1)
    plt.title("Parameter Evolution: Divergence of Block Norms")
    plt.xlabel("Training Steps")
    plt.ylabel("Block L2 Norm")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "norm_divergence.png"))
    plt.close()
    
    # Run Auto-Discovery based on final settled statistics
    w_view = model.ffn.w_in.weight.view(N_BLOCKS, BLOCK_SIZE, D_MODEL)
    final_norms = torch.norm(w_view, p=2, dim=(1, 2)).detach().cpu().numpy()
    final_loads = model.ffn.block_load.cpu().numpy()
    
    discover_architecture(final_norms, final_loads)

if __name__ == "__main__":
    run_experiment()

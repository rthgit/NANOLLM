#!/usr/bin/env python3
"""
Norm-Driven Routing (NDR) Phase 12 — Final Block Scaling Sweep
==============================================================
Validates "Implicit Control from Parameter Statistics" by sweeping 
the number of dynamic blocks N in a Dense FFN while maintaining 
a constant 25% active compute ratio.

Test cases (N_BLOCKS -> TOP_K):
- N = 8   -> Top 2
- N = 16  -> Top 4
- N = 32  -> Top 8
- N = 64  -> Top 16
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase12")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Base Hyperparameters ──
VOCAB_SIZE = 128
D_MODEL = 64
HIDDEN_DIM = 512
SEQ_LEN = 32
BATCH_SIZE = 32
N_STEPS = 500
LR = 1e-3
AUX_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──
def get_dataset(num_samples=500):
    torch.manual_seed(42)
    templates = torch.randint(0, VOCAB_SIZE, (5, SEQ_LEN))
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 5, (1,)).item()].clone()
        base[torch.randperm(SEQ_LEN)[:2]] = torch.randint(0, VOCAB_SIZE, (2,))
        data.append(base)
    return torch.stack(data)

# ── Block-Sparse Architecture ──
class NDRBlockSparseFFN(nn.Module):
    def __init__(self, n_blocks, top_k):
        super().__init__()
        self.n_blocks = n_blocks
        self.top_k = top_k
        self.block_size = HIDDEN_DIM // n_blocks
        
        self.w_in = nn.Linear(D_MODEL, HIDDEN_DIM)
        self.w_out = nn.Linear(HIDDEN_DIM, D_MODEL)
        self.register_buffer("block_load", torch.zeros(n_blocks))
        
    def forward(self, x):
        batch, seq, d = x.shape
        x_flat = x.view(-1, D_MODEL)
        
        # Norm-Driven Signature extraction
        w_view = self.w_in.weight.view(self.n_blocks, self.block_size, D_MODEL)
        sigs = torch.norm(w_view, p=2, dim=1) # (n_blocks, D_MODEL)
        
        # NDR Score Compute
        router_logits = torch.abs(x_flat) @ sigs.T
        if self.training: router_logits = router_logits + torch.randn_like(router_logits) * 0.1
            
        routing_weights = F.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Mask Generation
        active_blocks_mask = torch.zeros(x_flat.size(0), self.n_blocks, device=x.device)
        active_blocks_mask.scatter_(1, top_k_indices, 1.0)
        neuron_mask = active_blocks_mask.unsqueeze(-1).expand(-1, -1, self.block_size).reshape(-1, HIDDEN_DIM)
        
        sparse_act = F.gelu(self.w_in(x_flat)) * neuron_mask
        
        # Aux Loss calculation
        expert_mask = F.one_hot(top_k_indices[:, 0], num_classes=self.n_blocks).float()
        aux_loss = self.n_blocks * torch.sum(expert_mask.mean(dim=0) * routing_weights.mean(dim=0))
        
        if self.training:
            with torch.no_grad(): self.block_load += expert_mask.sum(dim=0)
                
        out = self.w_out(sparse_act)
        return out.view(batch, seq, D_MODEL), aux_loss

class NDRTransformer(nn.Module):
    def __init__(self, n_blocks, top_k):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ffn = NDRBlockSparseFFN(n_blocks, top_k)
        self.norm1, self.norm2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        sparse_out, aux_loss = self.ffn(self.norm2(x))
        x = x + sparse_out
        return self.head(x), aux_loss

# ── Training Loop ──
def compute_gini(load_array):
    if sum(load_array) == 0: return 1.0
    la = np.sort(np.asarray(load_array))
    n = len(la)
    return float((2.0 * np.sum((np.arange(1, n+1) * la)) / (n * la.sum())) - (n+1)/n)

def run_sweep():
    print("=" * 70)
    print("  PHASE 12: NDR BLOCK SCALING SWEEP (D_HIDDEN=512, 25% Compute)")
    print("=" * 70)
    
    data = get_dataset()
    dataset_size = data.size(0)
    
    sweep_configs = [
        (8, 2),
        (16, 4),
        (32, 8),
        (64, 16)
    ]
    
    print(f"{'Blocks (N)':<12} | {'Active (K)':<12} | {'Final Loss':<12} | {'Gini Coeff':<12}")
    print("-" * 55)
    
    for n_blocks, top_k in sweep_configs:
        torch.manual_seed(1337)
        model = NDRTransformer(n_blocks, top_k).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        
        loss_hist = []
        for step in range(N_STEPS):
            batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(DEVICE)
            logits, aux_loss = model(batch[:, :-1])
            task_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
            loss = task_loss + AUX_LOSS_WEIGHT * aux_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_hist.append(task_loss.item())
            
        final_loss = np.mean(loss_hist[-20:])
        gini = compute_gini(model.ffn.block_load.cpu().numpy())
        
        print(f"{n_blocks:<12} | {top_k:<12} | {final_loss:<12.4f} | {gini:<12.3f}")

if __name__ == "__main__":
    run_sweep()

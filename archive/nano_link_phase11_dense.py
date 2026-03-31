#!/usr/bin/env python3
"""
Nano-Link Phase 11 — Generalization (Dense Block-Sparse MLPs)
=============================================================
Can Weight-Signature Routing transform a monolithic Dense Transformer 
into a dynamically sparse one without explicit experts?

We take a wide FFN, partition its hidden neurons into N blocks, 
extract the L2-Norm signature for each block directly from the FFN weights, 
and dynamically route tokens to only the Top-K blocks.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase11")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──
VOCAB_SIZE = 128
D_MODEL = 64
HIDDEN_DIM = 512
N_BLOCKS = 16
TOP_K = 4
BLOCK_SIZE = HIDDEN_DIM // N_BLOCKS
SEQ_LEN = 32
BATCH_SIZE = 32
N_STEPS = 600
LR = 1e-3
AUX_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Datasets ──
def get_dataset(num_samples=500):
    torch.manual_seed(42)
    templates = torch.randint(0, VOCAB_SIZE, (5, SEQ_LEN))
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 5, (1,)).item()].clone()
        base[torch.randperm(SEQ_LEN)[:2]] = torch.randint(0, VOCAB_SIZE, (2,))
        data.append(base)
    return torch.stack(data)

# ── Baselines ──
class StandardDenseTransformer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.attn1 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ffn1 = nn.Sequential(nn.Linear(D_MODEL, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, D_MODEL))
        self.norm1a, self.norm1b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn2 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ffn2 = nn.Sequential(nn.Linear(D_MODEL, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, D_MODEL))
        self.norm2a, self.norm2b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x2, _ = self.attn1(self.norm1a(x), self.norm1a(x), self.norm1a(x))
        x = x + self.ffn1(self.norm1b(x + x2))
        x2, _ = self.attn2(self.norm2a(x), self.norm2a(x), self.norm2a(x))
        x = x + self.ffn2(self.norm2b(x + x2))
        # Zero aux loss for standard model
        return self.head(x), torch.tensor(0.0, device=x.device)

# ── Block-Sparse Architecture ──
class BlockSparseFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_in = nn.Linear(D_MODEL, HIDDEN_DIM)
        self.w_out = nn.Linear(HIDDEN_DIM, D_MODEL)
        self.register_buffer("block_load", torch.zeros(N_BLOCKS))
        
    def forward(self, x):
        batch, seq, d = x.shape
        x_flat = x.view(-1, D_MODEL)
        
        # 1. Extract block signatures dynamically from w_in weights
        # w_in.weight is (HIDDEN_DIM, D_MODEL)
        w_view = self.w_in.weight.view(N_BLOCKS, BLOCK_SIZE, D_MODEL)
        sigs = torch.norm(w_view, p=2, dim=1) # Shape: (N_BLOCKS, D_MODEL)
        
        # 2. Compute Routing Scores (abs(x) @ sigs.T)
        router_logits = torch.abs(x_flat) @ sigs.T # (B*S, N_BLOCKS)
        
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.1
            
        routing_weights = F.softmax(router_logits, dim=-1) # (B*S, N_BLOCKS)
        top_k_weights, top_k_indices = torch.topk(routing_weights, TOP_K, dim=-1)
        
        # 3. Create Sparsity Mask for Neurons
        # top_k_indices: (B*S, TOP_K) -> which blocks are active
        active_blocks_mask = torch.zeros(x_flat.size(0), N_BLOCKS, device=x.device)
        active_blocks_mask.scatter_(1, top_k_indices, 1.0)
        
        # Expand block mask to neuron mask: (B*S, N_BLOCKS) -> (B*S, N_BLOCKS, BLOCK_SIZE)
        neuron_mask = active_blocks_mask.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(-1, HIDDEN_DIM)
        
        # 4. Apply Block Sparsity
        pre_act = self.w_in(x_flat)
        sparse_act = F.gelu(pre_act) * neuron_mask
        
        # 5. Routing Load Balancing Aux Loss
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=N_BLOCKS).float()
        f_i = expert_mask.mean(dim=0)
        P_i = routing_weights.mean(dim=0)
        aux_loss = N_BLOCKS * torch.sum(f_i * P_i)
        
        if self.training:
            with torch.no_grad():
                self.block_load += expert_mask.sum(dim=0)
                
        out = self.w_out(sparse_act)
        return out.view(batch, seq, D_MODEL), aux_loss

class BlockSparseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.attn1 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ffn1 = nn.Sequential(nn.Linear(D_MODEL, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, D_MODEL))
        self.norm1a, self.norm1b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        
        self.attn2 = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ffn2 = BlockSparseFFN()
        self.norm2a, self.norm2b = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x2, _ = self.attn1(self.norm1a(x), self.norm1a(x), self.norm1a(x))
        x = x + self.ffn1(self.norm1b(x + x2))
        
        x2, _ = self.attn2(self.norm2a(x), self.norm2a(x), self.norm2a(x))
        sparse_out, aux_loss = self.ffn2(self.norm2b(x + x2))
        x = x + sparse_out
        
        return self.head(x), aux_loss

# ── Training ──
def compute_gini(load_array):
    if sum(load_array) == 0: return 1.0
    la = np.sort(np.asarray(load_array))
    n = len(la)
    return float((2.0 * np.sum((np.arange(1, n+1) * la)) / (n * la.sum())) - (n+1)/n)

def run_experiment():
    print("=" * 70)
    print("  PHASE 11: GENERALIZATION TO DENSE BLOCK-SPARSE FFNs")
    print("=" * 70)
    print(f"  Configuration: D_HIDDEN={HIDDEN_DIM}, Blocks={N_BLOCKS}")
    print(f"  Active per token: {TOP_K} blocks ({TOP_K * BLOCK_SIZE} active neurons)")
    
    data = get_dataset(500)
    dataset_size = data.size(0)
    
    # 1. Train Massive Dense Baseline
    print("\n--- Training Massive Dense Baseline ---")
    model_dense = StandardDenseTransformer(HIDDEN_DIM).to(DEVICE)
    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=LR)
    
    loss_hist_dense = []
    
    for step in range(N_STEPS):
        batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(DEVICE)
        logits, _ = model_dense(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        
        opt_dense.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_dense.parameters(), 1.0)
        opt_dense.step()
        loss_hist_dense.append(loss.item())

    # 2. Train Block-Sparse FFN (Parameter-Free Routing)
    print("\n--- Training Block-Sparse FFN (Norm-Routed) ---")
    model_sparse = BlockSparseTransformer().to(DEVICE)
    opt_sparse = torch.optim.AdamW(model_sparse.parameters(), lr=LR)
    
    loss_hist_sparse = []
    
    for step in range(N_STEPS):
        batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(DEVICE)
        logits, aux_loss = model_sparse(batch[:, :-1])
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        loss = task_loss + AUX_LOSS_WEIGHT * aux_loss
        
        opt_sparse.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_sparse.parameters(), 1.0)
        opt_sparse.step()
        loss_hist_sparse.append(task_loss.item())
        
    # 3. Assess Parameter Match Baseline
    print("\n--- Training Small Dense Baseline (Active Params Match) ---")
    small_dim = TOP_K * BLOCK_SIZE
    model_small = StandardDenseTransformer(small_dim).to(DEVICE)
    opt_small = torch.optim.AdamW(model_small.parameters(), lr=LR)
    
    loss_hist_small = []
    for step in range(N_STEPS):
        batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(DEVICE)
        logits, _ = model_small(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        opt_small.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_small.parameters(), 1.0)
        opt_small.step()
        loss_hist_small.append(loss.item())
        
    print("\n[ FINAL RESULTS (Last 20 steps avg loss) ]")
    print(f"  Massive Dense FFN (100% FLOPs):    {np.mean(loss_hist_dense[-20:]):.4f}")
    print(f"  Block-Sparse FFN  ( {(TOP_K/N_BLOCKS)*100:.0f}% FLOPs):    {np.mean(loss_hist_sparse[-20:]):.4f}")
    print(f"  Small Dense FFN   ( {(TOP_K/N_BLOCKS)*100:.0f}% FLOPs):    {np.mean(loss_hist_small[-20:]):.4f}")
    
    gini = compute_gini(model_sparse.ffn2.block_load.cpu().numpy())
    print(f"  Block Routing Gini Coefficient:    {gini:.3f}")

if __name__ == "__main__":
    run_experiment()

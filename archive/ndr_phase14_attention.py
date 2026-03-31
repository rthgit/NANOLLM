#!/usr/bin/env python3
"""
Norm-Driven Routing (NDR) Phase 14 — Sparse Attention
=====================================================
Applies the "Implicit Control from Parameter Statistics" principle
to the Multi-Head Attention (MHA) module. 

Instead of executing all attention heads and linearly combining them, 
we extract a parameter-free routing signature from the attention projection 
matrices (e.g., $W_V$) and dynamically select only the Top-K heads per token.

This simulates Sparse-Attention without any dedicated gating networks,
cutting attention FLOPs drastically based purely on weight statistics.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase14")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──
VOCAB_SIZE = 128
D_MODEL = 128
N_HEADS = 16
TOP_K_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
SEQ_LEN = 32
BATCH_SIZE = 64
N_STEPS = 600
LR = 1.5e-3
AUX_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──
def get_dataset(num_samples=500):
    torch.manual_seed(42)
    templates = torch.randint(0, VOCAB_SIZE, (5, SEQ_LEN))
    data = []
    for _ in range(num_samples):
        base = templates[torch.randint(0, 5, (1,)).item()].clone()
        base[torch.randperm(SEQ_LEN)[:4]] = torch.randint(0, VOCAB_SIZE, (4,))
        data.append(base)
    return torch.stack(data)

# ── Architectures ──

class StandardMHA(nn.Module):
    """ Standard Dense Multi-Head Attention (100% FLOPs) """
    def __init__(self):
        super().__init__()
        # PyTorch MHA for the baseline
        self.mha = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out, torch.tensor(0.0, device=x.device)

class NormDrivenSparseMHA(nn.Module):
    """ Norm-Driven Sparse Attention (Top-K Heads per token) """
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)
        self.register_buffer("head_load", torch.zeros(N_HEADS))

    def forward(self, x):
        batch, seq, _ = x.shape
        x_flat = x.view(-1, D_MODEL) # (B*S, D)
        
        # 1. Routing via Parameter Statistics. 
        # We use the Value projection weights W_V to define the functional signature of a head.
        # w_v is shape (D_MODEL, D_MODEL)
        w_v_view = self.v_proj.weight.view(N_HEADS, HEAD_DIM, D_MODEL)
        sigs = torch.norm(w_v_view, p=2, dim=1) # (N_HEADS, D_MODEL)
        
        # 2. Score heads: how relevant is this token to each head's Value subspace?
        router_logits = torch.abs(x_flat) @ sigs.T # (B*S, N_HEADS)
        
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.1
            
        routing_weights = F.softmax(router_logits, dim=-1) # (B*S, N_HEADS)
        _, top_k_indices = torch.topk(routing_weights, TOP_K_HEADS, dim=-1)
        
        # 3. Create sparse mask for heads
        # (B*S, N_HEADS)
        active_heads_mask = torch.zeros(x_flat.size(0), N_HEADS, device=x.device)
        active_heads_mask.scatter_(1, top_k_indices, 1.0) 
        
        # 4. Standard Projections
        q = self.q_proj(x).view(batch, seq, N_HEADS, HEAD_DIM).transpose(1, 2) # (B, H, S, D_h)
        k = self.k_proj(x).view(batch, seq, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, N_HEADS, HEAD_DIM).transpose(1, 2)
        
        # 5. Scaled Dot-Product Attention 
        # In a real system, we'd ONLY compute QK^T for active heads via custom kernels (e.g., Triton).
        # Here we mask the output to simulate the physical sparsity.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        attn_probs = F.softmax(scores, dim=-1)
        
        # apply attention to V
        head_outputs = torch.matmul(attn_probs, v) # (B, H, S, D_h)
        
        # 6. Apply Norm-Driven Sparsity Mask 
        # Transform mask: (B*S, N_HEADS) -> (B, S, N_HEADS, 1) -> (B, N_HEADS, S, 1) to match head_outputs
        mask_reshaped = active_heads_mask.view(batch, seq, N_HEADS, 1).transpose(1, 2)
        sparse_head_outputs = head_outputs * mask_reshaped
        
        # Concat heads and project
        sparse_concat = sparse_head_outputs.transpose(1, 2).reshape(batch, seq, D_MODEL)
        out = self.o_proj(sparse_concat)
        
        # 7. Aux Loss
        expert_mask = F.one_hot(top_k_indices[:, 0], num_classes=N_HEADS).float()
        aux_loss = N_HEADS * torch.sum(expert_mask.mean(dim=0) * routing_weights.mean(dim=0))
        
        if self.training:
            with torch.no_grad():
                self.head_load += expert_mask.sum(dim=0)
                
        return out, aux_loss

class NDRTransformerAttention(nn.Module):
    def __init__(self, mha_module):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.mha = mha_module
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, D_MODEL*4), nn.GELU(), nn.Linear(D_MODEL*4, D_MODEL))
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        mha_out, aux_loss = self.mha(self.norm1(x))
        x = x + mha_out
        x = x + self.ffn(self.norm2(x))
        return self.head(x), aux_loss

# ── Training Loop ──
def compute_gini(load_array):
    if sum(load_array) == 0: return 1.0
    la = np.sort(np.asarray(load_array))
    n = len(la)
    return float((2.0 * np.sum((np.arange(1, n+1) * la)) / (n * la.sum())) - (n+1)/n)

def train(model, mode_name):
    print(f"\n--- Training {mode_name} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    data = get_dataset(500)
    dataset_size = data.size(0)
    
    loss_hist = []
    for step in range(N_STEPS):
        batch = data[torch.randint(0, dataset_size, (BATCH_SIZE,))].to(DEVICE)
        model.train()
        logits, aux_loss = model(batch[:, :-1])
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        
        loss = task_loss + AUX_LOSS_WEIGHT * aux_loss
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_hist.append(task_loss.item())
        
        if step % 150 == 0:
            print(f"  Step {step:<4} | Loss: {task_loss.item():.4f}")
            
    final_loss = np.mean(loss_hist[-20:])
    print(f"  Final Avg Loss: {final_loss:.4f}")
    return final_loss

def run_experiment():
    print("=" * 70)
    print(f"  PHASE 14: NORM-DRIVEN SPARSE ATTENTION")
    print(f"  Headers: {N_HEADS} total, Active K={TOP_K_HEADS} ({(TOP_K_HEADS/N_HEADS)*100:.0f}% MHA Compute)")
    print("=" * 70)
    
    # 1. Standard Dense Multi-Head Attention
    model_dense = NDRTransformerAttention(StandardMHA()).to(DEVICE)
    loss_dense = train(model_dense, "Standard Dense MHA (100% FLOPs)")
    
    # 2. NDR Sparse Attention
    mha_ndr = NormDrivenSparseMHA()
    model_ndr = NDRTransformerAttention(mha_ndr).to(DEVICE)
    loss_ndr = train(model_ndr, "Norm-Driven Sparse MHA (25% FLOPs)")
    
    print("\n[ FINAL RESULTS ]")
    print(f"  Standard Dense MHA: {loss_dense:.4f}")
    print(f"  NDR Sparse Heads  : {loss_ndr:.4f}")
    
    gini = compute_gini(mha_ndr.head_load.cpu().numpy())
    print(f"  NDR Head Gini     : {gini:.3f}")

if __name__ == "__main__":
    run_experiment()

#!/usr/bin/env python3
"""
Norm-Driven Routing (NDR) Phase 13 — Functional Specialization 
==============================================================
Validates that "Implicit Control from Parameter Statistics" genuinely 
induces modular feature selection by forcing the network to solve 
a strictly un-blendable Modulo-4 Multi-Task problem.

Task Map (based on token index % 4):
0 -> Copy    (prev_token)
1 -> Reverse (VOCAB_SIZE - 1 - prev_token)
2 -> Shift   ((prev_token + 10) % VOCAB_SIZE)
3 -> XOR     (prev_token ^ 42)

If NDR is functionally routing correctly, it will beat Dense/Random
and we will observe clear block specializations tied to the modulo.
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase13")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──
VOCAB_SIZE = 128
D_MODEL = 64
HIDDEN_DIM = 256
N_BLOCKS = 16
TOP_K = 4
BLOCK_SIZE = HIDDEN_DIM // N_BLOCKS
SEQ_LEN = 32
BATCH_SIZE = 64
N_STEPS = 1000
LR = 2e-3
AUX_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset (Modulo 4 Tasks) ──
def generate_multitask_batch(batch_size):
    # Features X are random tokens
    x = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    y = torch.zeros_like(x)
    
    # We predict y[t] from x[t-1]. To align nicely, we just say the network gets x[t], 
    # and has to output the transformed version of x[t] at the same timestep.
    # The 'modulo' condition relies purely on the token absolute position (t % 4).
    # We inject position embeddings so the network knows `t`.
    
    for t in range(SEQ_LEN):
        mod = t % 4
        if mod == 0:   y[:, t] = x[:, t]                                # Copy
        elif mod == 1: y[:, t] = VOCAB_SIZE - 1 - x[:, t]               # Reverse
        elif mod == 2: y[:, t] = (x[:, t] + 10) % VOCAB_SIZE            # Shift
        elif mod == 3: y[:, t] = x[:, t] ^ 42                           # XOR
        
    return x.to(DEVICE), y.to(DEVICE)

# ── Architectures ──
class DenseTransformer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        # We use a purely point-wise network (1 layer) to test FFN memory isolation!
        # No attention needed because the task is position and current-token dependent.
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, D_MODEL))
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x = x + self.ffn(self.norm(x))
        return self.head(x), torch.tensor(0.0, device=x.device)

class NDRTransformer(nn.Module):
    def __init__(self, router_type="norm"):   # "norm" or "random"
        super().__init__()
        self.router_type = router_type
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        
        # Block Sparse FFN
        self.w_in = nn.Linear(D_MODEL, HIDDEN_DIM)
        self.w_out = nn.Linear(HIDDEN_DIM, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)
        
        self.register_buffer("block_load", torch.zeros(N_BLOCKS))
        # Keep track of which modulos trigger which blocks for analysis
        self.register_buffer("modulo_activations", torch.zeros(4, N_BLOCKS)) 

    def forward(self, x):
        batch, seq = x.shape
        x_emb = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x_norm = self.norm(x_emb)
        x_flat = x_norm.view(-1, D_MODEL)
        
        if self.router_type == "norm":
            w_view = self.w_in.weight.view(N_BLOCKS, BLOCK_SIZE, D_MODEL)
            sigs = torch.norm(w_view, p=2, dim=1)
            router_logits = torch.abs(x_flat) @ sigs.T
        else: # Random Router
            router_logits = torch.randn(x_flat.size(0), N_BLOCKS, device=x.device)
            
        if self.training: router_logits = router_logits + torch.randn_like(router_logits) * 0.1
            
        routing_weights = F.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, TOP_K, dim=-1)
        
        active_blocks_mask = torch.zeros(x_flat.size(0), N_BLOCKS, device=x.device)
        active_blocks_mask.scatter_(1, top_k_indices, 1.0)
        neuron_mask = active_blocks_mask.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(-1, HIDDEN_DIM)
        
        sparse_act = F.gelu(self.w_in(x_flat)) * neuron_mask
        out = self.w_out(sparse_act)
        x_final = x_emb + out.view(batch, seq, D_MODEL)
        
        # Aux Loss calculation
        expert_mask = F.one_hot(top_k_indices[:, 0], num_classes=N_BLOCKS).float()
        aux_loss = N_BLOCKS * torch.sum(expert_mask.mean(dim=0) * routing_weights.mean(dim=0))
        
        # Track statistics
        if not self.training:
            with torch.no_grad():
                self.block_load += expert_mask.sum(dim=0)
                # Group by modulo: x_flat is (batch*seq, D_MODEL). 
                # We can rebuild the sequence dimension to find the mod.
                active_blocks_seq = active_blocks_mask.view(batch, seq, N_BLOCKS)
                for mod in range(4):
                    self.modulo_activations[mod] += active_blocks_seq[:, mod::4, :].sum((0, 1))
                
        return self.head(x_final), aux_loss

# ── Training ──
def train(model, mode_name):
    print(f"\n--- Training {mode_name} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    
    loss_hist = []
    for step in range(N_STEPS):
        x, y = generate_multitask_batch(BATCH_SIZE)
        model.train()
        logits, aux_loss = model(x)
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        loss = task_loss + AUX_LOSS_WEIGHT * aux_loss if "Dense" not in mode_name else task_loss
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_hist.append(task_loss.item())
        
        if step % 200 == 0:
            print(f"  Step {step:<4} | Loss: {task_loss.item():.4f}")
            
    # Evaluation (freeze tracking)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(50):
            x, y = generate_multitask_batch(BATCH_SIZE)
            logits, _ = model(x)
            v_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            val_losses.append(v_loss.item())
            
    final_val = np.mean(val_losses)
    print(f"  Final Validation Loss: {final_val:.4f}")
    return final_val

def plot_specialization(modulo_act, filename):
    plt.figure(figsize=(10, 4))
    sns.heatmap(modulo_act.cpu().numpy(), annot=False, cmap="viridis", cbar_kws={'label': 'Activations'})
    plt.xlabel("Block Index")
    plt.ylabel("Task Modulo (0=Copy, 1=Rev, 2=Shift, 3=XOR)")
    plt.title(f"Block Specialization Map ({filename})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}.png"), dpi=150)
    plt.close()

def run_experiment():
    print("=" * 70)
    print("  PHASE 13: FUNCTIONAL SPECIALIZATION (MULTI-TASK MODULO 4)")
    print("=" * 70)
    
    # Baseline 1: Standard Dense (100% FLOPs)
    model_dense = DenseTransformer(HIDDEN_DIM).to(DEVICE)
    val_dense = train(model_dense, "Dense FFN (100% FLOPs)")
    
    # Baseline 2: Match-Compute Dense (25% FLOPs)
    model_small = DenseTransformer(TOP_K * BLOCK_SIZE).to(DEVICE)
    val_small = train(model_small, "Small Dense FFN (25% FLOPs)")
    
    # Baseline 3: Random Router (25% FLOPs)
    model_random = NDRTransformer(router_type="random").to(DEVICE)
    val_random = train(model_random, "Random Block-Sparse (25% FLOPs)")
    plot_specialization(model_random.modulo_activations, "specialization_random")
    
    # Core: Norm-Driven Routing (25% FLOPs)
    model_ndr = NDRTransformer(router_type="norm").to(DEVICE)
    val_ndr = train(model_ndr, "Norm-Driven Block-Sparse (25% FLOPs)")
    plot_specialization(model_ndr.modulo_activations, "specialization_ndr")
    
    print("\n[ FINAL RESULTS SUMMARY ]")
    print(f"  Dense (Full) : {val_dense:.4f}")
    print(f"  Dense (Small): {val_small:.4f}")
    print(f"  Random Router: {val_random:.4f}")
    print(f"  NDR Router   : {val_ndr:.4f}")

if __name__ == "__main__":
    run_experiment()

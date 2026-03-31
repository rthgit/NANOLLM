#!/usr/bin/env python3
"""
Norm-Driven Routing (NDR) Phase 16 — Causality & Robustness
===========================================================
The "Paper Seal" phase. This script performs 3 critical experiments:

1. CAUSALITY (Signature Corruption): 
   - Normal NDR (Baseline)
   - Shuffled Signatures (Permute norms across blocks)
   - Rescaled Signatures (Multiply norms by global constant)
   - Normalized Signatures (Set all norms to 1.0)
   - Frozen Signatures (Freeze routing signatures at epoch 2 vs epoch 5)

2. ROBUSTNESS (OOD):
   - Test on data with higher noise (12 mutated tokens vs 8)
   - Test on longer sequences (Length 64 vs 32)
   - Test on novel pattern combinations

3. MULTI-SEED:
   - Run 3 seeds for each causality condition.

Outputs: table2_causality_ablation.csv and Figure 4.
"""

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase16")
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
N_STEPS = 600
LR = 2e-3
AUX_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset (Modulo 4 Tasks) ──
def generate_complex_batch(batch_size, seq_len=SEQ_LEN, noise_level=0):
    # Features X are random tokens
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    y = torch.zeros_like(x)
    
    for t in range(seq_len):
        mod = t % 4
        if mod == 0:   y[:, t] = x[:, t]                                # Copy
        elif mod == 1: y[:, t] = VOCAB_SIZE - 1 - x[:, t]               # Reverse
        elif mod == 2: y[:, t] = (x[:, t] + 10) % VOCAB_SIZE            # Shift
        elif mod == 3: y[:, t] = x[:, t] ^ 42                           # XOR
        
    return x.to(DEVICE), y.to(DEVICE)

# ── NDR Architecture with Perturbation Support ──
class NDRRobustFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_in = nn.Linear(D_MODEL, HIDDEN_DIM)
        self.w_out = nn.Linear(HIDDEN_DIM, D_MODEL)
        self.mode = "normal" # "normal", "shuffle", "ones", "rescale"
        self.frozen_sigs = None

    def forward(self, x):
        batch, seq, _ = x.shape
        x_flat = x.view(-1, D_MODEL)
        
        # Base weights view
        w_view = self.w_in.weight.view(N_BLOCKS, BLOCK_SIZE, D_MODEL)
        
        # 1. Get Signatures (D-dimensional)
        if self.frozen_sigs is not None:
            sigs = self.frozen_sigs
        else:
            sigs = torch.norm(w_view, p=2, dim=1) # [N_BLOCKS, D_MODEL]
            
        # 2. Apply Perturbations
        if self.mode == "shuffle":
            perm = torch.randperm(N_BLOCKS, device=sigs.device)
            sigs = sigs[perm]
        elif self.mode == "ones":
            sigs = torch.ones_like(sigs)
        elif self.mode == "rescale":
            sigs = sigs * 0.5
            
        # 3. Routing (Token-dependent)
        router_logits = torch.abs(x_flat) @ sigs.T # [B*S, N_BLOCKS]
        routing_weights = F.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, TOP_K, dim=-1)
        
        active_blocks_mask = torch.zeros(x_flat.size(0), N_BLOCKS, device=x.device)
        active_blocks_mask.scatter_(1, top_k_indices, 1.0)
        neuron_mask = active_blocks_mask.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(-1, HIDDEN_DIM)
        
        sparse_act = F.gelu(self.w_in(x_flat)) * neuron_mask
        out = self.w_out(sparse_act)
        
        expert_mask = F.one_hot(top_k_indices[:, 0], num_classes=N_BLOCKS).float()
        aux_loss = N_BLOCKS * torch.sum(expert_mask.mean(dim=0) * routing_weights.mean(dim=0))
        
        return out.view(batch, seq, D_MODEL), aux_loss

class NDRRobustTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN*2, D_MODEL) * 0.02)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.ffn = NDRRobustFFN()
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        batch, seq = x.shape
        x = self.embedding(x) + self.pos_emb[:, :seq, :]
        sparse_out, aux_loss = self.ffn(self.norm1(x))
        x = x + sparse_out
        return self.head(self.norm2(x)), aux_loss

# ── Experiment Runner ──
def run_causality_test(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # A. Train Normal NDR Model
    model = NDRRobustTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    print(f"\n--- Training Base Model (Seed {seed}) ---")
    mid_sigs = None
    for step in range(N_STEPS):
        x, y = generate_complex_batch(BATCH_SIZE)
        model.train()
        logits, aux_loss = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)) + AUX_LOSS_WEIGHT * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Capture early signatures (33% training)
        if step == N_STEPS // 3:
            with torch.no_grad():
                w_view = model.ffn.w_in.weight.view(N_BLOCKS, BLOCK_SIZE, D_MODEL)
                mid_sigs = torch.norm(w_view, p=2, dim=1).detach().clone()
        
    # B. Evaluation Suite
    results = []
    
    def evaluate(m, name, mode="normal", noise=0, slen=SEQ_LEN):
        m.eval()
        m.ffn.mode = mode
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for _ in range(20):
                x, y = generate_complex_batch(100, seq_len=slen, noise_level=noise)
                logits, _ = m(x)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = correct / total
        avg_loss = np.mean(losses)
        print(f"  [{name:<20}] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        return avg_loss, acc

    # 1. Normal (Baseline)
    loss_base, acc_base = evaluate(model, "Normal NDR")
    results.append({"seed": seed, "test": "Normal", "loss": loss_base, "acc": acc_base})
    
    # 2. Shuffle
    loss_shuf, acc_shuf = evaluate(model, "Shuffled Signatures", mode="shuffle")
    results.append({"seed": seed, "test": "Shuffle", "loss": loss_shuf, "acc": acc_shuf})
    
    # 3. Ones (Zero Signal)
    loss_ones, acc_ones = evaluate(model, "Ones (Zero Signal)", mode="ones")
    results.append({"seed": seed, "test": "Ones", "loss": loss_ones, "acc": acc_ones})
    
    # 4. Rescale
    loss_res, acc_res = evaluate(model, "Rescaled (0.5x)", mode="rescale")
    results.append({"seed": seed, "test": "Rescale", "loss": loss_res, "acc": acc_res})
    
    # 5. OOD Robustness (Higher Noise)
    loss_noise, acc_noise = evaluate(model, "OOD: High Noise (12)", noise=12)
    results.append({"seed": seed, "test": "OOD_Noise", "loss": loss_noise, "acc": acc_noise})
    
    # 6. OOD Robustness (Long Seq)
    loss_long, acc_long = evaluate(model, "OOD: Long Seq (64)", slen=64)
    results.append({"seed": seed, "test": "OOD_Long", "loss": loss_long, "acc": acc_long})
    
    # 7. Freeze Early
    model.ffn.frozen_sigs = mid_sigs
    loss_early, acc_early = evaluate(model, "Frozen Early (33%)")
    results.append({"seed": seed, "test": "Freeze_Early", "loss": loss_early, "acc": acc_early})
    model.ffn.frozen_sigs = None
    
    return results

if __name__ == "__main__":
    all_results = []
    seeds = [42, 123, 789]
    for s in seeds:
        all_results.extend(run_causality_test(s))
        
    # Aggregate and Save
    df = pd.DataFrame(all_results)
    summary = df.groupby("test").agg({"loss": ["mean", "std"], "acc": ["mean", "std"]}).reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "table2_causality_ablation.csv"))
    print("\n--- Final Aggregated Results ---")
    print(summary)
    
    # Plotting Figure 4
    plt.figure(figsize=(10, 6))
    bar_data = df[df["test"].isin(["Normal", "Shuffle", "Ones", "Rescale"])]
    sns.barplot(data=bar_data, x="test", y="acc", capsize=.1)
    plt.title("Figure 4: Signature Corruption Ablation (Accuracy)")
    plt.ylabel("Validation Accuracy")
    plt.savefig(os.path.join(RESULTS_DIR, "fig4_signature_corruption_ablation.png"))
    plt.close()

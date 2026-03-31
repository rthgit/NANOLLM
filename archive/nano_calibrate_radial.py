import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class RadialEncoding(nn.Module):
    def __init__(self, output_dim, vocab_size=32000, token_bits=8):
        super().__init__()
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.token_bits = token_bits
        self.phi = (1 + 5**0.5) / 2
        self.projection = nn.Linear(12, output_dim, bias=True)
        
    def get_geometric_features(self, x):
        device = x.device
        dtype = torch.float32
        bits = []
        for i in range(self.token_bits):
            bits.append((x >> i) & 1)
        bits = torch.stack(bits, dim=-1).to(dtype)
        n = self.token_bits
        indices = torch.arange(n, device=device).to(dtype)
        magnitudes = (self.phi ** indices)
        angles = 2 * np.pi * indices / n
        re = torch.sum(bits * magnitudes * torch.cos(angles), dim=-1)
        im = torch.sum(bits * magnitudes * torch.sin(angles), dim=-1)
        abs_v = torch.sqrt(re**2 + im**2)
        arg_v = torch.atan2(im, re)
        feat = torch.stack([re, im, abs_v, arg_v, re*0.5, im*0.5, abs_v*0.5, arg_v*0.5, re*0.2, im*0.2, abs_v*0.2, arg_v*0.2], dim=-1)
        return feat

def calibrate_radial(model_ref, output_path="radial_projection.pt", epochs=100, batch_size=4096, device="cuda", token_bits=8):
    print(f"Calibrating Radial Projection for: {model_ref}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    config = AutoConfig.from_pretrained(model_ref)
    
    print("Loading original embeddings...")
    model = AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    original_embeddings = model.get_input_embeddings().weight.detach()
    vocab_size, hidden_size = original_embeddings.shape
    
    radial = RadialEncoding(hidden_size, vocab_size, token_bits=token_bits).to(device)
    optimizer = optim.Adam(radial.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Pre-calculating geometric features...")
    tokens = torch.arange(vocab_size)
    with torch.no_grad():
        all_features = radial.get_geometric_features(tokens) # [V, 12]
    
    dataset = TensorDataset(all_features, original_embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Distilling {vocab_size} tokens into 12-feature geometric projection ({token_bits}-bit radial)...")
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for feat_batch, target_batch in pbar:
            feat_batch, target_batch = feat_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            pred = radial.projection(feat_batch)
            loss = criterion(pred, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        
    torch.save(radial.projection.state_dict(), output_path)
    print(f"Saved calibrated projection to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--output", default="radial_projection.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--token-bits", type=int, default=8)
    args = parser.parse_args()
    
    calibrate_radial(args.model_ref, args.output, args.epochs, args.batch_size, args.device, args.token_bits)

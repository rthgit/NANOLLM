import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class RadialEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = (1 + 5**0.5) / 2
        
    def get_geometric_features(self, x):
        device = x.device
        dtype = torch.float32
        bits = []
        for i in range(8):
            bits.append((x >> i) & 1)
        bits = torch.stack(bits, dim=-1).to(dtype)
        n = 8
        indices = torch.arange(n, device=device).to(dtype)
        magnitudes = self.phi ** indices
        angles = 2 * np.pi * indices / n
        re = torch.sum(bits * magnitudes * torch.cos(angles), dim=-1)
        im = torch.sum(bits * magnitudes * torch.sin(angles), dim=-1)
        abs_v = torch.sqrt(re**2 + im**2)
        arg_v = torch.atan2(im, re)
        feat = torch.stack([re, im, abs_v, arg_v, re*0.5, im*0.5, abs_v*0.5, arg_v*0.5, re*0.2, im*0.2, abs_v*0.2, arg_v*0.2], dim=-1)
        return feat

def check():
    radial = RadialEncoding()
    tokens = torch.arange(32000)
    feat = radial.get_geometric_features(tokens)
    print(f"Radial Features - Min: {feat.min()}, Max: {feat.max()}, Mean: {feat.mean()}")
    
    weights = torch.load("E:/NANO/radial_projection.pt", map_location="cpu")
    w = weights["weight"]
    b = weights.get("bias")
    print(f"Radial Weights - Min: {w.min()}, Max: {w.max()}, Mean: {w.mean()}")
    if b is not None:
        print(f"Radial Bias - Min: {b.min()}, Max: {b.max()}, Mean: {b.mean()}")
    
    # Check for NaN/Inf
    if torch.isnan(feat).any(): print("NaN found in features!")
    if torch.isinf(feat).any(): print("Inf found in features!")
    if torch.isnan(w).any(): print("NaN found in weights!")
    if torch.isinf(w).any(): print("Inf found in weights!")
    if b is not None and torch.isnan(b).any(): print("NaN found in bias!")
    if b is not None and torch.isinf(b).any(): print("Inf found in bias!")

    # Check output range in float16
    out = F.linear(
        feat.to(torch.float32),
        w.to(torch.float32),
        b.to(torch.float32) if b is not None else None,
    ).to(torch.float16)
    print(f"Radial Output (FP16) - Min: {out.min()}, Max: {out.max()}, Mean: {out.mean()}")
    if torch.isnan(out).any(): print("NaN found in FP16 output!")

if __name__ == "__main__":
    check()

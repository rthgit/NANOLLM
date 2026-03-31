import torch
import torch.nn as nn
import numpy as np

class RadialEncoding(nn.Module):
    def __init__(self, output_dim, vocab_size=32000, token_bits=8):
        super().__init__()
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.token_bits = token_bits
        self.phi = (1 + 5**0.5) / 2
        self.projection = nn.Linear(12, output_dim, bias=True)
        
    def forward(self, x):
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
        out = self.projection(feat.to(torch.float32))
        return out

def test():
    # Test across a wide range of tokens
    tokens = torch.arange(0, 150000, 1)
    radial = RadialEncoding(3072, 128256)
    
    print(f"Testing {len(tokens)} tokens...")
    with torch.no_grad():
        out = radial(tokens)
        
    print(f"Out Min: {out.min().item()}")
    print(f"Out Max: {out.max().item()}")
    print(f"Out NaN count: {torch.isnan(out).sum().item()}")
    print(f"Out Inf count: {torch.isinf(out).sum().item()}")

if __name__ == "__main__":
    test()

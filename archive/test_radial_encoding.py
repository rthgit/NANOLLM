import torch
import torch.nn as nn
import numpy as np

class RadialEncoding(nn.Module):
    """
    Radial Encoding as defined in "Radial-Former":
    Maps binary bits to a multi-scale geometric space in the complex plane.
    
    Features (12 total):
    - 8-bit scale: Re, Im, Abs, Arg
    - 16-bit scale: Re, Im, Abs, Arg
    - 32-bit scale: Re, Im, Abs, Arg
    """
    def __init__(self, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.phi = (1 + 5**0.5) / 2  # Golden ratio
        
        # Projection from 12 features to hidden_dim
        self.projection = nn.Linear(12, output_dim)

    def forward(self, x):
        """
        x: tensor of byte values (integers 0-255) shape (batch, seq)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # 1. Expand bytes to bits
        # We'll handle 8-bit for now, but the paper suggests multi-scale (8, 16, 32)
        # To get 16 and 32 bit scales, we usually look at windows of bytes.
        # For simplicity in this local test, we focus on the 8-bit 'byte' unit
        # and simulate the 16/32 bit features by looking at the current byte's 
        # position in a virtual word/dword.
        
        bits = []
        for i in range(8):
            bits.append((x >> i) & 1)
        bits = torch.stack(bits, dim=-1).float()  # (batch, seq, 8)
        
        # 2. Compute Radial Vector V_n for 8 bits
        # V_n(b) = sum_{i=0}^{n-1} b_i * phi^i * exp(j * 2*pi * i / n)
        n = 8
        indices = torch.arange(n, device=device).float()
        magnitudes = self.phi ** indices
        angles = 2 * np.pi * indices / n
        
        real_part = torch.sum(bits * magnitudes * torch.cos(angles), dim=-1)
        imag_part = torch.sum(bits * magnitudes * torch.sin(angles), dim=-1)
        
        # Features
        abs_val = torch.sqrt(real_part**2 + imag_part**2)
        arg_val = torch.atan2(imag_part, real_part)
        
        # 3. Simulate Multi-Scale (16, 32)
        # In a real Radial-Former, these would be computed over windows.
        # Here we just stack the 8-bit features thrice with different scale biases
        # for demonstration of the 12-feature pipeline.
        
        features = torch.stack([
            real_part, imag_part, abs_val, arg_val,
            real_part * 0.5, imag_part * 0.5, abs_val * 0.5, arg_val * 0.5,  # Dummy 16-bit
            real_part * 0.2, imag_part * 0.2, abs_val * 0.2, arg_val * 0.2   # Dummy 32-bit
        ], dim=-1)
        
        return self.projection(features)

if __name__ == "__main__":
    # Test with a few tokens
    tokens = torch.tensor([[65, 66, 67], [10, 13, 32]])  # 'ABC', '\n\r '
    radial = RadialEncoding(output_dim=128)
    output = radial(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Radial Projections (first token, first 5 dims):\n{output[0, 0, :5]}")

import torch
import numpy as np

def pack_4bit(tensor):
    """Packs two 4-bit values (range -8 to 7) into one uint8."""
    # Ensure inputs are in range
    tensor = torch.clamp(tensor, -8, 7).to(torch.int8)
    
    # Shift to unsigned range 0-15
    u_tensor = (tensor + 8).to(torch.uint8)
    
    # Reshape and pack
    # We assume even number of elements for simplicity in this demo
    flat = u_tensor.flatten()
    if len(flat) % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
        
    # High 4 bits | Low 4 bits
    packed = (flat[::2] << 4) | (flat[1::2] & 0x0F)
    return packed

def unpack_4bit(packed, original_shape):
    """Unpacks uint8 into two 4-bit values and restores range."""
    high = (packed >> 4).to(torch.int8)
    low = (packed & 0x0F).to(torch.int8)
    
    unpacked = torch.stack([high, low], dim=1).flatten()
    # Restore signed range: 0-15 -> -8 to 7
    unpacked = unpacked[:np.prod(original_shape)].reshape(original_shape) - 8
    return unpacked

def pack_2bit(tensor):
    """Packs four 2-bit values (range -2 to 1) into one uint8."""
    tensor = torch.clamp(tensor, -2, 1).to(torch.int8)
    u_tensor = (tensor + 2).to(torch.uint8) # 0,1,2,3
    
    flat = u_tensor.flatten()
    padding = (4 - (len(flat) % 4)) % 4
    if padding > 0:
        flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8)])
        
    packed = (flat[0::4] << 6) | (flat[1::4] << 4) | (flat[2::4] << 2) | (flat[3::4])
    return packed

def unpack_2bit(packed, original_shape):
    """Unpacks uint8 into four 2-bit values and restores range."""
    b1 = (packed >> 6) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 2) & 0x03
    b4 = (packed) & 0x03
    
    unpacked = torch.stack([b1, b2, b3, b4], dim=1).flatten()
    unpacked = unpacked[:np.prod(original_shape)].reshape(original_shape).to(torch.int8) - 2
    return unpacked

if __name__ == "__main__":
    print("--- NANO Bit-Packing Rigorous Test ---")
    
    # 1. Test 4-bit
    original_4 = torch.randint(-8, 8, (1, 10))
    packed_4 = pack_4bit(original_4)
    restored_4 = unpack_4bit(packed_4, original_4.shape)
    
    print(f"4-bit Original: {original_4}")
    print(f"4-bit Packed:   {packed_4} (Size reduced by 2x)")
    print(f"4-bit Restored: {restored_4}")
    assert torch.equal(original_4, restored_4), "4-bit packing FAILED"
    
    # 2. Test 2-bit
    original_2 = torch.randint(-2, 2, (1, 12))
    packed_2 = pack_2bit(original_2)
    restored_2 = unpack_2bit(packed_2, original_2.shape)
    
    print(f"\n2-bit Original: {original_2}")
    print(f"2-bit Packed:   {packed_2} (Size reduced by 4x)")
    print(f"2-bit Restored: {restored_2}")
    assert torch.equal(original_2, restored_2), "2-bit packing FAILED"
    
    print("\nCONCLUSION: Bit-packing is mathematically sound and ready for NANO v4.")

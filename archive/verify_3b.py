import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os

model_path = 'E:/testmob/nanollm_a40/local_runs/Llama-3.2-3B-NANO-Reconstructed'

print(f"Loading config from {model_path}...")
config = AutoConfig.from_pretrained(model_path)
print(f"Config num_hidden_layers: {config.num_hidden_layers}")

print("Loading model for architectural validation...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

actual_layers = len(model.model.layers)
print(f"Actual layers in nn.ModuleList: {actual_layers}")

if actual_layers == config.num_hidden_layers:
    print("VERIFICATION SUCCESS: Architecture matches config.")
else:
    print("VERIFICATION FAILURE: Architecture mismatch!")

print(f"Attention heads: {model.config.num_attention_heads}")
print(f"Hidden size: {model.config.hidden_size}")

# Check for energy scaling (heuristic: check a weight norm in o_proj)
# Actually, the best check is just the layer count and config.

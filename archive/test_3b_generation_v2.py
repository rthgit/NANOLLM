import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_path = 'E:/testmob/nanollm_a40/local_runs/Llama-3.2-3B-NANO-Reconstructed'

print(f"Loading model and tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

prompt = "Once upon a time, there was a brave knight who"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

print(f"Generating output (temp=0.1) for: '{prompt}'...")
t1 = time.time()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=15,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nResponse: {decoded}")
print(f"Generation took {time.time() - t1:.2f}s")

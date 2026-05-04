import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "Explain what a neural network is in exactly 3 simple sentences.",
    "What is overfitting in machine learning? Answer in 2 sentences.",
    "Write a short bullet list with 3 uses of linear algebra in AI.",
    "Why do transformers use attention? Answer briefly.",
]


def gb(value):
    return value / 1024**3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local exported folder or HF repo id.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    model_id = args.model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        device_map="cuda",
        dtype=torch.float16,
    ).eval()

    print("embed type:", type(model.model.embed_tokens).__name__)
    print("lm_head type:", type(model.lm_head).__name__)
    print("allocated after load GB:", round(gb(torch.cuda.memory_allocated()), 4))
    print("reserved after load GB:", round(gb(torch.cuda.memory_reserved()), 4))

    for i, prompt in enumerate(DEFAULT_PROMPTS, 1):
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(next(model.parameters()).device)

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        ans = tok.decode(out[0][inp["input_ids"].shape[-1] :], skip_special_tokens=True)
        print(f"\n[{i}] {prompt}\n{ans}")
        print("peak GB:", round(gb(torch.cuda.max_memory_allocated()), 4))


if __name__ == "__main__":
    main()

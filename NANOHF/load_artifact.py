"""Loader NANO-v3.1 UNIVERSAL (Inference Only)"""
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class TrueQuantLinear(nn.Module):
    def __init__(self, pq, ps, pi, dq, ds, di, out_features, bias=None, bits=8, device="cuda:0"):
        super().__init__()
        self.out_features = out_features
        self.bits = int(bits)
        self.register_buffer("pq", pq.to(device=device, dtype=torch.int8))
        self.register_buffer("ps", ps.to(device=device, dtype=torch.float16))
        self.register_buffer("pi", pi.to(device=device, dtype=torch.long))
        self.register_buffer("dq", dq.to(device=device, dtype=torch.int8))
        self.register_buffer("ds", ds.to(device=device, dtype=torch.float16))
        self.register_buffer("di", di.to(device=device, dtype=torch.long))
        if bias is not None:
            self.register_buffer("bias", bias.to(device=device, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        d, dt = x.device, x.dtype
        f = x.to(torch.float16).reshape(-1, x.shape[-1])
        o = torch.zeros(f.shape[0], self.out_features, dtype=torch.float16, device=d)
        if self.pq.shape[0] > 0:
            o.index_copy_(-1, self.pi.to(d), f @ (self.pq.to(d, torch.float16) * self.ps.to(d).unsqueeze(1)).t())
        if self.dq.shape[0] > 0:
            o.index_copy_(-1, self.di.to(d), f @ (self.dq.to(d, torch.float16) * self.ds.to(d).unsqueeze(1)).t())
        if self.bias is not None:
            o = o + self.bias.to(d)
        return o.reshape(*x.shape[:-1], self.out_features).to(dt)


def _set(root, name, value):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = value
    else:
        setattr(parent, parts[-1], value)


def get_module(root, name):
    cur = root
    for p in name.split("."):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur


def load_artifact(artifact_dir):
    d = Path(artifact_dir)
    spec = json.loads((d / "spec.json").read_text("utf-8"))
    state = torch.load(d / "quantized_modules.pt", map_location="cpu")

    use_4bit = os.getenv("NANO_LOAD_4BIT", "0").strip().lower() in {"1", "true", "yes", "on"}
    qcfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        if use_4bit
        else BitsAndBytesConfig(load_in_8bit=True)
    )

    model = AutoModelForCausalLM.from_pretrained(
        spec["base_model_id"],
        quantization_config=qcfg,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["base_model_id"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, s in state.items():
        dev = next(get_module(model, name).parameters()).device
        bits = s["bits"]
        if "deg_q_packed" in s:
            pk, pad = s["deg_q_packed"], s["pad"]
            if bits == 2:
                dq = torch.stack([pk & 3, (pk >> 2) & 3, (pk >> 4) & 3, (pk >> 6) & 3], dim=-1).view(pk.shape[0], -1)
                if pad > 0:
                    dq = dq[:, :-pad]
                dq = dq.to(torch.int8) - 1
            else:
                dq = torch.stack([pk & 15, (pk >> 4) & 15], dim=-1).view(pk.shape[0], -1)
                if pad > 0:
                    dq = dq[:, :-pad]
                dq = dq.to(torch.int8) - 7
        else:
            dq = s.get("deg_q", torch.zeros(0, dtype=torch.int8))

        _set(
            model,
            name,
            TrueQuantLinear(
                s["prot_q"],
                s["prot_scale"],
                s["prot_idx"],
                dq,
                s["deg_scale"],
                s["deg_idx"],
                s["out_features"],
                s.get("bias"),
                bits,
                device=str(dev),
            ),
        )
    return model.eval(), tokenizer, spec


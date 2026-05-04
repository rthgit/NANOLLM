import torch
import torch.nn as nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


def _set_module(root, name, module):
    cur = root
    parts = name.split(".")
    for part in parts[:-1]:
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    setattr(cur, parts[-1], module)


class NanoInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, has_bias=False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.has_bias = bool(has_bias)
        self.register_buffer("q", torch.empty((self.out_features, self.in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.empty((self.out_features,), dtype=torch.float16))
        if self.has_bias:
            self.register_buffer("bias", torch.empty((self.out_features,), dtype=torch.float16))

    def forward(self, x):
        dt = x.dtype
        f = x.to(torch.float16).reshape(-1, x.shape[-1])
        y = torch.empty((f.shape[0], self.out_features), dtype=torch.float16, device=f.device)

        chunk_rows = 4096
        scales = self.scale.to(f.device)
        for start in range(0, self.out_features, chunk_rows):
            end = min(start + chunk_rows, self.out_features)
            q = self.q[start:end].to(f.device, torch.float16)
            w = q * scales[start:end].unsqueeze(1)
            y[:, start:end] = f @ w.t()

        if self.has_bias:
            y = y + self.bias.to(f.device)

        return y.reshape(*x.shape[:-1], self.out_features).to(dt)


class NanoTrueQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, prot_rows, deg_rows, bits, has_bias=False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.has_bias = bool(has_bias)
        self.register_buffer("prot_q", torch.empty((prot_rows, self.in_features), dtype=torch.int8))
        self.register_buffer("prot_scale", torch.empty((prot_rows,), dtype=torch.float16))
        self.register_buffer("prot_idx", torch.empty((prot_rows,), dtype=torch.long))
        packed_cols = max(1, (self.in_features + (8 // self.bits) - 1) // (8 // self.bits))
        if deg_rows == 0:
            packed_cols = 0
        self.register_buffer("deg_q_packed", torch.empty((deg_rows, packed_cols), dtype=torch.uint8))
        self.register_buffer("deg_scale", torch.empty((deg_rows,), dtype=torch.float16))
        self.register_buffer("deg_idx", torch.empty((deg_rows,), dtype=torch.long))
        if self.has_bias:
            self.register_buffer("bias", torch.empty((self.out_features,), dtype=torch.float16))

    def forward(self, x):
        dt = x.dtype
        f = x.to(torch.float16).reshape(-1, x.shape[-1])
        y = torch.zeros((f.shape[0], self.out_features), dtype=torch.float16, device=f.device)

        chunk_rows = 2048

        if self.prot_q.shape[0] > 0:
            prot_idx = self.prot_idx.to(f.device)
            prot_scale = self.prot_scale.to(f.device)
            for start in range(0, self.prot_q.shape[0], chunk_rows):
                end = min(start + chunk_rows, self.prot_q.shape[0])
                q = self.prot_q[start:end].to(f.device, torch.float16)
                w = q * prot_scale[start:end].unsqueeze(1)
                y.index_copy_(-1, prot_idx[start:end], f @ w.t())

        if self.deg_q_packed.shape[0] > 0:
            deg_idx = self.deg_idx.to(f.device)
            deg_scale = self.deg_scale.to(f.device)
            for start in range(0, self.deg_q_packed.shape[0], chunk_rows):
                end = min(start + chunk_rows, self.deg_q_packed.shape[0])
                pk = self.deg_q_packed[start:end].to(f.device)

                if self.bits == 2:
                    dq = torch.stack(
                        [pk & 3, (pk >> 2) & 3, (pk >> 4) & 3, (pk >> 6) & 3],
                        dim=-1,
                    ).view(pk.shape[0], -1).to(torch.int8) - 1
                elif self.bits == 4:
                    dq = torch.stack(
                        [pk & 15, (pk >> 4) & 15],
                        dim=-1,
                    ).view(pk.shape[0], -1).to(torch.int8) - 7
                else:
                    raise ValueError(f"unsupported packed bits: {self.bits}")

                if dq.shape[1] > self.in_features:
                    dq = dq[:, : self.in_features]

                w = dq.to(torch.float16) * deg_scale[start:end].unsqueeze(1)
                y.index_copy_(-1, deg_idx[start:end], f @ w.t())

        if self.has_bias:
            y = y + self.bias.to(f.device)

        return y.reshape(*x.shape[:-1], self.out_features).to(dt)


class NanoEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("q", torch.empty((self.num_embeddings, self.embedding_dim), dtype=torch.int8))
        self.register_buffer("scale", torch.empty((self.num_embeddings,), dtype=torch.float16))

    def forward(self, input_ids):
        return self.q[input_ids].to(torch.float16) * self.scale[input_ids].to(torch.float16).unsqueeze(-1)


class NanoTiedHead(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        object.__setattr__(self, "_embed_tokens_ref", embed_tokens)

    def _get_embed_tokens(self):
        return object.__getattribute__(self, "_embed_tokens_ref")

    def forward(self, x):
        dt = x.dtype
        f = x.to(torch.float16).reshape(-1, x.shape[-1])
        embed_tokens = self._get_embed_tokens()

        if hasattr(embed_tokens, "q"):
            qbuf = embed_tokens.q
            sbuf = getattr(embed_tokens, "scale", None)
            if sbuf is None:
                sbuf = getattr(embed_tokens, "scales")
            scales = sbuf.to(f.device)

            out_features = qbuf.shape[0]
            y = torch.empty((f.shape[0], out_features), dtype=torch.float16, device=f.device)

            chunk_rows = 4096
            for start in range(0, out_features, chunk_rows):
                end = min(start + chunk_rows, out_features)
                w = qbuf[start:end].to(f.device, torch.float16) * scales[start:end].unsqueeze(1)
                y[:, start:end] = f @ w.t()

            return y.reshape(*x.shape[:-1], out_features).to(dt)

        if hasattr(embed_tokens, "weight"):
            w = embed_tokens.weight.to(f.device, torch.float16)
            y = f @ w.t()
            return y.reshape(*x.shape[:-1], w.shape[0]).to(dt)

        raise AttributeError("embed_tokens has neither q nor weight")


class NanoQwenForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen2Config

    def tie_weights(self, *args, **kwargs):
        return

    def __init__(self, config):
        config.tie_word_embeddings = False
        super().__init__(config)

        mods = getattr(config, "nanollm_modules", {})
        for name, spec in mods.items():
            kind = spec["kind"]
            if kind == "embedding":
                mod = NanoEmbedding(spec["num_embeddings"], spec["embedding_dim"])
            elif kind == "int8_linear":
                mod = NanoInt8Linear(spec["in_features"], spec["out_features"], spec.get("has_bias", False))
            elif kind == "truequant_linear":
                mod = NanoTrueQuantLinear(
                    spec["in_features"],
                    spec["out_features"],
                    spec["prot_rows"],
                    spec["deg_rows"],
                    spec["bits"],
                    spec.get("has_bias", False),
                )
            else:
                raise ValueError(f"unknown Nano module kind: {kind}")
            _set_module(self, name, mod)

        self.lm_head = NanoTiedHead(self.model.embed_tokens)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

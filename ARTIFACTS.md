# NanoLLM v3.1 Artifact Manifest

| Model | File | Bytes | SHA256 | Test |
| --- | --- | ---: | --- | --- |
| Qwen2.5-3B-Instruct | `final_artifact_3B.zip` | 799,189,680 | `736639CC6813114DBCA1DD85456948B0E2A44163E43BA1AEDDBDC653D9113B95` | PASS, avg cosine 0.990625, min cosine 0.984375 |
| Qwen2.5-7B-Instruct | `final_artifact_7B.zip` | 891,419,698 | `F1C4D1D4BAE44E84B0388F8C1DB2004C9E2B35E4EFC21F8685E7D3810C7A9423` | PASS, avg cosine 0.990625, min cosine 0.98046875 |
| Qwen2.5-14B-Instruct | `final_artifact_Qwen2.5-14B-Instruct_pruned_pass.zip` | 1,482,019,132 | `8F535C0F05D630F08F1D54C6D89FC7DC29E8D2161426347DB3DDD2A7489F9F97` | PASS, avg cosine 0.990625, min cosine 0.98046875 |

These zip files are intentionally ignored by Git and should be uploaded to Hugging Face with `release_hf_v31.py`.

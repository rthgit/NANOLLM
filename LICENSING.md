# Licensing Structure

The current intended public position is a dual-license or dual-layer research distribution.

That means:

1. one layer for Qwen-derived materials
2. one layer for Nano-specific code and packaging

## Layer 1: Qwen-Derived Materials

If a release package contains materials derived from Qwen, those parts keep the Qwen Research License obligations.

Practically, that means:

- non-commercial research and evaluation use
- upstream attribution retained
- modified files clearly identified where applicable
- no silent relicensing of Qwen-derived artifacts as if they were purely Nano-owned

## Layer 2: Nano-Specific Code and Packaging

The Nano-specific additions authored for this repository are distributed under the Nano research-only layer described in [`LICENSE`](./LICENSE).

That layer covers:

- exporter scripts
- runtime loader code
- packaging logic
- benchmarking helpers
- release-specific documentation

## Hugging Face Metadata

Hugging Face front matter expects a single `license:` field, but this project is not described accurately by a single permissive SPDX identifier.

So the safe metadata choice remains:

- `license: other`

The human-readable body should then explain the dual-license structure explicitly.

## Release Packaging Rule

For any public compact release package, include:

- `LICENSE`
- `NOTICE`

## Documentation Rule

The public-facing text should say:

- the release is dual-license or dual-layer
- it is built with Qwen
- Nano-specific runtime and packaging changes are additional repository-authored components

# Licensing Structure

This repository currently mixes two legal layers that should not be collapsed into a single simplified claim.

## Layer 1: Upstream Base Models

When a release is derived from an upstream model such as `Qwen/Qwen2.5-3B-Instruct`, the upstream model keeps its own license terms.

That applies to:

- the original pretrained weights
- any obligations attached to the upstream model family
- any usage restrictions or attribution requirements defined by the upstream publisher

## Layer 2: Nano Code and Release Packaging

The Nano-specific code in this repository remains governed by the Nano repository license:

- exporter code
- runtime loader code
- packing logic
- release-specific scripts
- release packaging and integration logic

Today that license is defined in [`LICENSE`](./LICENSE).

## Practical Rule

For public documentation, model cards, and release notes:

- do not present the release as if a single SPDX identifier fully describes it
- do describe the distribution as composite or dual-license in plain language
- do keep the upstream model license and the Nano repository license conceptually separate

## Hugging Face Metadata

Because Hugging Face front matter expects a single `license:` value, the safe metadata choice for the current Nano compact release docs is:

- `license: other`

The human-readable body should then explain the composite or dual-license structure explicitly.

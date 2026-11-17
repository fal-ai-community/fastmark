# FastMark

A model-free, single-kernel video watermark. It runs on frames an app already
holds on the GPU, immediately before the existing encode, so it adds no decode
or encode pass and no model weights.

- **Embed** (`fastmark.embed.CudaEmbedder`): one warp per 16x16 luma tile adds a
  keyed, zero-mean raised-cosine carrier with the minimum-energy integer step
  that reaches the detection margin `tau`; tiles already at margin are not
  written, flat cells are untouched, and chroma never changes. The CUDA kernel
  is NVRTC-compiled with the libnvrtc torch ships (no CUDA toolkit needed),
  works in place on packed planar yuv420p (`embed_yuv420p`) or interleaved RGB
  (`embed_rgb`), and is bit-identical to `fastmark.reference`.
- **Payload**: `PRODUCTION_PARAMS` is presence-only: every tile is a keyed
  pilot, which gives the detector 4x the evidence of the 48-bit identifier
  profile at the same embedding energy. The identifier profile
  (`payload_bits=48`) remains available for experiments.
- **Detect** (`fastmark.detector`): blind, CPU, no original needed; reports a
  union-bounded false-positive bound (production threshold 1e-9) and recovers
  the identifier only when the tag verifies (identifier profile only). A mark
  alone is `attribution_status="mark_only"`, never "verified".
- **Key** (`fastmark.resolve_key`): `FASTMARK_SECRET_KEY` when set, otherwise an
  HMAC derivation of `C2PA_REQUEST_ID_ENCRYPTION_KEY`.

## Usage

```python
from fastmark import production_tables, resolve_key
from fastmark.embed import CudaEmbedder

key = resolve_key()
marker = CudaEmbedder(production_tables(key), device)
marker.embed_yuv420p(packed_frames)  # in place, before the device-to-host copy
```

## Install

```bash
pip install "fastmark @ git+https://github.com/fal-ai-community/fastmark.git@<sha>"
```

The fal registry pins it by commit in the MiniMax H3 Max requirements.

## Tests

```bash
pip install -e ".[test]"
pytest
```

"""FastMark: model-free, single-kernel, pre-encode luma video watermark.

The embedder runs on frames the app already holds on the GPU, right before its
existing encode, so it adds no decode/encode pass. See ``README.md``.
"""

from __future__ import annotations

import hashlib
import os

from fastmark.params import (
    ALGORITHM,
    CarrierTables,
    FastMarkParams,
    build_tables,
)

# Presence-only (every tile is a keyed pilot); measured PSNR-Y 46.4 dB median on
# delivered H3 clips.
PRODUCTION_PARAMS = FastMarkParams(tau=3.0)

FASTMARK_KEY_ENV = "FASTMARK_SECRET_KEY"


def resolve_key() -> bytes | None:
    """The FastMark secret from ``FASTMARK_SECRET_KEY``; ``None`` when unset.

    Marks are only detectable with the key live when they were embedded, so this
    secret must never be rotated or derived from another one.
    """
    explicit = os.environ.get(FASTMARK_KEY_ENV)
    if not explicit:
        return None
    return hashlib.sha256(explicit.encode()).digest()


def production_tables(
    key: bytes, params: FastMarkParams = PRODUCTION_PARAMS
) -> CarrierTables:
    """Carrier tables for the request-independent presence mark."""
    return build_tables(key, params)


__all__ = [
    "ALGORITHM",
    "FASTMARK_KEY_ENV",
    "PRODUCTION_PARAMS",
    "FastMarkParams",
    "production_tables",
    "resolve_key",
]

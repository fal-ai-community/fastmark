"""FastMark: model-free, single-kernel, pre-encode luma video watermark.

The embedder runs on frames the app already holds on the GPU, right before its
existing encode, so it adds no decode/encode pass. See ``README.md``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

from fastmark.params import (
    ALGORITHM,
    CarrierTables,
    FastMarkParams,
    build_tables,
)

# Presence-only: every tile is a keyed pilot (no identifier), which quadruples
# the detection evidence at the same embedding energy and visibility as the
# 48-bit identifier profile (tau=3.0, PSNR-Y ~55 dB).
PRODUCTION_PARAMS = FastMarkParams(tau=3.0, payload_bits=0, tag_bits=0, pilot_frac=1.0)

FASTMARK_KEY_ENV = "FASTMARK_SECRET_KEY"
_DERIVATION_SOURCE_ENV = "C2PA_REQUEST_ID_ENCRYPTION_KEY"
_DERIVATION_LABEL = ALGORITHM.encode() + b"\0watermark-key"


def resolve_key() -> bytes | None:
    """The FastMark secret: ``FASTMARK_SECRET_KEY`` or, failing that, an HMAC
    derivation of the C2PA request-id key. ``None`` when neither is set."""
    explicit = os.environ.get(FASTMARK_KEY_ENV)
    if explicit:
        return hashlib.sha256(explicit.encode()).digest()
    source = os.environ.get(_DERIVATION_SOURCE_ENV)
    if not source:
        return None
    material = base64.b64decode(source)
    return hmac.new(material, _DERIVATION_LABEL, hashlib.sha256).digest()


def production_tables(
    key: bytes, params: FastMarkParams = PRODUCTION_PARAMS
) -> CarrierTables:
    """Carrier tables for the request-independent presence mark."""
    return build_tables(key, params, asset_id=None)


__all__ = [
    "ALGORITHM",
    "FASTMARK_KEY_ENV",
    "PRODUCTION_PARAMS",
    "FastMarkParams",
    "production_tables",
    "resolve_key",
]

"""FastMark parameters and keyed carrier tables.

Everything that touches the embedder hot path is integer-exact so the NumPy
reference and the CUDA kernel produce bit-identical output.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import asdict, dataclass

import numpy as np

ALGORITHM = "fal-fastmark-v1"
DOMAIN = ALGORITHM.encode() + b"\0"

# 1-D integer bump profiles (raised-cosine-like). Their outer product is the
# within-cell carrier shape; smooth cells avoid block-edge high frequencies that
# codecs discard anyway and that read as "screen door" texture.
_BUMPS = {
    2: (1, 1),
    4: (1, 3, 3, 1),
    8: (1, 2, 3, 4, 4, 3, 2, 1),
    16: (1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1),
}


@dataclass(frozen=True)
class FastMarkParams:
    period: int = 512  # carrier repeats every `period` px (both axes)
    tile: int = 16  # projection support (tile x tile luma samples)
    cell: int = 8  # carrier cell; each tile holds (tile/cell)^2 cells
    tau: float = 2.0  # required normalized margin, luma LSB along unit carrier
    # Perceptual mask: per-cell luma std thresholds -> squared weight levels.
    sigma_thresholds: tuple[float, ...] = (1.0, 3.0, 6.0, 12.0)
    w2_levels: tuple[int, ...] = (0, 1, 2, 4, 8)  # len = len(thresholds) + 1
    caps: tuple[int, ...] = (0, 1, 2, 2, 3)  # max |delta| per level (LSB)
    luma_weights: tuple[int, int, int] = (54, 183, 19)  # BT.709 /256; must sum to 256
    skip_short: bool = True  # spend nothing on tiles that cannot reach the margin
    mode: str = (
        "informed"  # "informed" (min-energy) | "fixed" (spread-spectrum ablation)
    )
    fixed_alpha: float = 1.0  # amplitude for mode="fixed" (LSB at peak, w2=1)

    def __post_init__(self) -> None:
        if self.cell not in _BUMPS:
            raise ValueError(f"cell must be one of {sorted(_BUMPS)}")
        if self.tile % self.cell or self.period % self.tile:
            raise ValueError("tile must be a multiple of cell and divide period")
        if ((self.tile // self.cell) ** 2) % 2:
            raise ValueError("a tile needs an even number of cells (zero-mean carrier)")
        if len(self.w2_levels) != len(self.sigma_thresholds) + 1 or len(
            self.caps
        ) != len(self.w2_levels):
            raise ValueError("w2_levels/caps must have len(sigma_thresholds)+1 entries")
        if sum(self.luma_weights) != 256:
            raise ValueError(
                "luma weights must sum to 256 so +n on R,G,B moves Y by exactly n"
            )
        if self.tau <= 0 and self.mode == "informed":
            raise ValueError(
                "tau must be positive: an unset strength is not a watermark"
            )

    def to_dict(self) -> dict:
        return asdict(self)


def prf(key: bytes, label: str, nbytes: int) -> bytes:
    """HMAC-SHA256 in counter mode, domain-separated by algorithm and label."""
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        out += hmac.new(
            key,
            DOMAIN + label.encode() + b"\0" + counter.to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        counter += 1
    return bytes(out[:nbytes])


def prf_u64(key: bytes, label: str, n: int) -> np.ndarray:
    return np.frombuffer(prf(key, label, 8 * n), dtype=">u8").astype(np.uint64)


def prf_signs(key: bytes, label: str, n: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(prf(key, label, (n + 7) // 8), dtype=np.uint8))[
        :n
    ]
    return bits.astype(np.int8) * 2 - 1


@dataclass
class CarrierTables:
    """Per-key state: one int8 period x period carrier (256 KiB at P=512).

    Presence-only: every tile is a keyed pilot, so the carrier carries no payload.
    """

    params: FastMarkParams
    carrier: np.ndarray  # (P, P) int8: signed carrier incl. pilot signs
    tau_int: int  # integer margin in carrier units
    lmax: int  # bisection upper bound
    var_thresholds: np.ndarray  # int64 thresholds on n*sum(y^2) - sum(y)^2
    w2_levels: np.ndarray
    caps: np.ndarray


def build_tables(key: bytes, params: FastMarkParams) -> CarrierTables:
    P, T, c = params.period, params.tile, params.cell
    nt = P // T
    ntiles = nt * nt
    cells_per_side = T // c
    m = cells_per_side**2

    # Balanced (zero-mean) cell signs inside every tile.
    order = prf_u64(key, "cell-order", ntiles * m).reshape(ntiles, m)
    ranks = np.argsort(np.argsort(order, axis=1, kind="stable"), axis=1, kind="stable")
    cell_sign = np.where(ranks < m // 2, 1, -1).astype(np.int8)  # (ntiles, m)

    pilot_sign = prf_signs(key, "pilot-sign", ntiles).astype(np.int8)

    bump = np.array(_BUMPS[c], np.int8)
    bump2 = np.outer(bump, bump).astype(np.int16)  # (c, c)

    tile_sign = pilot_sign.reshape(nt, nt)
    cs = cell_sign.reshape(nt, nt, cells_per_side, cells_per_side)
    # expand: (nt, nt, cps, cps, c, c) -> (P, P)
    full = cs[:, :, :, :, None, None] * bump2[None, None, None, None, :, :]
    full = full * tile_sign[:, :, None, None, None, None]
    carrier = full.transpose(0, 2, 4, 1, 3, 5).reshape(P, P).astype(np.int8)

    pnorm = math.sqrt(float((bump2.astype(np.int64) ** 2).sum()) * m)
    tau_int = int(math.ceil(params.tau * pnorm))
    lmax = int(bump2.max()) * int(max(params.w2_levels)) + 1
    n = c * c
    var_thr = np.array(
        [int(math.ceil((s * n) ** 2)) for s in params.sigma_thresholds], np.int64
    )
    return CarrierTables(
        params=params,
        carrier=carrier,
        tau_int=tau_int,
        lmax=lmax,
        var_thresholds=var_thr,
        w2_levels=np.array(params.w2_levels, np.int32),
        caps=np.array(params.caps, np.int32),
    )

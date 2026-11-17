"""Readable, integer-exact NumPy reference embedder (test oracle for the CUDA kernel).

Per tile k (support S_k, keyed zero-mean carrier p, luma y):

    s   = sum_i p_i * y_i                      (exact int)
    gap = max(0, TAU - s)                      (no change when already satisfied)
    choose integer steps n_i >= 0 along sign(p_i) minimising sum_i n_i^2 / w2_i
        subject to  sum_i |p_i| n_i >= gap,  n_i <= min(cap(w2_i), headroom_i)

The separable convex integer problem is solved exactly (up to ties) by a
Lagrangian threshold: the k-th unit step at pixel i has marginal cost
(2k-1)/w2_i and gain |p_i|, so with multiplier lam

    n_i(lam) = min(cap_i, (floor(|p_i| w2_i / lam) + 1) // 2)

and we bisect for the largest integer lam whose achieved margin covers gap.
The same delta is added to R, G and B (luma-only: Cb/Cr unchanged, and
Y = (54R + 183G + 19B + 128) >> 8 moves by exactly n).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastmark.params import CarrierTables


@dataclass
class EmbedStats:
    tiles: int = 0
    tiles_changed: int = 0
    tiles_short: int = 0  # could not reach the margin within caps
    pixels_changed: int = 0
    sum_sq: int = 0
    pixels: int = 0

    def add(self, other: "EmbedStats") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["frac_tiles_untouched"] = 1 - self.tiles_changed / max(1, self.tiles)
        d["frac_tiles_short"] = self.tiles_short / max(1, self.tiles)
        d["frac_pixels_changed"] = self.pixels_changed / max(1, self.pixels)
        d["mse_luma"] = self.sum_sq / max(1, self.pixels)
        return d


def luma_from_rgb(rgb: np.ndarray, weights=(54, 183, 19)) -> np.ndarray:
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (weights[0] * r + weights[1] * g + weights[2] * b + 128) >> 8


def _tiles(a: np.ndarray, T: int) -> np.ndarray:
    nty, ntx = a.shape[0] // T, a.shape[1] // T
    return a[: nty * T, : ntx * T].reshape(nty, T, ntx, T).transpose(0, 2, 1, 3)


def compute_delta(
    y: np.ndarray, up: np.ndarray, down: np.ndarray, tables: CarrierTables
) -> tuple[np.ndarray, EmbedStats]:
    """Integer delta (H, W) for luma y with per-pixel headroom."""
    p = tables.params
    T, c, P = p.tile, p.cell, p.period
    H, W = y.shape
    nty, ntx = H // T, W // T
    Hc, Wc = nty * T, ntx * T
    delta = np.zeros((H, W), np.int32)
    stats = EmbedStats(pixels=H * W, tiles=nty * ntx)
    if nty == 0 or ntx == 0:
        return delta, stats

    yy = y[:Hc, :Wc].astype(np.int64)
    rows = np.arange(Hc) % P
    cols = np.arange(Wc) % P
    car = tables.carrier[rows[:, None], cols[None, :]].astype(np.int64)

    # Per-cell variance -> perceptual level.
    cy = yy.reshape(Hc // c, c, Wc // c, c)
    s1 = cy.sum(axis=(1, 3))
    s2 = (cy * cy).sum(axis=(1, 3))
    V = (c * c) * s2 - s1 * s1
    level = (V[..., None] >= tables.var_thresholds[None, None, :]).sum(-1)
    w2 = np.repeat(np.repeat(tables.w2_levels[level], c, 0), c, 1).astype(np.int64)
    capl = np.repeat(np.repeat(tables.caps[level], c, 0), c, 1).astype(np.int64)

    absp = np.abs(car)
    sgn = np.sign(car)
    head = np.where(sgn > 0, up[:Hc, :Wc], down[:Hc, :Wc]).astype(np.int64)
    cap = np.minimum(capl, head)
    cap = np.where(sgn == 0, 0, cap)
    score = absp * w2

    if p.mode == "fixed":
        # Spread-spectrum ablation: fixed amplitude everywhere, keyed ordered dither.
        rng = np.random.default_rng(12345)
        dither = rng.random((P, P))[rows[:, None], cols[None, :]]
        amp = p.fixed_alpha * absp / absp.max() * w2
        n = np.minimum(cap, np.floor(amp + dither).astype(np.int64))
        d = (sgn * n).astype(np.int32)
        delta[:Hc, :Wc] = d
        stats.tiles_changed = int(
            (_tiles(n, T).reshape(nty, ntx, -1) > 0).any(-1).sum()
        )
        stats.pixels_changed = int((n > 0).sum())
        stats.sum_sq = int((n * n).sum())
        return delta, stats

    st = (_tiles(car * yy, T)).sum(axis=(2, 3))  # (nty, ntx)
    gap = np.maximum(0, tables.tau_int - st)

    tscore = _tiles(score, T).reshape(nty, ntx, -1)
    tabsp = _tiles(absp, T).reshape(nty, ntx, -1)
    tcap = _tiles(cap, T).reshape(nty, ntx, -1)

    def achieved(lam: np.ndarray) -> np.ndarray:
        n = np.minimum(tcap, (tscore // lam[..., None] + 1) // 2)
        return (tabsp * n).sum(-1)

    lo = np.ones_like(gap)
    hi = np.full_like(gap, tables.lmax)
    a1 = achieved(lo)
    short = a1 < gap
    nit = int(np.ceil(np.log2(tables.lmax)))
    for _ in range(nit):
        mid = (lo + hi) >> 1
        ok = achieved(mid) >= gap
        active = (hi - lo) > 1
        lo = np.where(active & ok, mid, lo)
        hi = np.where(active & ~ok, mid, hi)
    lam = lo
    n = np.minimum(tcap, (tscore // lam[..., None] + 1) // 2)
    n = np.where((gap > 0)[..., None], n, 0)
    if p.skip_short:
        n = np.where(short[..., None], 0, n)

    nfull = n.reshape(nty, ntx, T, T).transpose(0, 2, 1, 3).reshape(Hc, Wc)
    delta[:Hc, :Wc] = (sgn * nfull).astype(np.int32)
    stats.tiles_changed = int((n.sum(-1) > 0).sum())
    stats.tiles_short = int((short & (gap > 0)).sum())
    stats.pixels_changed = int((n > 0).sum())
    stats.sum_sq = int((n * n).sum())
    return delta, stats


def embed_rgb(
    frame: np.ndarray, tables: CarrierTables
) -> tuple[np.ndarray, EmbedStats]:
    """frame: (H, W, 3) uint8 RGB -> watermarked copy."""
    y = luma_from_rgb(frame, tables.params.luma_weights)
    f = frame.astype(np.int32)
    up = 255 - f.max(-1)
    down = f.min(-1)
    delta, stats = compute_delta(y, up, down, tables)
    out = (f + delta[..., None]).astype(np.uint8)
    return out, stats


def embed_y(
    plane: np.ndarray, tables: CarrierTables, lo: int = 16, hi: int = 235
) -> tuple[np.ndarray, EmbedStats]:
    """Planar luma (encoder buffer). Headroom respects the nominal range."""
    y = plane.astype(np.int32)
    delta, stats = compute_delta(
        y, np.maximum(0, hi - y), np.maximum(0, y - lo), tables
    )
    return (y + delta).astype(plane.dtype), stats

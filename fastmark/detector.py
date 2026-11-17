"""Blind FastMark detector (no original needed, no model weights).

Per scale hypothesis s (suspect resampled by 1/s to native carrier scale):

  G(x)     = luma correlated with the c x c cell bump  (one 2-D filter)
  For every tile-grid residue r in [0,T)^2 and every frame tile g at that
  residue, the tile feature for carrier pattern k is
        s(g, k) = sum_{cells j} sign_kj * G(origin_g + c*j)
  Tiles hold 4 cells with balanced signs, so k only enters through one of 6
  sign patterns: precompute phi(s/tau) for all 6 patterns, phi = clip(., -1, 1),
  and fold over tile positions modulo the carrier period and over frames into
  a (T, T, nt, nt, 6) accumulator - constant size, streaming, any frame subset.

  For each offset d = T*q + r the pilot votes are V(q, k) = Acc_r[(k+q) mod nt, pat(k)].

Statistics (both exact under the random-key null, conditional on content):
  * sign test: #pilots with V > 0 ~ Binomial(K, 1/2)  (phi is odd, pilot chip
    signs are iid key bits independent of the content)
  * linear: sum V / sqrt(sum V^2), Hoeffding: P(>= t) <= exp(-t^2/2)
Decision: union bound over 2 * P^2 * n_scales hypotheses.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from fastmark.params import _BUMPS, CarrierTables, FastMarkParams, build_tables

DEFAULT_SCALES = (1.0, 0.75, 0.5)
LOG10_FPR_THRESHOLD = -9.0

_PATTERNS = np.array(
    [p for p in itertools.product((1, -1), repeat=4) if sum(p) == 0], np.float32
)  # (6, 4)


@dataclass
class Detection:
    watermark_detected: bool
    watermark_confidence: float  # Gaussian-equivalent z of the per-hypothesis tail
    log10_fpr_bound: float  # union-bounded log10 P(false positive)
    scale: float
    offset: tuple[int, int]
    pilot_agreement: float = 0.0
    stat_kind: str = ""
    attribution_status: str = (
        "not_detected"  # mark_only | not_detected | insufficient_signal
    )
    algorithm: str = "fal-fastmark-v1"
    signed_manifest_found: bool = False  # set by the provenance resolver, never here
    signed_manifest_trusted: bool = False
    content_binding_match: bool = False
    per_scale: dict = field(default_factory=dict)
    frames: int = 0
    log10_tail: float = 0.0  # best single-hypothesis tail (before the union bound)

    def public(self) -> dict:
        return {
            "watermark_detected": self.watermark_detected,
            "watermark_confidence": round(self.watermark_confidence, 3),
            "log10_false_positive_bound": round(self.log10_fpr_bound, 2),
            "signed_manifest_found": self.signed_manifest_found,
            "signed_manifest_trusted": self.signed_manifest_trusted,
            "content_binding_match": self.content_binding_match,
            "attribution_status": self.attribution_status,
            "algorithm": self.algorithm,
        }


def _bump(c: int) -> np.ndarray:
    b = np.array(_BUMPS[c], np.float32)
    return np.outer(b, b)


def _resample(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return img
    import cv2

    h, w = img.shape
    return cv2.resize(
        img,
        (max(1, round(w * factor)), max(1, round(h * factor))),
        interpolation=cv2.INTER_CUBIC,
    )


def tile_patterns(tables: CarrierTables) -> np.ndarray:
    """Pattern index (0..5) per carrier tile, from the base carrier signs."""
    p = tables.params
    P, T, c = p.period, p.tile, p.cell
    nt = P // T
    sg = (
        np.sign(tables.carrier[c // 2 :: c, c // 2 :: c])
        .reshape(nt, 2, nt, 2)
        .transpose(0, 2, 1, 3)
        .reshape(nt, nt, 4)
    )
    match = (sg[:, :, None, :] == _PATTERNS[None, None]).all(-1)
    assert match.sum(-1).min() == 1
    return match.argmax(-1)


class FoldAccumulator:
    """Per-scale fold of clipped tile votes. Scales are materialised lazily so a
    clear detection at native scale skips the (costlier) resampled scales; the
    union bound always counts every scale in `scales`."""

    def __init__(self, tables: CarrierTables, scales=DEFAULT_SCALES):
        p = tables.params
        if p.tile != 2 * p.cell:
            raise ValueError("detector requires tile == 2 * cell (4 cells per tile)")
        self.tables = tables
        self.scales = tuple(scales)
        self.acc: dict[float, np.ndarray] = {}
        self._frames: list[np.ndarray] = []
        self.frames = 0
        self.texture = 0.0

    def add_luma(self, y: np.ndarray) -> None:
        y = np.asarray(y)
        self._frames.append(y)
        self.frames += 1
        self.texture += float(np.std(np.diff(y.astype(np.float32), axis=1)))
        for s in self.acc:
            self._accumulate(s, y)

    def scale(self, s: float) -> np.ndarray:
        if s not in self.acc:
            p = self.tables.params
            nt = p.period // p.tile
            self.acc[s] = np.zeros((p.tile, p.tile, nt, nt, 6), np.float32)
            for y in self._frames:
                self._accumulate(s, y)
        return self.acc[s]

    def _accumulate(self, s: float, y: np.ndarray) -> None:
        import cv2

        p = self.tables.params
        P, T, c = p.period, p.tile, p.cell
        nt = P // T
        kappa = float(self.tables.tau_int)
        ys = _resample(y.astype(np.float32), 1.0 / s)
        H, W = ys.shape
        if H < T or W < T:
            return
        G = cv2.filter2D(
            ys, cv2.CV_32F, _bump(c), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT
        )
        acc = self.acc[s]
        for ry in range(T):
            ni = (H - ry) // T
            if ni <= 0:
                continue
            for rx in range(T):
                nj = (W - rx) // T
                if nj <= 0:
                    continue
                cells = [
                    G[
                        ry + c * a : ry + c * a + T * ni : T,
                        rx + c * b : rx + c * b + T * nj : T,
                    ]
                    for a in (0, 1)
                    for b in (0, 1)
                ]
                S = np.stack(cells, -1) @ _PATTERNS.T  # (ni, nj, 6)
                np.clip(S / kappa, -1.0, 1.0, out=S)
                pi, pj = -(-ni // nt) * nt, -(-nj // nt) * nt
                buf = np.zeros((pi, pj, 6), np.float32)
                buf[:ni, :nj] = S
                acc[ry, rx] += buf.reshape(pi // nt, nt, pj // nt, nt, 6).sum(
                    axis=(0, 2)
                )


def _log10_binom_sf(k: int, n: int) -> float:
    """log10 P(X >= k) for X ~ Binomial(n, 1/2)."""
    if k <= 0:
        return 0.0
    if k > n:
        return -math.inf
    lgn = math.lgamma(n + 1)
    terms = [lgn - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k, n + 1)]
    top = max(terms)
    log_sf = top + math.log(sum(math.exp(t - top) for t in terms)) - n * math.log(2)
    return min(0.0, log_sf / math.log(10))


def _z_from_log10(lt: float) -> float:
    if lt > -300:
        p = 10.0**lt
        if p >= 1.0:
            return -math.inf
        return -NormalDist().inv_cdf(p)
    return math.sqrt(-2 * lt * math.log(10))


def votes_at(
    acc_r: np.ndarray, pats: np.ndarray, mask: np.ndarray, q: tuple[int, int]
) -> np.ndarray:
    nt = pats.shape[0]
    ky, kx = np.nonzero(mask)
    return acc_r[(ky + q[0]) % nt, (kx + q[1]) % nt, pats[ky, kx]]


def detect(
    acc: FoldAccumulator,
    log10_fpr_threshold: float = LOG10_FPR_THRESHOLD,
    tables: CarrierTables | None = None,
    early_stop: bool = True,
) -> Detection:
    """`tables` overrides the accumulator's key (the fold itself is key-independent)."""
    tables = tables or acc.tables
    if (
        tables.tau_int != acc.tables.tau_int
        or tables.params.period != acc.tables.params.period
    ):
        raise ValueError("tables must share tau/period with the accumulator")
    p = tables.params
    P, T = p.period, p.tile
    nt = P // T
    pats = tile_patterns(tables)
    ky, kx = np.nonzero(np.ones((nt, nt), bool))  # every tile is a pilot
    qy, qx = np.divmod(np.arange(nt * nt), nt)
    rows = (ky[None, :] + qy[:, None]) % nt
    cols = (kx[None, :] + qx[:, None]) % nt
    pk = pats[ky, kx][None, :]
    n_hyp = 2 * P * P * len(acc.scales)
    best = None
    per_scale = {}
    for s in acc.scales:
        if (
            early_stop
            and best is not None
            and math.log10(n_hyp) + best[1] <= log10_fpr_threshold
        ):
            break  # already decisive under the full-search union bound
        A = acc.scale(s)
        if not np.any(A):
            continue
        sbest = None
        for ry in range(T):
            for rx in range(T):
                V = A[ry, rx][rows, cols, pk]  # (nq, K)
                n = (V != 0).sum(1)
                agree = (V > 0).sum(1)
                zs = (2 * agree - n) / np.sqrt(np.maximum(n, 1))
                zl = V.sum(1) / np.sqrt((V * V).sum(1) + 1e-12)
                i = int(np.argmax(np.maximum(zs, zl)))
                lt_s = _log10_binom_sf(int(agree[i]), int(n[i])) if n[i] else 0.0
                lt_l = -(zl[i] ** 2) / 2 / math.log(10) if zl[i] > 0 else 0.0
                lt, kind = (lt_s, "sign") if lt_s <= lt_l else (lt_l, "linear")
                if sbest is None or lt < sbest[0]:
                    sbest = (lt, (ry, rx), divmod(i, nt), kind, agree[i] / max(1, n[i]))
        assert sbest is not None  # T >= 1, so every scale scores some offset
        per_scale[s] = round(-sbest[0], 2)
        if best is None or sbest[0] < best[1]:
            best = (s,) + sbest
    if best is None:
        return Detection(
            False,
            0.0,
            0.0,
            1.0,
            (0, 0),
            attribution_status="insufficient_signal",
            frames=acc.frames,
        )
    s, lt, (ry, rx), (qy_, qx_), kind, agree_frac = best
    log10_fpr = min(0.0, math.log10(n_hyp) + lt)
    det = Detection(
        watermark_detected=bool(log10_fpr <= log10_fpr_threshold),
        watermark_confidence=_z_from_log10(lt),
        log10_fpr_bound=log10_fpr,
        scale=s,
        offset=((T * qy_ + ry) % P, (T * qx_ + rx) % P),
        pilot_agreement=float(agree_frac),
        stat_kind=kind,
        per_scale=per_scale,
        frames=acc.frames,
        log10_tail=lt,
    )
    if not det.watermark_detected:
        # Too little texture/frames to carry a mark: say so rather than a confident "absent".
        det.attribution_status = (
            "insufficient_signal"
            if acc.texture / max(1, acc.frames) < 0.5
            else "not_detected"
        )
        return det
    det.attribution_status = "mark_only"  # carrier presence is not authentication
    return det


def detector_tables(key: bytes, params: FastMarkParams) -> CarrierTables:
    return build_tables(key, params)


def detect_frames(
    frames_luma,
    key: bytes,
    params: FastMarkParams,
    scales=DEFAULT_SCALES,
    log10_fpr_threshold: float = LOG10_FPR_THRESHOLD,
) -> Detection:
    tables = detector_tables(key, params)
    acc = FoldAccumulator(tables, scales)
    for y in frames_luma:
        acc.add_luma(y)
    return detect(acc, log10_fpr_threshold)

import hashlib

import numpy as np
import pytest

from fastmark.detector import (
    FoldAccumulator,
    detect_frames,
    tile_patterns,
    votes_at,
)
from fastmark.params import FastMarkParams, build_tables
from fastmark.reference import embed_rgb, embed_y, luma_from_rgb

KEY = b"\x01" * 32
P = FastMarkParams(period=256, tau=3.0)


def _content(n=4, h=768, w=1024, seed=0):
    rng = np.random.default_rng(seed)
    import cv2

    base = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 12)
    base = (base - base.min()) / (base.max() - base.min()) * 100 + 60
    tex = rng.normal(0, 6, (h, w))
    frames = []
    for t in range(n):
        y = np.clip(base + tex + rng.normal(0, 1, (h, w)), 0, 255)
        rgb = np.stack([y * 0.9, y, y * 1.05], -1)
        frames.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return frames


def _mark(frames, params=P, key=KEY):
    t = build_tables(key, params)
    return [embed_rgb(f, t)[0] for f in frames]


def test_roundtrip_and_public_fields():
    m = _mark(_content())
    d = detect_frames([luma_from_rgb(f) for f in m], KEY, P, scales=(1.0,))
    assert d.watermark_detected
    assert d.log10_fpr_bound < -9
    pub = d.public()
    assert (
        pub["attribution_status"] == "mark_only"
    )  # never "verified" from the mark alone
    assert pub["signed_manifest_trusted"] is False


def test_wrong_key_and_unmarked_rejected():
    frames = _content()
    m = _mark(frames)
    assert not detect_frames(
        [luma_from_rgb(f) for f in m], b"\x02" * 32, P, scales=(1.0,)
    ).watermark_detected
    assert not detect_frames(
        [luma_from_rgb(f) for f in frames], KEY, P, scales=(1.0,)
    ).watermark_detected


def test_chunking_and_session_independence():
    frames = _content(4)
    t = build_tables(KEY, P)
    whole = [embed_rgb(f, t)[0] for f in frames]
    t2 = build_tables(KEY, P)  # fresh session, same inputs
    parts = [embed_rgb(f, t2)[0] for f in frames[2:]] + [
        embed_rgb(f, t2)[0] for f in frames[:2]
    ]
    assert all(np.array_equal(a, b) for a, b in zip(whole, parts[2:] + parts[:2]))


def test_only_luma_changes_and_bounded():
    frames = _content(2)
    m = _mark(frames)
    for a, b in zip(frames, m):
        d = b.astype(int) - a.astype(int)
        assert (d[..., 0] == d[..., 1]).all() and (
            d[..., 1] == d[..., 2]
        ).all()  # chroma-neutral
        assert np.abs(d).max() <= max(P.caps)


def test_satisfied_tiles_untouched_and_flat_frames_insufficient():
    flat = [np.full((256, 256, 3), 128, np.uint8)] * 4
    m = _mark(flat)
    assert all(
        np.array_equal(a, b) for a, b in zip(flat, m)
    )  # no headroom -> no change
    d = detect_frames([luma_from_rgb(f) for f in m], KEY, P, scales=(1.0,))
    assert not d.watermark_detected and d.attribution_status == "insufficient_signal"


def test_crop_translation_recovered():
    m = _mark(_content())
    ys = [luma_from_rgb(f)[37:, 53:] for f in m]
    d = detect_frames(ys, KEY, P, scales=(1.0,))
    assert d.watermark_detected
    assert d.offset == ((-37) % 256, (-53) % 256)


def test_y_plane_respects_range():
    y = np.clip(np.random.default_rng(0).normal(128, 40, (128, 128)), 0, 255).astype(
        np.uint8
    )
    out, _ = embed_y(y, build_tables(KEY, P), lo=16, hi=235)
    # never pushed further outside the nominal range; in-range samples stay in range
    up, down = out.astype(int) > y, out.astype(int) < y
    assert (out[up] <= 235).all() and (out[down] >= 16).all()


def test_unset_strength_is_rejected():
    with pytest.raises(ValueError):
        FastMarkParams(tau=0.0)


def test_votes_match_bruteforce():
    params = FastMarkParams(period=64, tile=16, cell=8, tau=0.05)
    t = build_tables(b"key", params)
    rng = np.random.default_rng(1)
    y = rng.normal(128, 20, size=(150, 210)).astype(np.float32)
    acc = FoldAccumulator(t, scales=(1.0,))
    acc.add_luma(y)
    pats = tile_patterns(t)
    P, T, nt = 64, 16, 4
    kappa = t.tau_int
    for d in [(0, 0), (5, 37), (63, 17), (20, 48)]:
        expect = np.zeros((nt, nt))
        r = (d[0] % T, d[1] % T)
        for oy in range(r[0], 150 - T + 1, T):
            for ox in range(r[1], 210 - T + 1, T):
                ky, kx = ((oy - d[0]) % P) // T, ((ox - d[1]) % P) // T
                s = (
                    t.carrier[(np.arange(oy, oy + T) - d[0]) % P][
                        :, (np.arange(ox, ox + T) - d[1]) % P
                    ]
                    * y[oy : oy + T, ox : ox + T]
                ).sum()
                expect[ky, kx] += np.clip(s / kappa, -1, 1)
        q = ((d[0] // T), (d[1] // T))
        got = votes_at(acc.scale(1.0)[r[0], r[1]], pats, np.ones((nt, nt), bool), q)
        np.testing.assert_allclose(got, expect.ravel(), rtol=1e-4, atol=1e-4)


def test_key_comes_only_from_the_dedicated_secret(monkeypatch):
    from fastmark import resolve_key

    monkeypatch.delenv("FASTMARK_SECRET_KEY", raising=False)
    monkeypatch.setenv("C2PA_REQUEST_ID_ENCRYPTION_KEY", "AAAA")
    assert resolve_key() is None
    monkeypatch.setenv("FASTMARK_SECRET_KEY", "explicit")
    key = resolve_key()
    assert key is not None and len(key) == 32


def test_production_carrier_is_pinned():
    """Changing the carrier makes every delivered clip undetectable."""
    from fastmark import production_tables

    tables = production_tables(KEY)
    digest = hashlib.sha256(tables.carrier.tobytes()).hexdigest()
    assert digest == "ab5e0c61f9aa839a098ab0dc60482cb1e51a7dd85715592a270f84e2b7f69815"
    assert tables.tau_int == 360


def test_production_presence_mark_detected():
    from fastmark import PRODUCTION_PARAMS, production_tables

    tables = production_tables(KEY)
    rng = np.random.default_rng(3)
    y = np.clip(rng.normal(128, 30, (1024, 1024)), 16, 235).astype(np.uint8)
    marked, stats = embed_y(y, tables)
    assert stats.tiles_changed > 0
    d = detect_frames([marked], KEY, PRODUCTION_PARAMS, scales=(1.0,))
    assert d.watermark_detected
    assert not detect_frames(
        [y], KEY, PRODUCTION_PARAMS, scales=(1.0,)
    ).watermark_detected
    assert not detect_frames(
        [marked], b"\x02" * 32, PRODUCTION_PARAMS, scales=(1.0,)
    ).watermark_detected

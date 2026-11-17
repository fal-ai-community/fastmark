"""Single-launch CUDA FastMark embedder, compiled with torch's NVRTC wrapper.

One warp per 16x16 tile (cell 8); lane ``l`` owns row ``l // 2``, columns
``(l % 2) * 8 .. +8``. A tile already at margin is never written; otherwise the
integer Lagrangian is bisected and ``+-n`` is written in place to R, G, B (RGB
mode) or to Y (planar mode). Bit-identical to
:mod:`fastmark.reference` (integer arithmetic only).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fastmark.params import CarrierTables

_SRC = r"""
#define T 16
extern "C" __global__ void fastmark_embed(
    unsigned char* __restrict__ img, int frame_stride, int row_stride, int px_stride,
    int H, int W, int nframes, int mode_rgb, int ylo, int yhi, int wr, int wg, int wb,
    const signed char* __restrict__ carrier, int P,
    int tau, int lmax, int nit, int skip_short,
    const int* __restrict__ var_thr, const int* __restrict__ w2_lv, const int* __restrict__ caps, int nlevels,
    unsigned long long* __restrict__ stats)
{
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int ntx = W / T, nty = H / T;
    const int f = blockIdx.y;
    if (warp >= ntx * nty || f >= nframes) return;
    const int ty = warp / ntx, tx = warp - ty * ntx;
    const int row = ty * T + (lane >> 1);
    const int col0 = tx * T + (lane & 1) * 8;
    unsigned char* base = img + (long long)f * frame_stride + (long long)row * row_stride + (long long)col0 * px_stride;
    const signed char* crow = carrier + (row % P) * P + (col0 % P);

    int y[8], car[8], up[8], dn[8];
    int s = 0, s1 = 0, s2 = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r, g, b, yy;
        if (mode_rgb) {
            r = base[i * px_stride]; g = base[i * px_stride + 1]; b = base[i * px_stride + 2];
            yy = (wr * r + wg * g + wb * b + 128) >> 8;
            int mx = max(r, max(g, b)), mn = min(r, min(g, b));
            up[i] = 255 - mx; dn[i] = mn;
        } else {
            yy = base[i * px_stride];
            up[i] = max(0, yhi - yy); dn[i] = max(0, yy - ylo);
        }
        y[i] = yy; car[i] = crow[i];
        s += car[i] * yy; s1 += yy; s2 += yy * yy;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    #pragma unroll
    for (int o = 2; o <= 8; o <<= 1) {
        s1 += __shfl_xor_sync(0xffffffff, s1, o);
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    int gap = tau - s;
    if (gap <= 0) return;

    const int V = 64 * s2 - s1 * s1;
    int lvl = 0;
    for (int j = 0; j < nlevels - 1; ++j) lvl += (V >= var_thr[j]);
    const int w2 = w2_lv[lvl], capl = caps[lvl];

    int score[8], cap[8], absp[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        absp[i] = abs(car[i]);
        int h = car[i] > 0 ? up[i] : dn[i];
        cap[i] = car[i] == 0 ? 0 : min(capl, h);
        score[i] = absp[i] * w2;
    }
    auto achieved = [&](int lam) {
        int a = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) a += absp[i] * min(cap[i], (score[i] / lam + 1) >> 1);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        return a;
    };
    const bool short_tile = achieved(1) < gap;
    if (short_tile && skip_short) {
        if (lane == 0) atomicAdd(&stats[1], 1ULL);
        return;
    }
    int lo = 1, hi = lmax;
    for (int it = 0; it < nit; ++it) {
        int mid = (lo + hi) >> 1;
        bool ok = achieved(mid) >= gap;
        if (hi - lo > 1) { if (ok) lo = mid; else hi = mid; }
    }
    int changed = 0, sq = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int n = min(cap[i], (score[i] / lo + 1) >> 1);
        if (n) {
            int d = car[i] > 0 ? n : -n;
            if (mode_rgb) {
                base[i * px_stride] += d; base[i * px_stride + 1] += d; base[i * px_stride + 2] += d;
            } else {
                base[i * px_stride] += d;
            }
            changed += 1; sq += n * n;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        changed += __shfl_xor_sync(0xffffffff, changed, o);
        sq += __shfl_xor_sync(0xffffffff, sq, o);
    }
    if (lane == 0) {
        if (changed) atomicAdd(&stats[0], 1ULL);
        if (short_tile) atomicAdd(&stats[1], 1ULL);
        atomicAdd(&stats[2], (unsigned long long)changed);
        atomicAdd(&stats[3], (unsigned long long)sq);
    }
}
"""

_WARPS_PER_BLOCK = 8
_kernels: dict[int, Any] = {}


def _compile_cubin(arch: str) -> bytes:
    """NVRTC-compile the header-free kernel with the libnvrtc torch ships.

    ``torch.cuda._compile_kernel`` would add CUDA toolkit include paths, which
    requires ``CUDA_HOME`` even though this kernel includes nothing.
    """
    import ctypes

    from torch.cuda._utils import _get_nvrtc_library

    nvrtc = _get_nvrtc_library()

    def check(result: int) -> None:
        if result != 0:
            message = ctypes.c_char_p()
            nvrtc.nvrtcGetErrorString(result, ctypes.byref(message))
            raise RuntimeError(f"NVRTC error: {message.value!r}")

    program = ctypes.c_void_p()
    check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), _SRC.encode(), b"fastmark.cu", 0, None, None
        )
    )
    try:
        options = [f"--gpu-architecture=sm_{arch}".encode(), b"-std=c++17"]
        result = nvrtc.nvrtcCompileProgram(
            program, len(options), (ctypes.c_char_p * len(options))(*options)
        )
        if result != 0:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"FastMark kernel compile failed: {log.value!r}")
        size = ctypes.c_size_t()
        check(nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)))
        cubin = ctypes.create_string_buffer(size.value)
        check(nvrtc.nvrtcGetCUBIN(program, cubin))
        return cubin.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def _kernel(device_index: int) -> Any:
    kernel = _kernels.get(device_index)
    if kernel is None:
        import torch
        from torch.cuda._utils import _cuda_load_module

        with torch.cuda.device(device_index):
            major, minor = torch.cuda.get_device_capability(device_index)
            cubin = _compile_cubin(f"{major}{minor}")
            kernel = _cuda_load_module(cubin, ["fastmark_embed"])["fastmark_embed"]
        _kernels[device_index] = kernel
    return kernel


class CudaEmbedder:
    """Per-key device state; every ``embed_*`` call works in place.

    Launches go on the caller's current CUDA stream and never synchronize.
    """

    def __init__(self, tables: CarrierTables, device: Any) -> None:
        import torch

        params = tables.params
        if (params.tile, params.cell) != (16, 8) or params.mode != "informed":
            raise ValueError(
                "CUDA kernel is specialised for tile=16, cell=8, informed mode"
            )
        self.tables = tables
        self.device = torch.device(device)
        levels = len(tables.w2_levels)
        thresholds = np.zeros(max(1, levels - 1), np.int32)
        thresholds[: levels - 1] = np.minimum(
            tables.var_thresholds, np.iinfo(np.int32).max
        )
        to_device = {"device": self.device, "non_blocking": False}
        self.carrier = torch.from_numpy(np.ascontiguousarray(tables.carrier)).to(
            **to_device
        )
        self.var_thr = torch.from_numpy(thresholds).to(**to_device)
        self.w2 = torch.from_numpy(tables.w2_levels.astype(np.int32)).to(**to_device)
        self.caps = torch.from_numpy(tables.caps.astype(np.int32)).to(**to_device)
        self.levels = levels
        self.stats = torch.zeros(4, dtype=torch.int64, device=self.device)
        self.iterations = int(math.ceil(math.log2(tables.lmax)))
        self._kernel = _kernel(self.device.index or 0)

    def embed_yuv420p(self, packed: Any) -> None:
        """Mark the Y plane of packed planar yuv420p ``(..., T, H*3//2, W)`` uint8."""
        if packed.ndim < 3:
            raise ValueError("packed yuv420p frames need (..., T, H*3//2, W)")
        if not packed.is_contiguous():
            raise ValueError("packed yuv420p frames must be contiguous")
        rows, width = int(packed.shape[-2]), int(packed.shape[-1])
        if rows % 3:
            raise ValueError("packed yuv420p rows must be H*3//2")
        frames = packed.reshape(-1, rows, width)
        self._launch(frames, height=rows * 2 // 3, width=width, rgb=False)

    def embed_rgb(self, frames: Any) -> None:
        """Mark ``(N, H, W, 3)`` or ``(H, W, 3)`` uint8 interleaved RGB."""
        frames = frames[None] if frames.ndim == 3 else frames
        if frames.stride(3) != 1:
            raise ValueError("RGB channels must be interleaved (HWC)")
        self._launch(
            frames, height=int(frames.shape[1]), width=int(frames.shape[2]), rgb=True
        )

    def _launch(
        self,
        frames: Any,
        *,
        height: int,
        width: int,
        rgb: bool,
        ylo: int = 16,
        yhi: int = 235,
    ) -> None:
        import torch

        if frames.dtype != torch.uint8 or frames.device != self.device:
            raise TypeError(f"uint8 frames on {self.device} required")
        count = int(frames.shape[0])
        tiles = (height // 16) * (width // 16)
        if not count or not tiles:
            return
        frame_stride, row_stride, pixel_stride = (
            int(frames.stride(0)),
            int(frames.stride(1)),
            int(frames.stride(2)),
        )
        if frame_stride >= 2**31:
            raise ValueError("frame stride exceeds the kernel's int32 range")
        tables = self.tables
        grid = ((tiles + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK, count, 1)
        self._kernel(
            grid=grid,
            block=(32 * _WARPS_PER_BLOCK, 1, 1),
            args=[
                frames,
                frame_stride,
                row_stride,
                pixel_stride,
                height,
                width,
                count,
                1 if rgb else 0,
                ylo,
                yhi,
                *(int(weight) for weight in tables.params.luma_weights),
                self.carrier,
                int(tables.params.period),
                int(tables.tau_int),
                int(tables.lmax),
                self.iterations,
                1 if tables.params.skip_short else 0,
                self.var_thr,
                self.w2,
                self.caps,
                self.levels,
                self.stats,
            ],
        )

    def reset_stats(self) -> None:
        """Zero the coverage counters (stream-ordered, no sync)."""
        self.stats.zero_()

    def read_stats(self) -> dict[str, int]:
        """Coverage counters; synchronizes with the device."""
        values = self.stats.cpu().tolist()
        return {
            "tiles_changed": int(values[0]),
            "tiles_short": int(values[1]),
            "pixels_changed": int(values[2]),
            "sum_sq": int(values[3]),
        }

"""
3D Mamba / Selective State-Space (S6) building blocks for SACN.
===============================================================
A modern state-space encoder block for volumetric medical images, replacing
the 3D ConvNeXt blocks. Mamba's selective scan gives linear-time global
receptive field over the whole volume — stronger long-range context than
convolutions and cheaper than dense attention — which is where the encoder's
representational power comes from for thin, spatially-extended cartilage sheets.

Portability: the official `mamba-ssm` selective-scan CUDA kernel is used
automatically when available (your RTX 5090). When it is not — CPU, or a
machine without the compiled kernel — we fall back to a correct pure-PyTorch
reference scan, so shapes/gradients/tests are verifiable everywhere. The two
paths share the same math; only speed differs.

References: Gu & Dao, "Mamba" (2023); Xing et al., "SegMamba" (MICCAI 2024,
tri-orientated 3D scanning). Here we use a direction-robust bidirectional scan
over a raster flattening, which is the cheapest variant that removes the causal
left-to-right bias of a single scan.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional fast CUDA kernel.
try:
    from mamba_ssm.ops.selective_scan_interface import (  # type: ignore
        selective_scan_fn as _mamba_selective_scan_fn,
    )
    _HAS_MAMBA_KERNEL = True
except Exception:  # pragma: no cover - depends on environment
    _mamba_selective_scan_fn = None
    _HAS_MAMBA_KERNEL = False

_WARNED_FALLBACK = False


def selective_scan_ref(
    u: torch.Tensor,       # (B, D, L)  input (post-conv, post-SiLU)
    delta: torch.Tensor,   # (B, D, L)  timestep (already softplus-positive)
    A: torch.Tensor,       # (D, N)     state matrix (negative)
    B: torch.Tensor,       # (B, N, L)  input-dependent
    C: torch.Tensor,       # (B, N, L)  input-dependent
    D: torch.Tensor,       # (D,)       skip connection
) -> torch.Tensor:
    """Pure-PyTorch selective scan (sequential over L). Returns y: (B, D, L).

    Correct reference for the S6 recurrence
        h_l = exp(delta_l * A) h_{l-1} + (delta_l * B_l) u_l
        y_l = C_l . h_l + D u_l
    Runs in O(L) Python steps — fine for tests and small token counts; the CUDA
    kernel is used for real training.
    """
    batch, dim, L = u.shape
    dstate = A.shape[1]
    u = u.float(); delta = delta.float()
    A = A.float(); B = B.float(); C = C.float()

    deltaA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :])      # (B,D,L,N)
    Bp = B.permute(0, 2, 1)[:, None]                                   # (B,1,L,N)
    deltaB_u = delta.unsqueeze(-1) * Bp * u.unsqueeze(-1)              # (B,D,L,N)

    h = u.new_zeros(batch, dim, dstate)
    ys = []
    for i in range(L):
        h = deltaA[:, :, i] * h + deltaB_u[:, :, i]                    # (B,D,N)
        Ci = C[:, :, i].unsqueeze(1)                                   # (B,1,N)
        ys.append((h * Ci).sum(-1))                                    # (B,D)
    y = torch.stack(ys, dim=2)                                         # (B,D,L)
    return y + u * D[None, :, None]


def selective_scan(u, delta, A, B, C, D) -> torch.Tensor:
    """Dispatch to the CUDA kernel when available, else the reference scan."""
    global _WARNED_FALLBACK
    if _HAS_MAMBA_KERNEL and u.is_cuda:
        try:
            # mamba-ssm expects delta pre-softplus handled via delta_softplus;
            # we already applied softplus, so pass delta_softplus=False.
            return _mamba_selective_scan_fn(
                u, delta, A, B, C, D, z=None, delta_bias=None,
                delta_softplus=False,
            )
        except Exception as e:  # pragma: no cover
            if not _WARNED_FALLBACK:
                warnings.warn(f"mamba-ssm kernel failed ({e}); using reference scan.")
                _WARNED_FALLBACK = True
    return selective_scan_ref(u, delta, A, B, C, D)


class SelectiveSSM(nn.Module):
    """Single-direction Mamba (S6) token mixer over a sequence (B, L, C)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A initialised as -[1..d_state] per channel (standard S4D real init).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)                                # (B,L,2*d_inner)
        xin, z = xz.chunk(2, dim=-1)                        # each (B,L,d_inner)

        xin = xin.transpose(1, 2)                           # (B,d_inner,L)
        xin = self.conv1d(xin)[..., :L]
        xin = F.silu(xin)                                   # (B,d_inner,L)

        x_t = xin.transpose(1, 2)                           # (B,L,d_inner)
        x_dbl = self.x_proj(x_t)
        dt, Bm, Cm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                   # (B,L,d_inner)

        A = -torch.exp(self.A_log.float())                  # (d_inner,d_state)
        y = selective_scan(
            u=xin,
            delta=dt.transpose(1, 2),                       # (B,d_inner,L)
            A=A,
            B=Bm.transpose(1, 2),                           # (B,d_state,L)
            C=Cm.transpose(1, 2),
            D=self.D,
        )                                                   # (B,d_inner,L)
        y = y.transpose(1, 2)                               # (B,L,d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)                             # (B,L,d_model)


class BiSSM(nn.Module):
    """Bidirectional scan: forward + reverse, removing causal ordering bias."""

    def __init__(self, d_model: int, **kw) -> None:
        super().__init__()
        self.fwd = SelectiveSSM(d_model, **kw)
        self.bwd = SelectiveSSM(d_model, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yf = self.fwd(x)
        yb = self.bwd(x.flip(1)).flip(1)
        return 0.5 * (yf + yb)


class LayerNorm3d(nn.Module):
    """Channel-first LayerNorm for (B, C, D, H, W)."""

    def __init__(self, c: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None, None] * x \
            + self.bias[None, :, None, None, None]


class Mamba3DBlock(nn.Module):
    """3D Mamba block: LN → bidirectional selective scan over flattened volume
    → layer-scaled residual, then a channel MLP (ConvNeXt-style) residual.

    Input/return: (B, C, D, H, W).
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2,
                 drop_path: float = 0.0, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = LayerNorm3d(dim)
        self.ssm = BiSSM(dim, d_state=d_state, expand=expand)
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))

        self.norm2 = LayerNorm3d(dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, 4 * dim, 1), nn.GELU(), nn.Conv3d(4 * dim, dim, 1))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path_p = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_path_p
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                                 dtype=x.dtype, device=x.device)
        return x / keep * mask.floor_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # ── Selective-scan token mixing over the flattened volume ──
        h = self.norm1(x)
        seq = h.flatten(2).transpose(1, 2)                  # (B, L=D*H*W, C)
        seq = self.ssm(seq)
        h = seq.transpose(1, 2).view(B, C, D, H, W)
        x = x + self._drop_path(self.gamma1[None, :, None, None, None] * h)
        # ── Channel MLP ──
        h = self.mlp(self.norm2(x))
        x = x + self._drop_path(self.gamma2[None, :, None, None, None] * h)
        return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | mamba-ssm kernel available: {_HAS_MAMBA_KERNEL}")

    # ── Check 1: reference selective scan shapes ──
    Bn, Dc, L, N = 2, 6, 20, 4
    u = torch.randn(Bn, Dc, L); dt = F.softplus(torch.randn(Bn, Dc, L))
    A = -torch.rand(Dc, N); Bm = torch.randn(Bn, N, L); Cm = torch.randn(Bn, N, L)
    Dp = torch.ones(Dc)
    y = selective_scan_ref(u, dt, A, Bm, Cm, Dp)
    assert y.shape == (Bn, Dc, L), y.shape
    print(f"✅ Check 1 — selective_scan_ref: {tuple(y.shape)}")

    # ── Check 2: SelectiveSSM mixer is causal (token t doesn't see t+1) ──
    ssm = SelectiveSSM(d_model=8, d_state=8).eval()
    x = torch.randn(1, 12, 8)
    y1 = ssm(x)
    x2 = x.clone(); x2[:, -1] += 5.0                        # perturb last token
    y2 = ssm(x2)
    early = (y1[:, :5] - y2[:, :5]).abs().max().item()
    late = (y1[:, -1] - y2[:, -1]).abs().max().item()
    assert early < 1e-5 < late, f"causality broken: early={early}, late={late}"
    print(f"✅ Check 2 — forward scan causal: Δearly={early:.1e}, Δlate={late:.2e}")

    # ── Check 3: Mamba3DBlock forward + backward ──
    blk = Mamba3DBlock(dim=8, d_state=8).to(device).train()
    v = torch.randn(2, 8, 6, 6, 6, device=device, requires_grad=True)
    out = blk(v)
    assert out.shape == v.shape, out.shape
    out.mean().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    print(f"✅ Check 3 — Mamba3DBlock fwd/bwd OK: {tuple(out.shape)}")

    # ── Check 4: bidirectional scan is non-causal (last-token perturbation
    #    now influences early tokens via the reverse path) ──
    bi = BiSSM(8, d_state=8).eval()
    xx = torch.randn(1, 10, 8)
    y1 = bi(xx)
    xx2 = xx.clone(); xx2[:, -1] += 5.0
    y2 = bi(xx2)
    early = (y1[:, :3] - y2[:, :3]).abs().max().item()
    # Forward-only scan yields exactly 0 here (Check 2); any nonzero leakage to
    # early tokens proves the reverse path is contributing.
    assert early > 1e-6, f"reverse path not contributing: Δearly={early}"
    print(f"✅ Check 4 — BiSSM non-causal (reverse path live): Δearly={early:.2e}")

    print(f"\n✅ All mamba3d smoke-test checks passed on device={device}")

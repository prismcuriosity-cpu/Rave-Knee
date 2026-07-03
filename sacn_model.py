"""
SACN: Severity-Aware Cartilage Network
======================================
A from-scratch 3D segmentation architecture for knee osteoarthritis (OA)
cartilage segmentation, designed as the primary model contribution alongside
the RGSSPD adapter (which remains a strong frozen-backbone baseline).

Where RGSSPD conditions a *frozen* SwinUNETR on OA severity only as a post-hoc
retrieval-gated adapter, SACN makes severity conditioning **intrinsic to the
network** and adds **clinically decisive auxiliary supervision**. Three ideas
motivate the design for knee OA specifically:

  1. Architectural severity conditioning (FiLM).
     The retrieved soft KL-grade vote (w_mild, w_moderate, w_severe) modulates
     every decoder stage via Feature-wise Linear Modulation. Severity therefore
     shapes feature computation itself, not just a late output blend — letting
     the network allocate capacity differently for the eroded, fibrillated
     cartilage of severe OA versus the near-uniform cartilage of mild knees.

  2. Boundary + thickness multi-task heads.
     Articular cartilage is a thin (1–4 mm) sheet; Dice and HD95 are won or
     lost at its surface. SACN jointly predicts (a) a signed-distance field for
     each cartilage compartment and (b) a per-voxel cartilage-thickness map —
     *the* biomarker radiologists track for OA progression (JSW / cartilage
     loss). Supervising thickness directly aligns the learned representation
     with the clinical endpoint, not just the pixel labels.

  3. Evidential per-voxel uncertainty (optional).
     The segmentation head can emit Dirichlet evidence, yielding calibrated
     voxel-wise uncertainty in a single forward pass — needed for trustworthy
     clinical deployment and increasingly expected by top-tier journals.

The encoder uses 3D ConvNeXt-style blocks (depthwise 7^3 conv → channel MLP
with layer-scale), a strong, pure-PyTorch backbone that trains stably on the
modest cohorts typical of musculoskeletal MRI and requires no CUDA-only kernels
(so it is CPU-smoke-testable here while you train on the RTX 5090).

Target: Medical Image Analysis / IEEE TMI. See PUBLICATION_PLAN.md for the
experimental protocol, baselines, ablations, and statistical testing plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SACNConfig:
    """Configuration for SACN.

    Attributes:
        in_channels: input MRI channels (1 for single-sequence DESS/T1).
        num_classes: total segmentation classes incl. background (OAI-ZIB: 6).
        cartilage_classes: label ids of cartilage compartments (for aux heads).
        base_channels: stage-0 width; stages scale 1/2/4/8/16x.
        stage_depths: ConvNeXt blocks per encoder stage (4 stages).
        severity_dim: dimensionality of the severity gate vector (3: soft vote).
        drop_path: stochastic-depth rate (linearly scaled across blocks).
        use_evidential: emit Dirichlet evidence for per-voxel uncertainty.
        deep_supervision: attach seg heads at intermediate decoder scales.
    """
    in_channels: int = 1
    num_classes: int = 6
    cartilage_classes: Tuple[int, ...] = (2, 4, 5)  # FC, MTC, LTC (OAI-ZIB)
    base_channels: int = 32
    stage_depths: Tuple[int, int, int, int] = (2, 2, 4, 2)
    severity_dim: int = 3
    drop_path: float = 0.1
    use_evidential: bool = False
    deep_supervision: bool = True

    @property
    def num_cartilage(self) -> int:
        return len(self.cartilage_classes)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm3d(nn.Module):
    """Channel-first LayerNorm for (B, C, D, H, W) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise over the channel dim only (matches ConvNeXt's LN semantics).
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None, None] * x \
            + self.bias[None, :, None, None, None]


class DropPath(nn.Module):
    """Per-sample stochastic depth (drops whole residual branches)."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep * mask.floor_()


class ConvNeXtBlock3D(nn.Module):
    """3D ConvNeXt block: depthwise 7^3 conv → channel MLP → layer-scale residual."""

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma[None, :, None, None, None] * x
        return shortcut + self.drop_path(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: y = (1 + gamma) * x + beta.

    gamma/beta are produced from the severity gate by a shared MLP; the `1 +`
    keeps modulation an identity-centred perturbation so an all-zero severity
    signal leaves features unchanged (stable warm-start).
    """

    def __init__(self, cond_dim: int, num_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * num_features),
        )
        # Zero-init the final layer → gamma=beta=0 → identity at start.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.num_features = num_features

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.net(cond)                       # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)          # (B, C) each
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return (1.0 + gamma) * x + beta


class UpBlock(nn.Module):
    """Decoder stage: trilinear upsample → fuse skip → ConvNeXt block → FiLM."""

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int,
                 cond_dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(in_dim, out_dim, kernel_size=1)
        self.fuse = nn.Conv3d(out_dim + skip_dim, out_dim, kernel_size=1)
        self.block = ConvNeXtBlock3D(out_dim, drop_path=drop_path)
        self.film = FiLM(cond_dim, out_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                          align_corners=False)
        x = self.reduce(x)
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.block(x)
        return self.film(x, cond)


# ---------------------------------------------------------------------------
# SACN
# ---------------------------------------------------------------------------

class SACN(nn.Module):
    """Severity-Aware Cartilage Network.

    Forward returns a dict:
        seg_logits:  (B, num_classes, D, H, W)   — primary segmentation
        boundary:    (B, num_cartilage, D, H, W) — signed distance field (tanh)
        thickness:   (B, num_cartilage, D, H, W) — per-voxel thickness (>=0)
        evidence:    (B, num_classes, D, H, W)   — Dirichlet evidence (if enabled)
        uncertainty: (B, 1, D, H, W)             — voxel uncertainty (if enabled)
        aux_seg:     list of (B, num_classes, ...) — deep-supervision logits
    """

    def __init__(self, cfg: SACNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        b = cfg.base_channels
        dims = [b, 2 * b, 4 * b, 8 * b, 16 * b]   # stage 0..4
        self.dims = dims

        # Stochastic-depth schedule across all encoder blocks.
        total_blocks = sum(cfg.stage_depths)
        dpr = torch.linspace(0, cfg.drop_path, max(total_blocks, 1)).tolist()

        # Full-resolution stem (keeps a full-res skip so output is full-res).
        self.stem = nn.Sequential(
            nn.Conv3d(cfg.in_channels, dims[0], kernel_size=3, padding=1),
            LayerNorm3d(dims[0]),
            nn.GELU(),
            ConvNeXtBlock3D(dims[0], drop_path=0.0),
        )

        # Encoder: 4 downsampling stages.
        self.down_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        blk = 0
        for i in range(4):
            self.down_layers.append(nn.Sequential(
                LayerNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
            depth = cfg.stage_depths[i]
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock3D(dims[i + 1], drop_path=dpr[blk + j])
                for j in range(depth)
            ]))
            blk += depth

        cond_dim = cfg.severity_dim
        # Decoder: symmetric up path with FiLM at every stage.
        self.up4 = UpBlock(dims[4], dims[3], dims[3], cond_dim, cfg.drop_path)
        self.up3 = UpBlock(dims[3], dims[2], dims[2], cond_dim, cfg.drop_path)
        self.up2 = UpBlock(dims[2], dims[1], dims[1], cond_dim, cfg.drop_path)
        self.up1 = UpBlock(dims[1], dims[0], dims[0], cond_dim, cfg.drop_path)

        # Primary + auxiliary heads (all at full resolution after up1).
        self.seg_head = nn.Conv3d(dims[0], cfg.num_classes, kernel_size=1)
        self.boundary_head = nn.Conv3d(dims[0], cfg.num_cartilage, kernel_size=1)
        self.thickness_head = nn.Conv3d(dims[0], cfg.num_cartilage, kernel_size=1)

        if cfg.deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(dims[1], cfg.num_classes, kernel_size=1),
                nn.Conv3d(dims[2], cfg.num_classes, kernel_size=1),
            ])
        else:
            self.aux_heads = None

    # ------------------------------------------------------------------
    def _default_severity(self, batch: int, device: torch.device) -> torch.Tensor:
        """Uniform severity gate (no prior) — used when none is supplied."""
        return torch.full((batch, self.cfg.severity_dim),
                          1.0 / self.cfg.severity_dim, device=device)

    def forward(self, x: torch.Tensor,
                severity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        if severity is None:
            severity = self._default_severity(B, x.device)
        elif severity.dim() == 1:
            severity = severity.unsqueeze(0).expand(B, -1)

        # ── Encoder ──
        s0 = self.stem(x)                     # full res, dims[0]
        skips = [s0]
        feat = s0
        for down, stage in zip(self.down_layers, self.stages):
            feat = down(feat)
            feat = stage(feat)
            skips.append(feat)                # dims[1..4] at /2../16
        # skips = [s0(/1), s1(/2), s2(/4), s3(/8), s4(/16)]

        # ── Decoder (FiLM-conditioned on severity) ──
        d4 = self.up4(skips[4], skips[3], severity)   # /8, dims[3]
        d3 = self.up3(d4, skips[2], severity)         # /4, dims[2]
        d2 = self.up2(d3, skips[1], severity)         # /2, dims[1]
        d1 = self.up1(d2, skips[0], severity)         # /1, dims[0]

        out: Dict[str, torch.Tensor] = {}
        out["seg_logits"] = self.seg_head(d1)
        out["boundary"] = torch.tanh(self.boundary_head(d1))
        out["thickness"] = F.softplus(self.thickness_head(d1))

        if self.cfg.use_evidential:
            evidence = F.softplus(out["seg_logits"])          # (B,K,...)
            alpha = evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            out["evidence"] = evidence
            out["uncertainty"] = self.cfg.num_classes / S     # (B,1,...)

        if self.aux_heads is not None and self.training:
            out["aux_seg"] = [self.aux_heads[0](d2), self.aux_heads[1](d3)]
        return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def soft_dice_loss(logits: torch.Tensor, gt_onehot: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """Mean foreground soft-Dice loss (skips background channel 0)."""
    probs = torch.softmax(logits, dim=1)
    dims = (0, 2, 3, 4)
    tp = (probs * gt_onehot).sum(dims)
    fp = (probs * (1 - gt_onehot)).sum(dims)
    fn = ((1 - probs) * gt_onehot).sum(dims)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return (1.0 - dice[1:]).mean()


def _one_hot(gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B,1,D,H,W) or (B,D,H,W) int labels → (B,K,D,H,W) one-hot float."""
    if gt.dim() == 4:
        gt = gt.unsqueeze(1)
    B, _, D, H, W = gt.shape
    oh = torch.zeros(B, num_classes, D, H, W, device=gt.device, dtype=torch.float32)
    return oh.scatter_(1, gt.long(), 1.0)


def evidential_seg_loss(evidence: torch.Tensor, gt_onehot: torch.Tensor,
                        lam: float = 0.1) -> torch.Tensor:
    """Dirichlet type-II MLE (Bayes-risk Dice-free) + KL-to-uniform regulariser.

    Uses the expected-cross-entropy form of the evidential loss (Sensoy et al.,
    2018) so mis-evidence on wrong classes is penalised and shrinks toward the
    uniform Dirichlet, giving calibrated uncertainty.
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    # Expected cross-entropy: sum_k y_k (psi(S) - psi(alpha_k))
    ll = (gt_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1).mean()
    # KL(Dirichlet(alpha_tilde) || Dirichlet(1)) on the mis-evidence.
    alpha_tilde = gt_onehot + (1 - gt_onehot) * alpha
    K = evidence.shape[1]
    S_t = alpha_tilde.sum(dim=1, keepdim=True)
    term1 = torch.lgamma(S_t.squeeze(1)) - torch.lgamma(
        torch.tensor(float(K), device=evidence.device))
    term1 = term1 - torch.lgamma(alpha_tilde).sum(dim=1)
    term2 = ((alpha_tilde - 1) *
             (torch.digamma(alpha_tilde) - torch.digamma(S_t))).sum(dim=1)
    kl = (term1 + term2).mean()
    return ll + lam * kl


def sacn_loss(
    outputs: Dict[str, torch.Tensor],
    gt_mask: torch.Tensor,
    cfg: SACNConfig,
    gt_boundary: Optional[torch.Tensor] = None,
    gt_thickness: Optional[torch.Tensor] = None,
    lambda_boundary: float = 0.3,
    lambda_thickness: float = 0.2,
    lambda_aux: float = 0.4,
    evidential_ramp: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Composite SACN objective.

    Args:
        outputs: dict from SACN.forward.
        gt_mask: (B,1,D,H,W)/(B,D,H,W) int labels.
        cfg: SACNConfig.
        gt_boundary: (B,num_cartilage,D,H,W) target signed distance field, or None.
        gt_thickness: (B,num_cartilage,D,H,W) target thickness map, or None.
        lambda_*: task weights.
        evidential_ramp: anneal factor in [0,1] for the evidential KL term.

    Returns:
        (total_loss, breakdown_dict).
    """
    seg_logits = outputs["seg_logits"]
    gt_onehot = _one_hot(gt_mask, cfg.num_classes)

    # ── Segmentation: Dice (+ evidential or CE) ──
    dice = soft_dice_loss(seg_logits, gt_onehot)
    if cfg.use_evidential and "evidence" in outputs:
        ev = evidential_seg_loss(outputs["evidence"], gt_onehot,
                                 lam=0.1 * evidential_ramp)
        seg = dice + ev
        ev_val = float(ev.detach())
    else:
        ce = F.cross_entropy(seg_logits, gt_mask.long().squeeze(1)
                             if gt_mask.dim() == 5 else gt_mask.long())
        seg = dice + ce
        ev_val = float(ce.detach())

    total = seg
    breakdown = {"dice": float(dice.detach()), "seg_aux": ev_val}

    # ── Boundary signed-distance regression (cartilage only) ──
    if gt_boundary is not None and "boundary" in outputs:
        bd = F.l1_loss(outputs["boundary"], gt_boundary)
        total = total + lambda_boundary * bd
        breakdown["boundary"] = float(bd.detach())

    # ── Thickness regression (masked to cartilage voxels) ──
    if gt_thickness is not None and "thickness" in outputs:
        mask = (gt_thickness > 0).float()
        denom = mask.sum().clamp_min(1.0)
        th = (F.l1_loss(outputs["thickness"], gt_thickness, reduction="none")
              * mask).sum() / denom
        total = total + lambda_thickness * th
        breakdown["thickness"] = float(th.detach())

    # ── Deep supervision ──
    if "aux_seg" in outputs and outputs["aux_seg"]:
        aux_total = 0.0
        for aux in outputs["aux_seg"]:
            aux_gt = F.interpolate(
                gt_onehot, size=aux.shape[2:], mode="nearest")
            aux_total = aux_total + soft_dice_loss(aux, aux_gt)
        aux_total = aux_total / len(outputs["aux_seg"])
        total = total + lambda_aux * aux_total
        breakdown["aux"] = float(aux_total.detach())

    breakdown["total"] = float(total.detach())
    return total, breakdown


# ---------------------------------------------------------------------------
# Target-field helpers (build boundary / thickness supervision from a mask)
# ---------------------------------------------------------------------------

def build_target_fields(
    gt_mask, cartilage_classes: Sequence[int], clip_mm: float = 8.0,
):
    """Derive signed-distance and thickness targets from a label volume.

    Pure-numpy/scipy so it can run in the DataLoader. Returns:
        boundary: (num_cartilage, D, H, W) signed distance, +inside, normalised
                  to [-1, 1] by clip_mm (assumes ~isotropic voxels; scale outside).
        thickness: (num_cartilage, D, H, W) local thickness proxy (voxels),
                   = 2 * interior distance transform, zero outside the class.
    """
    import numpy as np
    from scipy import ndimage as ndi

    if hasattr(gt_mask, "detach"):
        gt = gt_mask.detach().cpu().numpy()
    else:
        gt = np.asarray(gt_mask)
    gt = np.squeeze(gt)

    bnd = np.zeros((len(cartilage_classes),) + gt.shape, dtype=np.float32)
    thk = np.zeros_like(bnd)
    for i, c in enumerate(cartilage_classes):
        region = (gt == c)
        if not region.any():
            continue
        dt_in = ndi.distance_transform_edt(region)
        dt_out = ndi.distance_transform_edt(~region)
        signed = dt_in - dt_out
        bnd[i] = np.clip(signed / clip_mm, -1.0, 1.0)
        thk[i] = (2.0 * dt_in).astype(np.float32)  # local thickness proxy
    return bnd, thk


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = SACNConfig(in_channels=1, num_classes=6, base_channels=8,
                     stage_depths=(1, 1, 2, 1), use_evidential=True,
                     deep_supervision=True)
    model = SACN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ SACN built: {n_params/1e6:.2f}M params, dims={model.dims}")

    x = torch.randn(2, 1, 32, 32, 32, device=device)
    severity = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], device=device)

    # ── Check 1: train-mode forward returns all heads + aux ──
    model.train()
    out = model(x, severity)
    assert out["seg_logits"].shape == (2, 6, 32, 32, 32), out["seg_logits"].shape
    assert out["boundary"].shape == (2, 3, 32, 32, 32)
    assert out["thickness"].shape == (2, 3, 32, 32, 32)
    assert out["thickness"].min() >= 0, "thickness must be non-negative"
    assert "evidence" in out and "uncertainty" in out
    assert out["uncertainty"].shape == (2, 1, 32, 32, 32)
    assert "aux_seg" in out and len(out["aux_seg"]) == 2
    print(f"✅ Check 1 — forward heads OK; "
          f"unc∈[{out['uncertainty'].min():.3f},{out['uncertainty'].max():.3f}]")

    # ── Check 2: severity actually changes the output (FiLM is live) ──
    with torch.no_grad():
        model.eval()
        o_a = model(x, torch.tensor([[1., 0., 0.]], device=device).expand(2, -1))
        o_b = model(x, torch.tensor([[0., 0., 1.]], device=device).expand(2, -1))
    delta = (o_a["seg_logits"] - o_b["seg_logits"]).abs().mean().item()
    # After zero-init FiLM this is ~0 at step 0; perturb FiLM to prove wiring.
    for m in model.modules():
        if isinstance(m, FiLM):
            with torch.no_grad():
                m.net[-1].weight.normal_(0, 0.1)
                m.net[-1].bias.normal_(0, 0.1)
    with torch.no_grad():
        o_a = model(x, torch.tensor([[1., 0., 0.]], device=device).expand(2, -1))
        o_b = model(x, torch.tensor([[0., 0., 1.]], device=device).expand(2, -1))
    delta2 = (o_a["seg_logits"] - o_b["seg_logits"]).abs().mean().item()
    assert delta2 > 1e-4, f"FiLM not conditioning output (Δ={delta2})"
    print(f"✅ Check 2 — severity conditioning live: Δlogits "
          f"{delta:.2e} (init) → {delta2:.2e} (perturbed)")

    # ── Check 3: composite loss is scalar w/ grad, backprops ──
    model.train()
    out = model(x, severity)
    gt = torch.randint(0, 6, (2, 1, 32, 32, 32), device=device)
    import numpy as np
    bnds, thks = [], []
    for b in range(2):
        bd, th = build_target_fields(gt[b], cfg.cartilage_classes)
        bnds.append(bd); thks.append(th)
    gt_boundary = torch.from_numpy(np.stack(bnds)).to(device)
    gt_thickness = torch.from_numpy(np.stack(thks)).to(device)
    loss, bd = sacn_loss(out, gt, cfg, gt_boundary, gt_thickness)
    assert loss.requires_grad and loss.ndim == 0
    loss.backward()
    grad_norm = sum(p.grad.abs().sum() for p in model.parameters()
                    if p.grad is not None).item()
    assert grad_norm > 0, "no gradient flowed"
    print(f"✅ Check 3 — sacn_loss={loss.item():.4f} | {bd}")

    # ── Check 4: target-field helper shapes & signs ──
    bd, th = build_target_fields(gt[0], cfg.cartilage_classes)
    assert bd.shape == (3, 32, 32, 32) and th.shape == (3, 32, 32, 32)
    assert bd.min() >= -1.0 and bd.max() <= 1.0
    assert th.min() >= 0.0
    print(f"✅ Check 4 — target fields: boundary∈[{bd.min():.2f},{bd.max():.2f}], "
          f"thickness_max={th.max():.1f}")

    # ── Check 5: non-evidential + no deep-supervision path ──
    cfg2 = SACNConfig(num_classes=6, base_channels=8, stage_depths=(1, 1, 1, 1),
                      use_evidential=False, deep_supervision=False)
    m2 = SACN(cfg2).to(device).train()
    o2 = m2(x)
    assert "evidence" not in o2 and "aux_seg" not in o2
    l2, _ = sacn_loss(o2, gt, cfg2)
    l2.backward()
    print(f"✅ Check 5 — minimal config OK: loss={l2.item():.4f}")

    print(f"\n✅ All SACN smoke-test checks passed on device={device}")

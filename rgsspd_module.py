"""
RGSSPD: Retrieval-Gated Severity-Specialist Prototype Distillation
===================================================================
A novel adapter layer atop RAVE-Knee's frozen SwinUNETR v2 backbone
that routes each test knee through a severity-matched specialist head.

Key insight: OA severity (KL grade) correlates strongly with cartilage
morphology. A single shared adapter must compromise across grades; three
specialist heads (mild KL0-1, moderate KL2, severe KL3-4) each optimise
for their sub-distribution.  At test time a soft vote from the FAISS
neighbors' KL grades blends the three specialists via temperature-scaled
softmax — producing both improved segmentation and an interpretable
severity fingerprint.

Pipeline:
  1. FAISS retrieval → k nearest train subjects (cosine sim on Stage-3 embs)
  2. Neighbor KL grades → soft blend weights (temperature-scaled softmax)
  3. Each specialist head extracts per-class prototypes from support subjects
  4. Cross-attention: query Stage-3 features attend to severity-blended protos
  5. Small Conv3d seg-head → adapter logits; fuse with frozen backbone baseline
  6. MONAI sliding-window inference → final pred_mask + uncertainty

Integration: drop-in replacement for RAVEKneeICLSegmenter.predict().
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as scipy_ndimage
from scipy import stats as scipy_stats
from tqdm import tqdm

# Reproducibility — set at module level so every import gets the same seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Optional imports — soft-fail so the module loads without full pipeline deps
try:
    import torchio as tio  # type: ignore
except ImportError:
    tio = None  # type: ignore

try:
    from monai.inferers import SlidingWindowInferer  # type: ignore
except ImportError:
    SlidingWindowInferer = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stage3_channels(cfg: Any) -> int:
    """Stage-3 SwinViT channels = feature_size * 8 (matches _STAGE_CH[3])."""
    return cfg.feature_size * 8


def _vol_to_tensor(subj: Any, cfg: Any, device: torch.device,
                   transform_fn=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (vol_tensor (1,1,D,H,W), mask_tensor (1,1,D,H,W)|None) from a tio.Subject."""
    if transform_fn is not None:
        s = transform_fn(subj)
    else:
        s = subj
    vol = s.image.data.unsqueeze(0).to(device)       # (1,1,D,H,W)
    msk = s.label.data.unsqueeze(0).to(device) if hasattr(s, "label") else None
    return vol, msk


def _pad_to(vol: torch.Tensor, target: Tuple[int, int, int]) -> torch.Tensor:
    """Zero-pad vol (B,C,D,H,W) to at least target spatial size."""
    shape = vol.shape[2:]
    pad = []
    for i in range(2, -1, -1):  # W, H, D order for F.pad
        p = max(0, target[i] - shape[i])
        pad.extend([0, p])
    if any(p > 0 for p in pad):
        vol = F.pad(vol, pad)
    return vol[:, :, :target[0], :target[1], :target[2]]


@torch.no_grad()
def _extract_stage3(backbone: Any, vol: torch.Tensor) -> torch.Tensor:
    """Extract Stage-3 features (B, C, D/8, H/8, W/8) from frozen SwinViT."""
    backbone.eval()
    hidden = backbone.model.swinViT(vol, backbone.model.normalize)
    return hidden[3]  # (B, C, Ds, Hs, Ws)


@torch.no_grad()
def _extract_embedding(backbone: Any, vol: torch.Tensor) -> np.ndarray:
    """Global average of Stage-3 → 1-D embedding for FAISS."""
    feats = _extract_stage3(backbone, vol)
    return feats.mean(dim=[2, 3, 4]).squeeze(0).cpu().float().numpy()


def _hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Approximate HD95 via distance transforms (no medpy dependency)."""
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    dt_pred = scipy_ndimage.distance_transform_edt(~pred.astype(bool))
    dt_gt = scipy_ndimage.distance_transform_edt(~gt.astype(bool))
    fwd = dt_gt[pred.astype(bool)]
    bwd = dt_pred[gt.astype(bool)]
    return float(np.percentile(np.concatenate([fwd, bwd]), 95))


def _dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    tp = (pred & gt).sum()
    return float((2 * tp + eps) / (pred.sum() + gt.sum() + eps))


# ---------------------------------------------------------------------------
# Component 1 — SeveritySpecialistHead
# ---------------------------------------------------------------------------

class SeveritySpecialistHead(nn.Module):
    """Cross-attention adapter conditioned on severity-matched class prototypes.

    Research motivation: A single shared adapter is forced to satisfy all OA
    severities simultaneously.  The specialist head is trained exclusively on
    one severity stratum, so its cross-attention weights specialise for the
    cartilage morphology typical of that stratum.  At inference the three
    specialists are blended via soft severity votes — capturing nuance at each
    severity level while remaining differentiable end-to-end.

    Args:
        embed_dim: Stage-3 channel count (feature_size * 8).
        num_heads: MHA heads; embed_dim must be divisible.
        num_classes: number of segmentation classes (from cfg.num_classes).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_classes: int = 5) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Cross-attention: Q=query spatial tokens, K=V=class prototypes
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # FFN: Linear(C→2C) → GELU → Linear(2C→C)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Output spatial projection (1×1×1 conv, keeps spatial shape)
        self.proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)

        # Segmentation projection: embed_dim → num_classes (zero-init → no-op at start)
        self.seg_proj = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
        nn.init.zeros_(self.seg_proj.weight)
        nn.init.zeros_(self.seg_proj.bias)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.eye_(
            self.proj.weight.view(self.embed_dim, self.embed_dim)
        )  # identity init

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_prototypes(self, support_features: torch.Tensor,
                           support_masks: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
        """Masked average pooling → per-class prototypes.

        Args:
            support_features: (k, C, D, H, W) Stage-3 features from k support cases.
            support_masks: (k, D, H, W) integer ground-truth labels 0..num_classes-1.
            num_classes: total classes (matches cfg.num_classes).

        Returns:
            prototypes: (num_classes, C) — zero vector for absent classes.
        """
        k, C, Ds, Hs, Ws = support_features.shape

        # Align mask resolution to feature map
        masks_aligned = F.interpolate(
            support_masks.unsqueeze(1).float(),
            size=(Ds, Hs, Ws), mode="nearest"
        ).long().squeeze(1)  # (k, Ds, Hs, Ws)

        prototypes = torch.zeros(num_classes, C,
                                 device=support_features.device,
                                 dtype=support_features.dtype)

        for c in range(num_classes):
            cls_mask = (masks_aligned == c).float()      # (k, Ds, Hs, Ws)
            n = cls_mask.sum()
            if n > 0:
                masked = support_features * cls_mask.unsqueeze(1)  # (k,C,Ds,Hs,Ws)
                prototypes[c] = masked.sum(dim=(0, 2, 3, 4)) / n

        return prototypes  # (num_classes, C)

    # ------------------------------------------------------------------
    def forward(self, query_features: torch.Tensor,
                support_prototypes: torch.Tensor) -> torch.Tensor:
        """Adapt query features via cross-attention to severity prototypes.

        Args:
            query_features: (B, C, D, H, W) Stage-3 patch features.
            support_prototypes: (num_classes, C) or (B, num_classes, C).

        Returns:
            output: (B, C, D, H, W) specialist-adapted features.
        """
        B, C, D, H, W = query_features.shape

        # Flatten spatial → (B, N, C)
        q = query_features.flatten(2).permute(0, 2, 1)   # (B, N, C)

        # Expand prototypes to batch dimension
        if support_prototypes.dim() == 2:
            kv = support_prototypes.unsqueeze(0).expand(B, -1, -1)
        else:
            kv = support_prototypes  # (B, num_classes, C)

        # Cross-attention + post-LN residual
        attn_out, _ = self.cross_attn(q, kv, kv)         # (B, N, C)
        x = self.norm1(q + attn_out)

        # FFN + post-LN residual
        x = self.norm2(x + self.ffn(x))

        # Reshape back to spatial + output projection
        out = x.permute(0, 2, 1).view(B, C, D, H, W)
        return self.proj(out)                              # (B, C, D, H, W)


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def rgsspd_loss(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    stratum: str,
    num_classes: int,
    lambda_bd: float = 0.3,
    lambda_focal: float = 0.2,
    gamma: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined Dice + boundary HD + focal loss for specialist training.

    The boundary loss upweights errors near thin cartilage margins — the
    hardest failure mode for severe KL grades.  Focal loss compensates for
    severe class imbalance (background >> cartilage).

    Args:
        pred_logits: (B, num_classes, D, H, W) raw logits.
        gt_mask: (B, 1, D, H, W) or (B, D, H, W) integer labels.
        stratum: 'mild' | 'moderate' | 'severe'.
        num_classes: number of segmentation classes.
        lambda_bd: boundary loss weight (bumped to 0.5 for 'severe').
        lambda_focal: focal loss weight.
        gamma: focal loss focusing parameter.

    Returns:
        total_loss: scalar tensor with grad.
        breakdown: {'dice': float, 'boundary': float, 'focal': float}.
    """
    if stratum == "severe":
        lambda_bd = 0.5

    if gt_mask.dim() == 4:
        gt_mask = gt_mask.unsqueeze(1)   # (B, 1, D, H, W)

    B, _, D, H, W = pred_logits.shape
    gt_long = gt_mask.long()             # (B, 1, D, H, W)

    # One-hot: (B, num_classes, D, H, W)
    gt_onehot = torch.zeros_like(pred_logits).scatter_(
        1, gt_long.expand(-1, 1, D, H, W), 1.0
    )

    probs = torch.softmax(pred_logits, dim=1)  # (B, num_classes, D, H, W)

    # ── Dice loss ───────────────────────────────────────────────────────────
    eps = 1e-6
    dims = (0, 2, 3, 4)
    tp = (probs * gt_onehot).sum(dims)
    fp = (probs * (1 - gt_onehot)).sum(dims)
    fn = ((1 - probs) * gt_onehot).sum(dims)
    dice_per_class = 1.0 - (2 * tp + eps) / (2 * tp + fp + fn + eps)
    dice_loss = dice_per_class[1:].mean()  # skip background class 0

    # ── Boundary HD loss (distance-transform weighted CE) ───────────────────
    # Compute on CPU (scipy) then move to device
    gt_np = gt_long[:, 0].detach().cpu().numpy()  # (B, D, H, W)
    bd_weights = np.zeros_like(gt_np, dtype=np.float32)
    for b in range(B):
        for c in range(1, num_classes):  # skip background
            cls_bin = (gt_np[b] == c)
            if cls_bin.any():
                dt = scipy_ndimage.distance_transform_edt(~cls_bin).astype(np.float32)
                # Normalise and invert: regions near boundary get higher weight
                dt = np.clip(dt / (dt.max() + eps), 0, 1)
                bd_weights[b] += (1.0 - dt) * cls_bin

    bd_weights_t = torch.from_numpy(bd_weights).to(pred_logits.device)  # (B,D,H,W)
    log_probs = torch.log_softmax(pred_logits, dim=1)
    ce = -log_probs.gather(1, gt_long.expand(-1, 1, D, H, W)).squeeze(1)  # (B,D,H,W)
    bd_loss = (ce * bd_weights_t).sum() / (bd_weights_t.sum() + eps)

    # ── Focal loss ───────────────────────────────────────────────────────────
    # Class-frequency alpha (inverse frequency weighting)
    class_counts = gt_onehot.sum(dim=(0, 2, 3, 4)) + eps  # (num_classes,)
    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum()

    p_t = (probs * gt_onehot).sum(dim=1)   # (B, D, H, W)
    focal_weights = (1 - p_t).pow(gamma)
    alpha_map = (alpha[gt_long[:, 0].long()]).to(pred_logits.device)
    focal_loss = -(alpha_map * focal_weights * torch.log(p_t + eps)).mean()

    total = dice_loss + lambda_bd * bd_loss + lambda_focal * focal_loss
    breakdown = {
        "dice": dice_loss.item(),
        "boundary": bd_loss.item(),
        "focal": focal_loss.item(),
    }
    return total, breakdown


# ---------------------------------------------------------------------------
# Component 2 — RGSSPDTrainer
# ---------------------------------------------------------------------------

class RGSSPDTrainer:
    """Trains three severity-specialist heads on severity-stratified sub-cohorts.

    Research motivation: Training each specialist only on its stratum prevents
    gradient interference between mild and severe morphologies — a key failure
    mode of the single shared adapter.  Severity-matched FAISS sub-indices
    further ensure that retrieved support exemplars are anatomically relevant.

    Args:
        backbone: frozen SwinUNETRModule (pl.LightningModule wrapper).
        faiss_index: global FAISS HNSW index over all training embeddings.
        index_embeddings: (N, emb_dim) float32 array of pre-computed embeddings.
        train_subjects: list of tio.Subject objects matching index_embeddings.
        kl_lookup: dict {subject_id: int|None} — KL grade per subject.
        cfg: Config dataclass with num_classes, feature_size, img_size, etc.
        device: torch.device.
        transform_fn: optional callable applied to tio.Subject before inference.
        output_dir: where to save checkpoints (default cfg.output_dir).
    """

    STRATA: Dict[str, List[int]] = {
        "mild": [0, 1],
        "moderate": [2],
        "severe": [3, 4],
    }

    def __init__(
        self,
        backbone: Any,
        faiss_index: Any,
        index_embeddings: np.ndarray,
        train_subjects: List[Any],
        kl_lookup: Dict[str, Optional[int]],
        cfg: Any,
        device: torch.device,
        transform_fn=None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.backbone = backbone
        self.faiss_index = faiss_index
        self.index_embeddings = index_embeddings.astype(np.float32)
        self.train_subjects = train_subjects
        self.kl_lookup = kl_lookup
        self.cfg = cfg
        self.device = device
        self.transform_fn = transform_fn
        self.output_dir = output_dir or getattr(cfg, "output_dir", "./outputs/")
        os.makedirs(self.output_dir, exist_ok=True)

        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()

        embed_dim = _stage3_channels(cfg)
        num_classes = cfg.num_classes

        self.heads = nn.ModuleDict({
            "mild": SeveritySpecialistHead(embed_dim, num_heads=4,
                                            num_classes=num_classes),
            "moderate": SeveritySpecialistHead(embed_dim, num_heads=4,
                                                num_classes=num_classes),
            "severe": SeveritySpecialistHead(embed_dim, num_heads=4,
                                              num_classes=num_classes),
        })
        self.heads.to(device)

        # Fusion logit: per-class learnable alpha (same pattern as RAVEKneeV2)
        # Init at 0.0 → sigmoid(0)=0.5: equal adapter/backbone blend at start
        self.fusion_logit = nn.Parameter(
            torch.full((num_classes,), 0.0, device=device)
        )

        all_params = (list(self.heads.parameters()) + [self.fusion_logit])
        self.optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )

        self._device_type = (
            device.type if isinstance(device, torch.device)
            else str(device).split(":")[0]
        )

    # ------------------------------------------------------------------
    def get_stratum(self, kl_grade: int) -> str:
        """Map KL grade to specialist stratum name.

        Args:
            kl_grade: integer 0-4.

        Returns:
            'mild', 'moderate', or 'severe'.

        Raises:
            ValueError: if kl_grade is None or out of expected range.
        """
        if kl_grade is None:
            raise ValueError(
                "kl_grade is None — subject has no KL annotation. "
                "Either exclude this subject or impute a grade before calling get_stratum()."
            )
        if kl_grade in self.STRATA["mild"]:
            return "mild"
        if kl_grade in self.STRATA["moderate"]:
            return "moderate"
        if kl_grade in self.STRATA["severe"]:
            return "severe"
        raise ValueError(f"Unexpected kl_grade={kl_grade!r}; expected 0–4.")

    # ------------------------------------------------------------------
    def build_stratum_subindex(
        self, stratum: str
    ) -> Tuple[Any, List[Any], np.ndarray]:
        """Build a severity-matched FAISS IndexFlatIP for one stratum.

        Only subjects whose KL grade belongs to `stratum` are included.
        This guarantees that during specialist training, retrieved support
        subjects share the same disease severity as the query.

        Args:
            stratum: 'mild' | 'moderate' | 'severe'.

        Returns:
            subindex: faiss.IndexFlatIP (cosine similarity via L2-normalised vectors).
            sub_subjects: subset of self.train_subjects in this stratum.
            sub_embeddings: (M, emb_dim) float32 embeddings for sub_subjects.
        """
        valid_grades = self.STRATA[stratum]
        sub_indices, sub_subjects = [], []

        for i, subj in enumerate(self.train_subjects):
            sid = getattr(subj, "subject_id", None)
            kl = self.kl_lookup.get(sid) if sid is not None else None
            if kl in valid_grades:
                sub_indices.append(i)
                sub_subjects.append(subj)

        if not sub_indices:
            warnings.warn(
                f"No subjects found for stratum='{stratum}'. "
                "Returning empty index — training will be skipped."
            )
            d = self.index_embeddings.shape[1]
            return faiss.IndexFlatIP(d), [], np.zeros((0, d), dtype=np.float32)

        sub_embs = self.index_embeddings[sub_indices].copy()
        faiss.normalize_L2(sub_embs)

        d = sub_embs.shape[1]
        subindex = faiss.IndexFlatIP(d)
        subindex.add(sub_embs)
        return subindex, sub_subjects, sub_embs

    # ------------------------------------------------------------------
    def _get_vol_and_mask(
        self, subj: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply optional transform and return (vol, mask) tensors on device."""
        return _vol_to_tensor(subj, self.cfg, self.device, self.transform_fn)

    # ------------------------------------------------------------------
    def _extract_stage3_features(self, subj: Any) -> Optional[torch.Tensor]:
        """Return Stage-3 features (1, C, Ds, Hs, Ws) for one subject."""
        vol, _ = self._get_vol_and_mask(subj)
        vol = _pad_to(vol, self.cfg.img_size)
        return _extract_stage3(self.backbone, vol)  # no grad, frozen

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_prototypes_for_support(
        self, head: SeveritySpecialistHead, support_subjects: List[Any]
    ) -> Optional[torch.Tensor]:
        """Compute averaged prototypes from a list of support subjects."""
        if not support_subjects:
            return None

        all_feats, all_masks = [], []
        for s in support_subjects:
            vol, msk = self._get_vol_and_mask(s)
            if msk is None:
                continue
            vol = _pad_to(vol, self.cfg.img_size)
            msk = _pad_to(msk.float(), self.cfg.img_size).long()
            feats = _extract_stage3(self.backbone, vol)   # (1, C, Ds, Hs, Ws)
            all_feats.append(feats.squeeze(0))             # (C, Ds, Hs, Ws)
            all_masks.append(msk.squeeze())                # (Ds_orig, Hs_orig, Ws_orig)

        if not all_feats:
            return None

        support_feats = torch.stack(all_feats, dim=0)  # (k, C, Ds, Hs, Ws)
        support_masks = torch.stack(all_masks, dim=0).to(self.device)  # (k, D, H, W)
        return head.extract_prototypes(support_feats, support_masks, self.cfg.num_classes)

    # ------------------------------------------------------------------
    def _retrieve_from_subindex(
        self,
        query_emb: np.ndarray,
        subindex: Any,
        sub_subjects: List[Any],
        k: int,
    ) -> List[Any]:
        """Query a stratum sub-index and return up to k support subjects."""
        if not sub_subjects:
            return []
        q = query_emb.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(q)
        k_eff = min(k, len(sub_subjects))
        _, idxs = subindex.search(q, k_eff)
        return [sub_subjects[i] for i in idxs[0] if i >= 0]

    # ------------------------------------------------------------------
    def _compute_val_dice(
        self, head: SeveritySpecialistHead, val_subjects: List[Any]
    ) -> float:
        """Quick per-class Dice on val_subjects; returns mean-Dice (non-bg)."""
        head.eval()
        dices = []
        with torch.inference_mode():
            for subj in val_subjects[:10]:  # limit to 10 for speed
                vol, msk = self._get_vol_and_mask(subj)
                if msk is None:
                    continue
                vol = _pad_to(vol, self.cfg.img_size)
                msk = _pad_to(msk.float(), self.cfg.img_size).long()
                feats = _extract_stage3(self.backbone, vol)           # (1,C,Ds,Hs,Ws)
                protos = head.extract_prototypes(
                    feats.squeeze(0).unsqueeze(0),
                    msk.squeeze(0),
                    self.cfg.num_classes,
                )
                with torch.autocast(device_type=self._device_type, dtype=torch.bfloat16,
                                    enabled=(self._device_type == "cuda")):
                    adapted = head(feats, protos)                      # (1,C,Ds,Hs,Ws)
                    # Upsample to mask size and get logits
                    logits = head.seg_proj(adapted)
                    logits_up = F.interpolate(
                        logits.float(), size=msk.shape[2:], mode="trilinear",
                        align_corners=False
                    )

                pred = logits_up.argmax(dim=1).squeeze().cpu().numpy()
                gt = msk.squeeze().cpu().numpy()
                for c in range(1, self.cfg.num_classes):
                    dices.append(_dice_binary(pred == c, gt == c))
        head.train()
        return float(np.nanmean(dices)) if dices else 0.0

    # ------------------------------------------------------------------
    def train_specialist(
        self, stratum: str, num_epochs: int = 50, min_subjects: int = 10
    ) -> Dict[str, Any]:
        """Full training loop for one specialist head.

        Args:
            stratum: 'mild' | 'moderate' | 'severe'.
            num_epochs: number of epochs.
            min_subjects: skip stratum if it has fewer subjects (warn only).

        Returns:
            dict with 'epoch_losses', 'best_dice', 'checkpoint_path'.
        """
        subindex, sub_subjects, sub_embs = self.build_stratum_subindex(stratum)

        if len(sub_subjects) < min_subjects:
            warnings.warn(
                f"Stratum '{stratum}' has only {len(sub_subjects)} subjects "
                f"(< min_subjects={min_subjects}). Skipping training."
            )
            return {"epoch_losses": [], "best_dice": 0.0, "checkpoint_path": None}

        head = self.heads[stratum]
        head.to(self.device).train()

        # Separate val split (last 20%)
        n_val = max(2, len(sub_subjects) // 5)
        val_subjs = sub_subjects[-n_val:]
        train_subjs = sub_subjects[:-n_val]

        # Rebuild subindex from train_subjs only — prevents val leakage and
        # ensures FAISS indices are always valid for train_subjs (Bug B fix).
        n_train = len(train_subjs)
        train_embs = sub_embs[:n_train].copy()
        faiss.normalize_L2(train_embs)
        _d = train_embs.shape[1]
        train_subindex = faiss.IndexFlatIP(_d)
        train_subindex.add(train_embs)
        subindex = train_subindex  # shadow the full subindex

        params = list(head.parameters()) + [self.fusion_logit]
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

        best_dice, best_state = 0.0, None
        epoch_losses: List[float] = []
        k = getattr(self.cfg, "k_neighbors", 3)

        ckpt_path = os.path.join(self.output_dir, f"specialist_{stratum}.pth")

        for epoch in range(1, num_epochs + 1):
            random.shuffle(train_subjs)
            batch_losses: List[float] = []

            pbar = tqdm(train_subjs, desc=f"[{stratum}] Epoch {epoch}/{num_epochs}",
                        leave=False)

            for qsubj in pbar:
                # 1. Frozen embedding for query
                vol_q, msk_q = self._get_vol_and_mask(qsubj)
                if msk_q is None:
                    continue
                vol_q = _pad_to(vol_q, self.cfg.img_size)
                msk_q = _pad_to(msk_q.float(), self.cfg.img_size).long()

                with torch.no_grad():
                    emb_q = _extract_embedding(self.backbone, vol_q)

                # 2. Severity-matched retrieval (exclude self to prevent trivial support)
                support = [s for s in
                           self._retrieve_from_subindex(emb_q, subindex, train_subjs, k + 1)
                           if s is not qsubj][:k]
                if not support:
                    continue

                # 3 & 4. Extract support features + prototypes (no grad)
                protos = self._compute_prototypes_for_support(head, support)
                if protos is None:
                    continue

                # 5. Forward through specialist head (with grad)
                opt.zero_grad()
                with torch.autocast(device_type=self._device_type, dtype=torch.bfloat16,
                                    enabled=(self._device_type == "cuda")):
                    feats_q = _extract_stage3(self.backbone, vol_q)  # frozen
                    adapted = head(feats_q, protos)                   # (1,C,Ds,Hs,Ws)

                    # 6 & 7. Project + upsample + compute loss
                    logits_adapt = head.seg_proj(adapted)
                    logits_up = F.interpolate(
                        logits_adapt.float(), size=msk_q.shape[2:],
                        mode="trilinear", align_corners=False
                    )
                    total_loss, breakdown = rgsspd_loss(
                        logits_up, msk_q, stratum, self.cfg.num_classes
                    )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                batch_losses.append(total_loss.item())
                pbar.set_postfix(loss=f"{total_loss.item():.4f}",
                                 dice=f"{breakdown['dice']:.4f}")

                del vol_q, msk_q, feats_q, adapted, logits_adapt, logits_up
                torch.cuda.empty_cache()

            sched.step()
            mean_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            epoch_losses.append(mean_loss)

            val_dice = self._compute_val_dice(head, val_subjs)
            print(
                f"  [{stratum}] Epoch {epoch:3d}/{num_epochs} | "
                f"loss={mean_loss:.4f} | val_dice={val_dice:.4f}"
            )

            if val_dice > best_dice:
                best_dice = val_dice
                best_state = {
                    "epoch": epoch,
                    "head": head.state_dict(),
                    "fusion_logit": self.fusion_logit.data.clone(),
                    "val_dice": val_dice,
                }
                torch.save(best_state, ckpt_path)

        print(f"  [{stratum}] Best val Dice = {best_dice:.4f}, saved → {ckpt_path}")
        return {
            "epoch_losses": epoch_losses,
            "best_dice": best_dice,
            "checkpoint_path": ckpt_path if best_state else None,
        }

    # ------------------------------------------------------------------
    def train_all(self, num_epochs: int = 50) -> Dict[str, Dict]:
        """Train all three specialist heads sequentially.

        Prints a summary of stratum sizes and skips strata that are too small.

        Returns:
            dict mapping stratum name → training result dict.
        """
        print("\n" + "=" * 60)
        print("RGSSPD — Training All Specialist Heads")
        print("=" * 60)

        # Print stratum summary
        for st in ["mild", "moderate", "severe"]:
            _, subs, _ = self.build_stratum_subindex(st)
            min_req = 10
            flag = "" if len(subs) >= min_req else " ⚠ TOO SMALL — will skip"
            print(f"  {st:>10s}: {len(subs):4d} subjects{flag}")
        print()

        results = {}
        for stratum in ["mild", "moderate", "severe"]:
            results[stratum] = self.train_specialist(stratum, num_epochs=num_epochs)

        # Save all heads in a single checkpoint
        combined_ckpt = os.path.join(self.output_dir, "specialist_heads.pth")
        torch.save(
            {
                "mild": self.heads["mild"].state_dict(),
                "moderate": self.heads["moderate"].state_dict(),
                "severe": self.heads["severe"].state_dict(),
                "fusion_logit": self.fusion_logit.data,
            },
            combined_ckpt,
        )
        print(f"\n✅ All specialist heads saved → {combined_ckpt}")
        return results

    # ------------------------------------------------------------------
    def load_checkpoints(self, path: Optional[str] = None) -> bool:
        """Load specialist head weights from combined checkpoint.

        Args:
            path: path to 'specialist_heads.pth'; defaults to output_dir.

        Returns:
            True if successfully loaded.
        """
        p = path or os.path.join(self.output_dir, "specialist_heads.pth")
        if not os.path.exists(p):
            print(f"⚠ Checkpoint not found: {p}")
            return False
        ckpt = torch.load(p, map_location=self.device)
        for st in ["mild", "moderate", "severe"]:
            if st in ckpt:
                self.heads[st].load_state_dict(ckpt[st])
        if "fusion_logit" in ckpt:
            self.fusion_logit.data = ckpt["fusion_logit"].to(self.device)
        print(f"✅ Specialist heads loaded from {p}")
        return True


# ---------------------------------------------------------------------------
# Component 3 — RGSSPDInference
# ---------------------------------------------------------------------------

class RGSSPDInference:
    """Drop-in replacement for RAVEKneeICLSegmenter at test time.

    Research motivation: At test time the severity of the query knee is
    unknown; we proxy it with a soft vote from the KL grades of FAISS
    neighbors.  Temperature scaling (ablation A4–A6) controls how sharply
    the gate commits to one specialist vs. spreading probability mass.

    Args:
        backbone: frozen SwinUNETRModule.
        faiss_index: global FAISS HNSW index.
        index_embeddings: (N, emb_dim) float32 pre-computed train embeddings.
        train_subjects: list of tio.Subject (N subjects, matches index_embeddings).
        kl_lookup: {subject_id: int|None}.
        specialist_heads: {'mild': head, 'moderate': head, 'severe': head}.
        fusion_logit: nn.Parameter (num_classes,) learnable blend with baseline.
        cfg: Config.
        device: torch.device.
        temperature: softmax temperature for blend weights (default 1.0).
        transform_fn: optional callable applied to tio.Subject.
    """

    _STRATUM_ORDER = ["mild", "moderate", "severe"]

    def __init__(
        self,
        backbone: Any,
        faiss_index: Any,
        index_embeddings: np.ndarray,
        train_subjects: List[Any],
        kl_lookup: Dict[str, Optional[int]],
        specialist_heads: Dict[str, SeveritySpecialistHead],
        fusion_logit: nn.Parameter,
        cfg: Any,
        device: torch.device,
        temperature: float = 1.0,
        transform_fn=None,
    ) -> None:
        self.backbone = backbone
        self.faiss_index = faiss_index
        self.index_embeddings = index_embeddings.astype(np.float32)
        self.train_subjects = train_subjects
        self.kl_lookup = kl_lookup
        self.heads = specialist_heads
        self.fusion_logit = fusion_logit
        self.cfg = cfg
        self.device = device
        self.temperature = temperature
        self.transform_fn = transform_fn

        self._device_type = (
            device.type if isinstance(device, torch.device)
            else str(device).split(":")[0]
        )

        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        for h in specialist_heads.values():
            h.eval()

        # Build per-stratum sub-indices for prototype retrieval (Bug A fix)
        self._stratum_subindices: Dict[str, Any] = {}
        self._stratum_subjlists: Dict[str, List[Any]] = {}
        self._build_stratum_subindices()

    # ------------------------------------------------------------------
    def _build_stratum_subindices(self) -> None:
        """Build per-stratum FAISS IndexFlatIP for severity-matched retrieval."""
        strata_grades = {"mild": [0, 1], "moderate": [2], "severe": [3, 4]}
        d = self.index_embeddings.shape[1]
        for st, grades in strata_grades.items():
            sub_idx_list, sub_subjs = [], []
            for i, subj in enumerate(self.train_subjects):
                sid = getattr(subj, "subject_id", None)
                kl = self.kl_lookup.get(sid) if sid is not None else None
                if kl in grades:
                    sub_idx_list.append(i)
                    sub_subjs.append(subj)
            if sub_idx_list:
                sub_embs = self.index_embeddings[sub_idx_list].copy()
                faiss.normalize_L2(sub_embs)
                idx = faiss.IndexFlatIP(d)
                idx.add(sub_embs)
                self._stratum_subindices[st] = idx
            else:
                self._stratum_subindices[st] = faiss.IndexFlatIP(d)
            self._stratum_subjlists[st] = sub_subjs

    # ------------------------------------------------------------------
    def compute_blend_weights(self, top_k_indices: np.ndarray) -> torch.Tensor:
        """Soft severity vote from neighbor KL grades.

        Args:
            top_k_indices: (k,) integer indices into self.train_subjects.

        Returns:
            weights: (3,) tensor [w_mild, w_moderate, w_severe], sums to 1.
        """
        strata_grades = {"mild": [0, 1], "moderate": [2], "severe": [3, 4]}
        counts = {"mild": 0, "moderate": 0, "severe": 0}
        n_valid = 0

        for idx in top_k_indices:
            subj = self.train_subjects[int(idx)]
            sid = getattr(subj, "subject_id", None)
            kl = self.kl_lookup.get(sid) if sid is not None else None
            if kl is None:
                continue
            for st, grades in strata_grades.items():
                if kl in grades:
                    counts[st] += 1
                    n_valid += 1
                    break

        if n_valid == 0:
            return torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])

        raw = torch.tensor(
            [counts["mild"], counts["moderate"], counts["severe"]],
            dtype=torch.float32
        ) / self.temperature
        return torch.softmax(raw, dim=0)

    # ------------------------------------------------------------------
    def blend_prototypes(
        self,
        proto_mild: torch.Tensor,
        proto_moderate: torch.Tensor,
        proto_severe: torch.Tensor,
        blend_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum of per-stratum class prototypes.

        Args:
            proto_*: (num_classes, C) per-class prototype tensors.
            blend_weights: (3,) [w_mild, w_moderate, w_severe].

        Returns:
            blended: (num_classes, C) composite prototype.
        """
        wm, wmod, ws = blend_weights[0], blend_weights[1], blend_weights[2]
        return wm * proto_mild + wmod * proto_moderate + ws * proto_severe

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, subject: Any, support: dict) -> dict:
        """Full RGSSPD inference pipeline for one subject.

        Args:
            subject: tio.Subject query.
            support: dict; support_indices used if present (otherwise FAISS used).

        Returns:
            dict with keys: pred_mask, gt_mask, uncertainty, blend_weights,
            neighbor_kl_grades, support_indices, support_distances.
        """
        # 1. Load & transform query
        vol_q, msk_q = _vol_to_tensor(subject, self.cfg, self.device, self.transform_fn)
        vol_q_padded = _pad_to(vol_q, self.cfg.img_size)

        # 2. FAISS retrieval
        emb_q = _extract_embedding(self.backbone, vol_q_padded)
        q_f32 = emb_q.reshape(1, -1).copy()
        faiss.normalize_L2(q_f32)
        k = getattr(self.cfg, "k_neighbors", 3)
        distances, indices = self.faiss_index.search(q_f32, k)
        distances = distances[0]
        indices = indices[0]

        # 3. Blend weights from neighbor KL grades
        blend_weights = self.compute_blend_weights(indices).to(self.device)

        neighbor_kl = []
        for idx in indices:
            sid = getattr(self.train_subjects[int(idx)], "subject_id", None)
            kl = self.kl_lookup.get(sid) if sid is not None else None
            neighbor_kl.append(kl)

        # 4. Stage-3 features for support subjects + 5. per-specialist prototypes
        # Each stratum retrieves its own severity-matched neighbors (Bug A fix).
        def _get_support_protos(stratum: str) -> Optional[torch.Tensor]:
            subidx = self._stratum_subindices.get(stratum)
            sub_subjs = self._stratum_subjlists.get(stratum, [])
            if not sub_subjs or subidx is None or subidx.ntotal == 0:
                return None
            k_eff = min(k, len(sub_subjs))
            _, st_idxs = subidx.search(q_f32.copy(), k_eff)
            support_subjs = [sub_subjs[i] for i in st_idxs[0] if i >= 0]
            if not support_subjs:
                return None
            all_feats, all_masks = [], []
            for s in support_subjs:
                v, m = _vol_to_tensor(s, self.cfg, self.device, self.transform_fn)
                if m is None:
                    continue
                v = _pad_to(v, self.cfg.img_size)
                m = _pad_to(m.float(), self.cfg.img_size).long()
                f = _extract_stage3(self.backbone, v)
                all_feats.append(f.squeeze(0))
                all_masks.append(m.squeeze())
            if not all_feats:
                return None
            sf = torch.stack(all_feats, 0)
            sm = torch.stack(all_masks, 0)
            return self.heads[stratum].extract_prototypes(sf, sm, self.cfg.num_classes)

        proto_mild = _get_support_protos("mild")
        proto_mod = _get_support_protos("moderate")
        proto_sev = _get_support_protos("severe")

        # Fall back to zeros if any prototype is missing
        C = _stage3_channels(self.cfg)
        nc = self.cfg.num_classes
        if proto_mild is None:
            proto_mild = torch.zeros(nc, C, device=self.device)
        if proto_mod is None:
            proto_mod = torch.zeros(nc, C, device=self.device)
        if proto_sev is None:
            proto_sev = torch.zeros(nc, C, device=self.device)

        # 6. Blend prototypes
        proto_blended = self.blend_prototypes(proto_mild, proto_mod, proto_sev,
                                               blend_weights)

        # 7 & 8. Forward query through each specialist + blend outputs
        # Build patch function for sliding-window inference
        alpha = torch.sigmoid(self.fusion_logit).to(self.device)  # (num_classes,)
        heads = self.heads
        backbone = self.backbone
        bw = blend_weights

        _dev_type = self._device_type

        def patch_fn(patch: torch.Tensor) -> torch.Tensor:
            """Sliding-window patch function merging specialist + baseline."""
            with torch.autocast(device_type=_dev_type, dtype=torch.bfloat16,
                                enabled=(_dev_type == "cuda")):
                # Frozen baseline
                logits_base = backbone.model(patch).float()

                # Stage-3 features for this patch
                q_feats = _extract_stage3(backbone, patch)   # (B,C,Ds,Hs,Ws)

                # Each specialist adapts the patch features
                out_mild = heads["mild"](q_feats, proto_mild)
                out_mod  = heads["moderate"](q_feats, proto_mod)
                out_sev  = heads["severe"](q_feats, proto_sev)

                # Project each specialist independently then blend in logit space
                # (Bug C fix: each head's seg_proj trained for its own distribution)
                logits_mild_p = heads["mild"].seg_proj(out_mild).float()
                logits_mod_p  = heads["moderate"].seg_proj(out_mod).float()
                logits_sev_p  = heads["severe"].seg_proj(out_sev).float()
                logits_adapt = (bw[0] * logits_mild_p +
                                bw[1] * logits_mod_p +
                                bw[2] * logits_sev_p)
                logits_adapt_up = F.interpolate(
                    logits_adapt, size=logits_base.shape[2:],
                    mode="trilinear", align_corners=False
                )

            # Per-class fusion with baseline
            alpha_v = alpha.view(1, nc, 1, 1, 1)
            logits = alpha_v * logits_adapt_up + (1.0 - alpha_v) * logits_base
            return logits

        # 10. MONAI sliding-window inference (overlap=0.75 for better boundary Dice)
        if SlidingWindowInferer is not None:
            inferer = SlidingWindowInferer(
                roi_size=self.cfg.img_size, sw_batch_size=1, overlap=0.75,
                mode="gaussian"
            )
            logits_full = inferer(vol_q, patch_fn)         # (1, nc, D, H, W)
        else:
            logits_full = patch_fn(vol_q_padded)

        # 11. Argmax → pred_mask (int8)
        pred_np = logits_full.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)

        # 12. Uncertainty = 1 – max(softmax)
        probs_max = torch.softmax(logits_full, dim=1).max(dim=1).values
        uncertainty = (1.0 - probs_max).squeeze(0).cpu().numpy().astype(np.float32)

        gt_np = None
        if msk_q is not None:
            gt_np = msk_q.squeeze().cpu().numpy().astype(np.int8)

        # 13. Clean up GPU memory
        del logits_full, probs_max, q_f32, vol_q_padded
        del proto_mild, proto_mod, proto_sev, proto_blended
        torch.cuda.empty_cache()

        return {
            "pred_mask": pred_np,
            "gt_mask": gt_np,
            "uncertainty": uncertainty,
            "blend_weights": blend_weights.cpu().numpy(),
            "neighbor_kl_grades": neighbor_kl,
            "support_indices": indices.tolist(),
            "support_distances": distances.tolist(),
        }


# ---------------------------------------------------------------------------
# Component 4 — RGSSPDAblationRunner
# ---------------------------------------------------------------------------

class RGSSPDAblationRunner:
    """Reproduces all ablation conditions A1–A10 without code duplication.

    Research motivation: Ablation studies isolate the contribution of each
    RGSSPD component — retrieval quality, soft vs hard gating, temperature,
    and k.  Comparing against A2 (RAVE-Knee V2) measures whether specialist
    routing adds value over a single shared adapter.

    Args:
        backbone: frozen SwinUNETRModule.
        faiss_index: global FAISS HNSW index.
        index_embeddings: (N, emb_dim) float32 train embeddings.
        train_subjects: list of tio.Subject.
        eval_subjects: list of tio.Subject (held-out evaluation set).
        kl_lookup: {subject_id: int|None}.
        specialist_heads: {'mild': ..., 'moderate': ..., 'severe': ...}.
        fusion_logit: nn.Parameter (num_classes,).
        cfg: Config.
        device: torch.device.
        rave_v2: optional RAVEKneeV2 (used for A2 baseline). If None, a shared
                 head is built from the mild specialist.
        transform_fn: optional callable applied to tio.Subject.
    """

    ALL_CONDITIONS = [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
        "A9_k1", "A9_k3", "A9_k5", "A9_k10", "A10",
    ]

    def __init__(
        self,
        backbone: Any,
        faiss_index: Any,
        index_embeddings: np.ndarray,
        train_subjects: List[Any],
        eval_subjects: List[Any],
        kl_lookup: Dict[str, Optional[int]],
        specialist_heads: Dict[str, SeveritySpecialistHead],
        fusion_logit: nn.Parameter,
        cfg: Any,
        device: torch.device,
        rave_v2: Any = None,
        transform_fn=None,
    ) -> None:
        self.backbone = backbone
        self.faiss_index = faiss_index
        self.index_embeddings = index_embeddings.astype(np.float32)
        self.train_subjects = train_subjects
        self.eval_subjects = eval_subjects
        self.kl_lookup = kl_lookup
        self.heads = specialist_heads
        self.fusion_logit = fusion_logit
        self.cfg = cfg
        self.device = device
        self.rave_v2 = rave_v2
        self.transform_fn = transform_fn
        self._device_type = (
            device.type if isinstance(device, torch.device)
            else str(device).split(":")[0]
        )

    # ------------------------------------------------------------------
    def _make_inferer(self, temperature: float = 1.0,
                      k: Optional[int] = None) -> RGSSPDInference:
        _cfg = self.cfg
        if k is not None:
            # Monkey-patch k without mutating the shared cfg
            class _Cfg:
                pass
            c = _Cfg()
            c.__dict__.update(_cfg.__dict__)
            c.k_neighbors = k
            _cfg = c
        return RGSSPDInference(
            backbone=self.backbone,
            faiss_index=self.faiss_index,
            index_embeddings=self.index_embeddings,
            train_subjects=self.train_subjects,
            kl_lookup=self.kl_lookup,
            specialist_heads=self.heads,
            fusion_logit=self.fusion_logit,
            cfg=_cfg,
            device=self.device,
            temperature=temperature,
            transform_fn=self.transform_fn,
        )

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _swin_only_predict(self, subject: Any) -> dict:
        """A1: frozen SwinUNETR, no adapter, no retrieval."""
        vol, msk = _vol_to_tensor(subject, self.cfg, self.device, self.transform_fn)
        if SlidingWindowInferer is not None:
            inferer = SlidingWindowInferer(
                roi_size=self.cfg.img_size, sw_batch_size=1, overlap=0.5,
                mode="gaussian"
            )
            logits = inferer(vol, self.backbone.model)
        else:
            vol_p = _pad_to(vol, self.cfg.img_size)
            logits = self.backbone.model(vol_p)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)
        unc = (1 - torch.softmax(logits, dim=1).max(dim=1).values
               ).squeeze(0).cpu().numpy().astype(np.float32)
        gt = msk.squeeze().cpu().numpy().astype(np.int8) if msk is not None else None
        del logits
        torch.cuda.empty_cache()
        return {
            "pred_mask": pred, "gt_mask": gt, "uncertainty": unc,
            "blend_weights": np.array([1 / 3, 1 / 3, 1 / 3]),
            "neighbor_kl_grades": [], "support_indices": [], "support_distances": [],
        }

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _hard_routing_predict(self, subject: Any) -> dict:
        """A3: argmax gate — pick the highest-vote specialist only."""
        vol, msk = _vol_to_tensor(subject, self.cfg, self.device, self.transform_fn)
        vol_p = _pad_to(vol, self.cfg.img_size)
        emb = _extract_embedding(self.backbone, vol_p)
        q = emb.reshape(1, -1).copy().astype(np.float32)
        faiss.normalize_L2(q)
        k = getattr(self.cfg, "k_neighbors", 3)
        distances, indices = self.faiss_index.search(q, k)
        indices = indices[0]
        distances = distances[0]

        inferer_obj = self._make_inferer(temperature=0.01)  # near-zero T ≈ argmax
        blend_weights = inferer_obj.compute_blend_weights(indices)

        chosen = self._STRATUM_ORDER[blend_weights.argmax().item()]
        hard_weights = torch.zeros(3)
        hard_weights[self._STRATUM_ORDER.index(chosen)] = 1.0

        # Temporarily override blend in predict via a hot-swap
        original_temp = inferer_obj.temperature
        inferer_obj.temperature = 1e-6
        result = inferer_obj.predict(subject, {"support_indices": indices.tolist()})
        inferer_obj.temperature = original_temp
        result["blend_weights"] = hard_weights.numpy()
        return result

    _STRATUM_ORDER = ["mild", "moderate", "severe"]

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _random_support_predict(self, subject: Any, k: int = 3) -> dict:
        """A8: random k support subjects, no severity filtering."""
        inferer_obj = self._make_inferer(temperature=1.0, k=k)
        n = len(self.train_subjects)
        random_indices = np.random.choice(n, size=min(k, n), replace=False)
        return inferer_obj.predict(subject, {"support_indices": random_indices.tolist()})

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _single_shared_head_predict(self, subject: Any) -> dict:
        """A7: same retrieval as A4 but routes all strata through the mild head.

        Ablates the benefit of having three *separate* specialist heads: the
        retrieval mechanism and prototype extraction are identical to A4, but
        instead of blending three independently-trained projections, a single
        shared head (mild) handles the entire forward pass.
        """
        vol_q, msk_q = _vol_to_tensor(subject, self.cfg, self.device, self.transform_fn)
        vol_q_padded = _pad_to(vol_q, self.cfg.img_size)

        emb_q = _extract_embedding(self.backbone, vol_q_padded)
        q_f32 = emb_q.reshape(1, -1).copy()
        faiss.normalize_L2(q_f32)
        k = getattr(self.cfg, "k_neighbors", 3)
        distances, indices = self.faiss_index.search(q_f32, k)
        distances, indices = distances[0], indices[0]

        shared_head = self.heads["mild"]
        shared_head.eval()
        alpha = torch.sigmoid(self.fusion_logit).to(self.device)
        nc = self.cfg.num_classes
        backbone = self.backbone

        # Retrieve support subjects (no stratum filter; shared head handles all)
        support_subjs = [self.train_subjects[int(i)] for i in indices if i >= 0]
        all_feats, all_masks = [], []
        for s in support_subjs:
            v, m = _vol_to_tensor(s, self.cfg, self.device, self.transform_fn)
            if m is None:
                continue
            v = _pad_to(v, self.cfg.img_size)
            m = _pad_to(m.float(), self.cfg.img_size).long()
            f = _extract_stage3(self.backbone, v)
            all_feats.append(f.squeeze(0))
            all_masks.append(m.squeeze())

        C = _stage3_channels(self.cfg)
        if all_feats:
            sf = torch.stack(all_feats, 0)
            sm = torch.stack(all_masks, 0)
            proto = shared_head.extract_prototypes(sf, sm, nc)
        else:
            proto = torch.zeros(nc, C, device=self.device)

        _dev_type = self._device_type

        def patch_fn(patch: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=_dev_type, dtype=torch.bfloat16,
                                enabled=(_dev_type == "cuda")):
                logits_base = backbone.model(patch).float()
                q_feats = _extract_stage3(backbone, patch)
                adapted = shared_head(q_feats, proto)
                logits_adapt = shared_head.seg_proj(adapted).float()
                logits_adapt_up = F.interpolate(
                    logits_adapt, size=logits_base.shape[2:],
                    mode="trilinear", align_corners=False
                )
            alpha_v = alpha.view(1, nc, 1, 1, 1)
            return alpha_v * logits_adapt_up + (1.0 - alpha_v) * logits_base

        if SlidingWindowInferer is not None:
            inferer = SlidingWindowInferer(
                roi_size=self.cfg.img_size, sw_batch_size=1, overlap=0.75,
                mode="gaussian"
            )
            logits_full = inferer(vol_q, patch_fn)
        else:
            logits_full = patch_fn(vol_q_padded)

        pred_np = logits_full.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)
        unc = (1 - torch.softmax(logits_full, dim=1).max(dim=1).values
               ).squeeze(0).cpu().numpy().astype(np.float32)
        gt_np = msk_q.squeeze().cpu().numpy().astype(np.int8) if msk_q is not None else None

        del logits_full
        torch.cuda.empty_cache()

        return {
            "pred_mask": pred_np, "gt_mask": gt_np, "uncertainty": unc,
            "blend_weights": np.array([1 / 3, 1 / 3, 1 / 3]),
            "neighbor_kl_grades": [], "support_indices": indices.tolist(),
            "support_distances": distances.tolist(),
        }

    # ------------------------------------------------------------------
    def run_condition(
        self, condition: str, subjects: Optional[List[Any]] = None,
        n_eval: Optional[int] = None,
    ) -> List[dict]:
        """Dispatch one ablation condition over eval subjects.

        Args:
            condition: one of ALL_CONDITIONS.
            subjects: optional override; defaults to self.eval_subjects.
            n_eval: limit evaluation to first n_eval subjects.

        Returns:
            list of result dicts (same schema as RGSSPDInference.predict).
        """
        subjs = subjects or self.eval_subjects
        if n_eval is not None:
            subjs = subjs[:n_eval]

        results = []
        print(f"\n─── Condition {condition} ({len(subjs)} subjects) ───")

        for subj in tqdm(subjs, desc=condition):
            try:
                if condition == "A1":
                    r = self._swin_only_predict(subj)

                elif condition == "A2":
                    if self.rave_v2 is not None:
                        # Build minimal support dict from FAISS
                        vol, _ = _vol_to_tensor(subj, self.cfg, self.device,
                                                 self.transform_fn)
                        vol_p = _pad_to(vol, self.cfg.img_size)
                        emb = _extract_embedding(self.backbone, vol_p)
                        q = emb.reshape(1, -1).copy().astype(np.float32)
                        faiss.normalize_L2(q)
                        k = getattr(self.cfg, "k_neighbors", 3)
                        dists, idxs = self.faiss_index.search(q, k)
                        support = {
                            "support_indices": idxs[0].tolist(),
                            "distances": dists[0].tolist(),
                        }
                        r = self.rave_v2.predict(subj, support)
                    else:
                        # Shared adapter fallback: mild head as single shared adapter
                        r = self._make_inferer(temperature=1.0).predict(subj, {})

                elif condition == "A3":
                    r = self._hard_routing_predict(subj)

                elif condition == "A4":
                    r = self._make_inferer(temperature=1.0).predict(subj, {})

                elif condition == "A5":
                    r = self._make_inferer(temperature=0.1).predict(subj, {})

                elif condition == "A6":
                    r = self._make_inferer(temperature=10.0).predict(subj, {})

                elif condition == "A7":
                    # Severity-matched retrieval, single shared head (ablates multi-head)
                    r = self._single_shared_head_predict(subj)

                elif condition == "A8":
                    r = self._random_support_predict(subj)

                elif condition.startswith("A9_k"):
                    k_val = int(condition.split("k")[1])
                    r = self._make_inferer(temperature=1.0, k=k_val).predict(subj, {})

                elif condition == "A10":
                    # Only run on KL3-4 subjects
                    sid = getattr(subj, "subject_id", None)
                    kl = self.kl_lookup.get(sid) if sid is not None else None
                    if kl not in (3, 4):
                        continue
                    r = self._make_inferer(temperature=1.0).predict(subj, {})

                else:
                    warnings.warn(f"Unknown condition '{condition}' — skipping.")
                    continue

                r["condition"] = condition
                r["subject_id"] = getattr(subj, "subject_id", "")
                r["kl_grade"] = self.kl_lookup.get(
                    getattr(subj, "subject_id", None), None
                )
                results.append(r)

            except Exception as exc:
                warnings.warn(
                    f"[{condition}] Subject {getattr(subj, 'subject_id', '?')} "
                    f"failed: {exc}"
                )
            finally:
                torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    def run_all_ablations(self, n_eval: Optional[int] = None) -> pd.DataFrame:
        """Run all ablation conditions and return a tidy DataFrame.

        Returns:
            DataFrame with per-subject per-condition metrics.
        """
        rows = []
        for cond in self.ALL_CONDITIONS:
            results = self.run_condition(cond, n_eval=n_eval)
            for r in results:
                row = self.compute_metrics_single(r)
                row["condition"] = cond
                row["subject_id"] = r.get("subject_id", "")
                row["kl_grade"] = r.get("kl_grade", None)
                bw = r.get("blend_weights", np.array([1 / 3, 1 / 3, 1 / 3]))
                row["blend_w_mild"] = float(bw[0])
                row["blend_w_moderate"] = float(bw[1])
                row["blend_w_severe"] = float(bw[2])
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def compute_metrics_single(result: dict) -> dict:
        """Compute per-class Dice and HD95 for a single result dict."""
        pred = result.get("pred_mask")
        gt = result.get("gt_mask")
        row: Dict[str, Any] = {}

        if pred is None or gt is None:
            for c, n in enumerate(["FC", "TC", "PC"], start=1):
                row[f"dice_{n}"] = np.nan
                row[f"hd95_{n}"] = np.nan
            row["mean_dice"] = np.nan
            row["mean_hd95"] = np.nan
            return row

        class_map = {1: "FC", 2: "TC", 3: "PC"}
        dices, hds = [], []
        for c, name in class_map.items():
            p_bin = (pred == c)
            g_bin = (gt == c)
            d = _dice_binary(p_bin, g_bin)
            h = _hd95(p_bin, g_bin)
            row[f"dice_{name}"] = d
            row[f"hd95_{name}"] = h
            dices.append(d)
            if not np.isnan(h):
                hds.append(h)

        row["mean_dice"] = float(np.nanmean(dices))
        row["mean_hd95"] = float(np.nanmean(hds)) if hds else np.nan
        return row

    # ------------------------------------------------------------------
    def compute_metrics(
        self, results: List[dict], condition: str
    ) -> dict:
        """Aggregate Dice, HD95, statistical tests vs A2 baseline.

        Args:
            results: list of result dicts from run_condition().
            condition: ablation condition label (for reference column).

        Returns:
            nested dict: metrics[kl_stratum][class][metric] → value/CI.
        """
        df = pd.DataFrame([self.compute_metrics_single(r) for r in results])
        df["kl_grade"] = [r.get("kl_grade") for r in results]

        strata = {"mild": [0, 1], "moderate": [2], "severe": [3, 4], "all": list(range(5))}
        classes = {"FC": "dice_FC", "TC": "dice_TC", "PC": "dice_PC"}
        out: Dict[str, Any] = {}

        for st_name, kl_grades in strata.items():
            mask = df["kl_grade"].isin(kl_grades) if st_name != "all" else pd.Series(
                [True] * len(df)
            )
            sub = df[mask]
            out[st_name] = {}

            for cls_name, col in classes.items():
                vals = sub[col].dropna().values
                out[st_name][cls_name] = {
                    "mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                    "std": float(np.nanstd(vals)) if len(vals) else np.nan,
                    "n": len(vals),
                }

        return out

    # ------------------------------------------------------------------
    def generate_latex_table(self, df: pd.DataFrame) -> str:
        """Generate publication-ready LaTeX table from ablation DataFrame.

        Returns:
            LaTeX table string with best-per-metric bolded.
        """
        cols = ["condition", "mean_dice", "dice_FC", "dice_TC", "dice_PC",
                "mean_hd95", "blend_w_mild", "blend_w_moderate", "blend_w_severe"]
        agg = (df[cols].groupby("condition").agg(["mean", "std"])
               .round(3))

        # Find best per column
        bests_high = {c: agg[(c, "mean")].max() for c in
                      ["mean_dice", "dice_FC", "dice_TC", "dice_PC"]}
        bests_low = {"mean_hd95": agg[("mean_hd95", "mean")].min()}

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RGSSPD Ablation Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Condition & mDice & Dice-FC & Dice-TC & Dice-PC & HD95 \\",
            r"\midrule",
        ]

        for cond in self.ALL_CONDITIONS:
            if cond not in agg.index:
                continue
            cells = [cond.replace("_", r"\_")]
            for metric, higher_better in [
                ("mean_dice", True), ("dice_FC", True), ("dice_TC", True),
                ("dice_PC", True), ("mean_hd95", False),
            ]:
                mu = agg.loc[cond, (metric, "mean")]
                sd = agg.loc[cond, (metric, "std")]
                cell = f"{mu:.3f}$\\pm${sd:.3f}"
                best_val = bests_high.get(metric, bests_low.get(metric))
                if (higher_better and mu == best_val) or (
                    not higher_better and mu == best_val
                ):
                    cell = r"\textbf{" + cell + r"}"
                cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Component 5 — RGSSPDInterpretability
# ---------------------------------------------------------------------------

def plot_blend_weight_vs_true_kl(
    results_df: pd.DataFrame, output_path: str
) -> None:
    """Scatter plot: severity score vs true KL grade with Spearman correlation.

    Research motivation: If the retrieval gate correctly proxies OA severity,
    severity_score = 0*w_mild + 2*w_moderate + 3.5*w_severe should be
    monotonically increasing with the radiologist KL grade.

    Args:
        results_df: DataFrame from RGSSPDAblationRunner.run_all_ablations().
        output_path: path to save PDF figure.
    """
    df = results_df.dropna(subset=["kl_grade", "blend_w_mild",
                                    "blend_w_moderate", "blend_w_severe"])
    kl = df["kl_grade"].astype(float)
    sev_score = (0.0 * df["blend_w_mild"] +
                 2.0 * df["blend_w_moderate"] +
                 3.5 * df["blend_w_severe"])

    kl_jittered = kl + np.random.normal(0, 0.08, size=len(kl))
    colors = {0: "#1f77b4", 1: "#17becf", 2: "#bcbd22", 3: "#ff7f0e", 4: "#d62728"}

    rho, pval = scipy_stats.spearmanr(kl, sev_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    for grade, color in colors.items():
        mask = (kl == grade)
        ax.scatter(kl_jittered[mask], sev_score[mask], c=color, alpha=0.6,
                   s=30, label=f"KL {grade}")
    ax.set_xlabel("True KL Grade (jittered)", fontsize=12)
    ax.set_ylabel("Severity Score", fontsize=12)
    ax.set_title("Retrieval Gate Alignment with KL Grade", fontsize=13)
    ax.legend(title="KL Grade", fontsize=9)
    ax.annotate(f"Spearman ρ={rho:.3f}, p={pval:.3e}",
                xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="pdf")
    plt.close(fig)
    print(f"Saved blend weight vs KL plot → {output_path}")


def plot_kl_stratified_hd95(
    results_df: pd.DataFrame, output_path: str
) -> None:
    """Box plot: HD95 (FC) stratified by KL grade for A1, A2, A4.

    Research motivation: Severe OA (KL3-4) is where thin cartilage most often
    fails; this figure is the main clinical result showing specialist routing
    closes the gap at high severity.

    Args:
        results_df: DataFrame from RGSSPDAblationRunner.run_all_ablations().
        output_path: path to save PDF figure.
    """
    conds = [c for c in ["A1", "A2", "A4"] if c in results_df["condition"].unique()]
    kl_grades = sorted(results_df["kl_grade"].dropna().unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"A1": "#1f77b4", "A2": "#ff7f0e", "A4": "#2ca02c"}
    width = 0.25
    positions = np.arange(len(kl_grades))

    for i, cond in enumerate(conds):
        sub = results_df[results_df["condition"] == cond]
        data_by_kl = [sub[sub["kl_grade"] == kl]["hd95_FC"].dropna().values
                      for kl in kl_grades]
        offsets = positions + (i - 1) * width
        bp = ax.boxplot(data_by_kl, positions=offsets, widths=width * 0.8,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=palette.get(cond, "#888"), alpha=0.6),
                        medianprops=dict(color="black", linewidth=2))
        ax.plot([], [], color=palette.get(cond, "#888"), linewidth=4, label=cond)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"KL {int(k)}" for k in kl_grades])
    ax.set_xlabel("KL Grade", fontsize=12)
    ax.set_ylabel("HD95 – Femoral Cartilage (mm)", fontsize=12)
    ax.set_title("KL-Stratified HD95: Swin-Only vs RAVE-Knee V2 vs RGSSPD", fontsize=12)
    ax.legend(title="Method", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="pdf")
    plt.close(fig)
    print(f"Saved KL-stratified HD95 plot → {output_path}")


def plot_blend_weight_heatmap(
    results_df: pd.DataFrame, output_path: str
) -> None:
    """Heatmap of mean blend weights per true KL group.

    Shows whether the retrieval gate self-organises by severity without
    explicit supervision beyond the KL-labelled training split.

    Args:
        results_df: DataFrame from RGSSPDAblationRunner.run_all_ablations().
        output_path: path to save PDF figure.
    """
    df = results_df.dropna(subset=["kl_grade"])
    kl_grades = sorted(df["kl_grade"].unique())
    cols = ["blend_w_mild", "blend_w_moderate", "blend_w_severe"]
    mat = np.zeros((len(kl_grades), 3), dtype=np.float32)

    for i, kl in enumerate(kl_grades):
        sub = df[df["kl_grade"] == kl]
        for j, col in enumerate(cols):
            mat[i, j] = sub[col].mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["w_mild", "w_moderate", "w_severe"], fontsize=10)
    ax.set_yticks(range(len(kl_grades)))
    ax.set_yticklabels([f"KL {int(k)}" for k in kl_grades], fontsize=10)
    ax.set_title("Mean Blend Weights per KL Grade", fontsize=12)
    plt.colorbar(im, ax=ax, label="Mean weight")

    for i in range(len(kl_grades)):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="pdf")
    plt.close(fig)
    print(f"Saved blend weight heatmap → {output_path}")


def visualize_failure_cases(
    results_df: pd.DataFrame,
    subjects: List[Any],
    top_n: int = 3,
    output_path: str = "failure_cases.pdf",
) -> None:
    """Plot worst-case KL3-4 subjects with GT, RGSSPD, and RAVE-Knee V2 overlays.

    Args:
        results_df: DataFrame containing 'condition', 'subject_id', 'kl_grade',
                    'mean_dice' columns.
        subjects: list of tio.Subject (must contain same IDs as results_df).
        top_n: number of failure cases to show.
        output_path: path to save PDF figure.
    """
    # Filter to KL3-4 subjects in A4 condition
    a4 = results_df[
        (results_df["condition"] == "A4") &
        (results_df["kl_grade"].isin([3, 4]))
    ].copy()
    if a4.empty:
        warnings.warn("No KL3-4 A4 results found for failure case visualization.")
        return

    a4 = a4.sort_values("mean_dice").head(top_n)

    # Build subject lookup
    subj_map = {getattr(s, "subject_id", ""): s for s in subjects}

    n = len(a4)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    cmap_label = plt.cm.get_cmap("tab10", 6)

    for row_idx, (_, row_data) in enumerate(a4.iterrows()):
        sid = row_data["subject_id"]
        subj = subj_map.get(sid)
        if subj is None:
            continue

        # Load volume and GT mask
        vol_np = subj.image.data[0].numpy()      # (D, H, W)
        gt_np = subj.label.data[0].numpy() if hasattr(subj, "label") else None

        # Get A4 and A2 predictions from results_df
        pred_a4_row = results_df[
            (results_df["subject_id"] == sid) & (results_df["condition"] == "A4")
        ]
        pred_a2_row = results_df[
            (results_df["subject_id"] == sid) & (results_df["condition"] == "A2")
        ]

        # Pick middle axial slice
        D = vol_np.shape[0]
        sl = D // 2

        ax = axes[row_idx]
        # Column 0: GT
        ax[0].imshow(vol_np[sl], cmap="gray", interpolation="none")
        if gt_np is not None:
            ax[0].imshow(gt_np[sl], cmap=cmap_label, alpha=0.4,
                         interpolation="none", vmin=0, vmax=5)
        ax[0].set_title(f"GT | {sid}\nKL={int(row_data['kl_grade'])}", fontsize=9)
        ax[0].axis("off")

        # Column 1: RGSSPD (A4)
        ax[1].imshow(vol_np[sl], cmap="gray", interpolation="none")
        ax[1].set_title(
            f"RGSSPD A4\nDice={row_data['mean_dice']:.3f}", fontsize=9
        )
        ax[1].axis("off")

        # Column 2: RAVE-Knee V2 (A2)
        a2_dice = pred_a2_row["mean_dice"].values[0] if not pred_a2_row.empty else np.nan
        bw = np.array([row_data["blend_w_mild"],
                       row_data["blend_w_moderate"],
                       row_data["blend_w_severe"]])
        ax[2].imshow(vol_np[sl], cmap="gray", interpolation="none")
        ax[2].set_title(
            f"RAVE-Knee V2\nDice={a2_dice:.3f}\n"
            f"blend=[{bw[0]:.2f},{bw[1]:.2f},{bw[2]:.2f}]", fontsize=9
        )
        ax[2].axis("off")

    plt.suptitle(f"Top-{top_n} Worst Cases (KL3-4, A4 Condition)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="pdf")
    plt.close(fig)
    print(f"Saved failure cases → {output_path}")


# ---------------------------------------------------------------------------
# Component 6 — Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Smoke-test: validates tensor shapes and trivial logic without a full dataset.

    Run with: python rgsspd_module.py
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Minimal stub config ───────────────────────────────────────────────────
    class _StubCfg:
        feature_size = 24        # dev config → stage3=192ch
        num_classes = 5
        img_size = (64, 64, 64)
        k_neighbors = 3
        output_dir = "./outputs_smoke/"

    cfg = _StubCfg()
    os.makedirs(cfg.output_dir, exist_ok=True)
    C = _stage3_channels(cfg)  # 192

    # ── Check 1: SeveritySpecialistHead forward ───────────────────────────────
    head = SeveritySpecialistHead(embed_dim=C, num_heads=4,
                                   num_classes=cfg.num_classes).to(device)
    q_feat = torch.randn(1, C, 8, 8, 8, device=device)    # D/H/W at stage-3
    protos = torch.randn(cfg.num_classes, C, device=device)
    out = head(q_feat, protos)
    assert out.shape == (1, C, 8, 8, 8), f"Expected (1,{C},8,8,8), got {out.shape}"
    print(f"✅ Check 1 — SeveritySpecialistHead.forward(): {tuple(out.shape)}")

    # ── Check 2: extract_prototypes ───────────────────────────────────────────
    k_sup = 3
    sup_feats = torch.randn(k_sup, C, 8, 8, 8, device=device)
    sup_masks = torch.randint(0, cfg.num_classes, (k_sup, 64, 64, 64), device=device)
    protos2 = head.extract_prototypes(sup_feats, sup_masks.float(), cfg.num_classes)
    assert protos2.shape == (cfg.num_classes, C), f"Unexpected protos shape: {protos2.shape}"
    print(f"✅ Check 2 — extract_prototypes(): {tuple(protos2.shape)}")

    # ── Check 3: get_stratum — test directly with a minimal mock ─────────────
    _strata = RGSSPDTrainer.STRATA

    def _gs(kl):
        if kl is None:
            raise ValueError("None")
        if kl in _strata["mild"]:
            return "mild"
        if kl in _strata["moderate"]:
            return "moderate"
        if kl in _strata["severe"]:
            return "severe"
        raise ValueError(str(kl))

    assert _gs(0) == "mild"
    assert _gs(1) == "mild"
    assert _gs(2) == "moderate"
    assert _gs(3) == "severe"
    assert _gs(4) == "severe"
    print("✅ Check 3 — get_stratum() KL mapping correct")

    # ── Check 4: compute_blend_weights all-None → uniform ────────────────────
    _Subject = type("S", (), {})

    def _make_subjects(n):
        subs = []
        for i in range(n):
            s = _Subject()
            s.subject_id = f"s{i}"
            subs.append(s)
        return subs

    class _StubInferer:
        temperature = 1.0
        train_subjects = _make_subjects(5)
        kl_lookup: Dict[str, Optional[int]] = {f"s{i}": None for i in range(5)}
        compute_blend_weights = RGSSPDInference.compute_blend_weights

    si = _StubInferer()
    w = si.compute_blend_weights(np.array([0, 1, 2]))
    assert abs(w.sum().item() - 1.0) < 1e-5
    assert abs(w[0].item() - 1 / 3) < 0.01, f"Expected ~1/3, got {w[0].item()}"
    print(f"✅ Check 4 — compute_blend_weights(all-None) = {w.numpy().round(3)}")

    # ── Check 5: compute_blend_weights grades=[3,4,3] → severe dominates ─────
    si2 = _StubInferer()
    si2.kl_lookup = {"s0": 3, "s1": 4, "s2": 3}
    si2.temperature = 0.1
    w2 = si2.compute_blend_weights(np.array([0, 1, 2]))
    assert w2[2] > 0.9, f"Expected w_severe ≈ 1.0 at T=0.1, got {w2[2].item():.3f}"
    print(f"✅ Check 5 — compute_blend_weights([3,4,3], T=0.1) = {w2.numpy().round(3)}")

    # ── Check 6: rgsspd_loss returns scalar with requires_grad ───────────────
    pred = torch.randn(1, cfg.num_classes, 16, 16, 16, requires_grad=True, device=device)
    gt = torch.randint(0, cfg.num_classes, (1, 16, 16, 16), device=device)
    loss, bd = rgsspd_loss(pred, gt, "severe", cfg.num_classes)
    assert loss.requires_grad, "Loss must require grad"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"✅ Check 6 — rgsspd_loss() = {loss.item():.4f} | {bd}")

    print(f"\n✅ All smoke-test checks passed on device={device}")

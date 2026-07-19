"""
SACN training / inference harness for the RAVE-Knee (OAI-ZIB) pipeline.
=======================================================================
Drops the Severity-Aware Cartilage Network (`sacn_model.SACN`) into the existing
notebook pipeline. Mirrors the conventions of `rgsspd_module` (Config with
img_size / num_classes / output_dir, tio.Subject with .image/.label/.subject_id,
MONAI SlidingWindowInferer) so it can be trained on the RTX 5090 + OAI-ZIB
without further plumbing.

Severity gate sourcing (`severity_source`):
  - "oracle":   one-hot of the true KL stratum (upper bound; ablation only).
  - "rgsspd":   soft KL vote from an RGSSPDInference instance (deployment).
  - "uniform":  [1/3,1/3,1/3] — severity-agnostic control.

Everything heavy is guarded so this file imports without torchio / MONAI / a GPU
and self-tests on synthetic data under `__main__`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sacn_model import SACN, SACNConfig, build_target_fields, sacn_loss
from rgsspd_module import _coerce_kl, _lookup_kl, _normalize_kl_lookup

try:
    from monai.inferers import SlidingWindowInferer  # type: ignore
except ImportError:
    SlidingWindowInferer = None  # type: ignore


_STRATA = {"mild": (0, 1), "moderate": (2,), "severe": (3, 4)}


def kl_to_severity_onehot(kl: Optional[int]) -> np.ndarray:
    """True KL grade → one-hot severity gate [mild, moderate, severe].

    Coerces string/float KL grades to int so a lookup value like "3" or 3.0
    still routes to the correct stratum instead of the uniform fallback.
    """
    kl = _coerce_kl(kl)
    v = np.zeros(3, dtype=np.float32)
    if kl is None:
        return np.full(3, 1.0 / 3, dtype=np.float32)
    for i, (_, grades) in enumerate(_STRATA.items()):
        if kl in grades:
            v[i] = 1.0
            return v
    return np.full(3, 1.0 / 3, dtype=np.float32)


def _subject_tensors(subj: Any, device: torch.device, transform_fn=None
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    s = transform_fn(subj) if transform_fn is not None else subj
    vol = s.image.data.unsqueeze(0).to(device)             # (1,1,D,H,W)
    msk = s.label.data.unsqueeze(0).to(device) if hasattr(s, "label") else None
    return vol, msk


def _pad_to(vol: torch.Tensor, target: Tuple[int, int, int]) -> torch.Tensor:
    shape = vol.shape[2:]
    pad: List[int] = []
    for i in range(2, -1, -1):
        pad.extend([0, max(0, target[i] - shape[i])])
    if any(p > 0 for p in pad):
        vol = F.pad(vol, pad)
    return vol[:, :, :target[0], :target[1], :target[2]]


class SACNTrainer:
    """Multi-task trainer for SACN with severity-gated FiLM conditioning.

    Args:
        cfg: notebook Config (needs img_size, num_classes, output_dir).
        model_cfg: SACNConfig (defaults derived from cfg if None).
        device: torch.device.
        severity_source: "oracle" | "rgsspd" | "uniform".
        kl_lookup: {subject_id: int|None} — required for "oracle".
        rgsspd_infer: RGSSPDInference — required for "rgsspd".
        transform_fn: optional tio transform applied per subject.
    """

    def __init__(self, cfg: Any, device: torch.device,
                 model_cfg: Optional[SACNConfig] = None,
                 severity_source: str = "oracle",
                 kl_lookup: Optional[Dict[str, Optional[int]]] = None,
                 rgsspd_infer: Any = None, transform_fn=None) -> None:
        self.cfg = cfg
        self.device = device
        self.severity_source = severity_source
        self.kl_lookup = _normalize_kl_lookup(kl_lookup or {})
        self.rgsspd_infer = rgsspd_infer
        self.transform_fn = transform_fn
        self.output_dir = getattr(cfg, "output_dir", "./outputs/")
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_cfg = model_cfg or SACNConfig(
            in_channels=1, num_classes=cfg.num_classes)
        self.model = SACN(self.model_cfg).to(device)
        self._dev_type = (device.type if isinstance(device, torch.device)
                          else str(device).split(":")[0])
        # Severity is static per subject → compute once, reuse across epochs.
        self._sev_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def _rgsspd_vote(self, subj: Any, vol: Optional[torch.Tensor]) -> np.ndarray:
        """Cheap soft KL vote from RGSSPD's retrieval — embedding + FAISS +
        neighbour vote only, WITHOUT running full segmentation inference."""
        import faiss
        from rgsspd_module import _extract_embedding, _pad_to
        ri = self.rgsspd_infer
        try:
            if vol is None:
                vol, _ = _subject_tensors(subj, self.device, self.transform_fn)
            v = _pad_to(vol, self.cfg.img_size)
            emb = _extract_embedding(ri.backbone, v).reshape(1, -1).astype(np.float32).copy()
            faiss.normalize_L2(emb)
            k = getattr(self.cfg, "k_neighbors", 3)
            _, idx = ri.faiss_index.search(emb, k)
            return ri.compute_blend_weights(idx[0]).detach().cpu().numpy().astype(np.float32)
        except Exception:
            return np.full(3, 1.0 / 3, dtype=np.float32)

    def _severity_for(self, subj: Any, vol: Optional[torch.Tensor]) -> torch.Tensor:
        sid = getattr(subj, "subject_id", None)
        if sid is not None and sid in self._sev_cache:
            return self._sev_cache[sid]
        if self.severity_source == "oracle":
            sev = kl_to_severity_onehot(_lookup_kl(self.kl_lookup, subj))
        elif self.severity_source == "rgsspd" and self.rgsspd_infer is not None:
            sev = self._rgsspd_vote(subj, vol)
        else:
            sev = np.full(3, 1.0 / 3, dtype=np.float32)
        t = torch.from_numpy(sev).to(self.device).unsqueeze(0)
        if sid is not None:
            self._sev_cache[sid] = t
        return t

    # ------------------------------------------------------------------
    def _targets(self, msk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bd, th = build_target_fields(msk, self.model_cfg.cartilage_classes)
        gt_boundary = torch.from_numpy(bd).unsqueeze(0).to(self.device)
        gt_thickness = torch.from_numpy(th).unsqueeze(0).to(self.device)
        return gt_boundary, gt_thickness

    # ------------------------------------------------------------------
    def train(self, train_subjects: List[Any], num_epochs: int = 100,
              lr: float = 2e-4, weight_decay: float = 1e-4,
              grad_clip: float = 1.0) -> Dict[str, Any]:
        """Full training loop (single-subject batches; accumulate as needed)."""
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        img = self.cfg.img_size
        history: List[float] = []

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            losses: List[float] = []
            for subj in train_subjects:
                vol, msk = _subject_tensors(subj, self.device, self.transform_fn)
                if msk is None:
                    continue
                vol = _pad_to(vol, img)
                msk = _pad_to(msk.float(), img).long()
                sev = self._severity_for(subj, vol)
                gt_boundary, gt_thickness = self._targets(msk)
                ramp = min(1.0, epoch / max(1, num_epochs // 5))

                opt.zero_grad()
                with torch.autocast(device_type=self._dev_type,
                                    dtype=torch.bfloat16,
                                    enabled=(self._dev_type == "cuda")):
                    out = self.model(vol, sev)
                    loss, _ = sacn_loss(
                        out, msk, self.model_cfg,
                        gt_boundary=gt_boundary, gt_thickness=gt_thickness,
                        evidential_ramp=ramp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()
                losses.append(float(loss.detach()))
            sched.step()
            history.append(float(np.mean(losses)) if losses else float("nan"))

        ckpt = os.path.join(self.output_dir, "sacn.pth")
        torch.save({"model": self.model.state_dict(),
                    "model_cfg": self.model_cfg.__dict__}, ckpt)
        return {"epoch_losses": history, "checkpoint_path": ckpt}

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, subject: Any, overlap: float = 0.75) -> Dict[str, Any]:
        """Sliding-window inference; returns pred_mask, thickness, uncertainty."""
        self.model.eval()
        vol, msk = _subject_tensors(subject, self.device, self.transform_fn)
        sev = self._severity_for(subject, vol)
        img = self.cfg.img_size

        def patch_fn(patch: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=self._dev_type, dtype=torch.bfloat16,
                                enabled=(self._dev_type == "cuda")):
                return self.model(patch, sev)["seg_logits"].float()

        if SlidingWindowInferer is not None:
            inferer = SlidingWindowInferer(roi_size=img, sw_batch_size=1,
                                           overlap=overlap, mode="gaussian")
            logits = inferer(vol, patch_fn)
        else:
            logits = patch_fn(_pad_to(vol, img))

        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int8)
        # Full multi-head pass at ROI for thickness / uncertainty maps.
        roi = self.model(_pad_to(vol, img), sev)
        thickness = roi["thickness"].squeeze(0).cpu().numpy().astype(np.float32)
        unc = (roi["uncertainty"].squeeze(0).cpu().numpy().astype(np.float32)
               if "uncertainty" in roi else None)
        gt = msk.squeeze().cpu().numpy().astype(np.int8) if msk is not None else None
        return {"pred_mask": pred, "thickness": thickness,
                "uncertainty": unc, "gt_mask": gt,
                "severity": sev.squeeze(0).cpu().numpy()}


# ---------------------------------------------------------------------------
# Smoke test on synthetic subjects (no torchio / MONAI / GPU needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    S = 16   # small volume so the pure-PyTorch Mamba fallback stays fast

    class _Cfg:
        img_size = (S, S, S)
        num_classes = 6
        output_dir = "./outputs_sacn_smoke/"

    class _Img:
        def __init__(self, d): self.data = d

    class _Subject:
        def __init__(self, sid, vol, lab):
            self.subject_id = sid
            self.image = _Img(vol)
            self.label = _Img(lab)

    def _make(sid):
        vol = torch.randn(1, S, S, S)
        lab = torch.zeros(1, S, S, S, dtype=torch.long)
        lab[0, 4:12, 4:12, 4:12] = 2
        lab[0, 2:4, 2:4, 2:4] = 4
        lab[0, 12:15, 12:15, 12:15] = 5
        lab[0, 0:2, 0:2, 0:2] = 1
        return _Subject(sid, vol, lab)

    cfg = _Cfg()
    subjects = [_make(f"s{i}") for i in range(4)]
    kl_lookup = {"s0": 0, "s1": 2, "s2": 3, "s3": 4}

    # Mamba in deeper stages only for a fast CPU smoke test (GPU: all stages).
    mcfg = SACNConfig(num_classes=6, base_channels=8, stage_depths=(1, 1, 1, 1),
                      use_evidential=True, deep_supervision=True,
                      encoder_block="mamba", mamba_stages=(1, 2, 3))
    trainer = SACNTrainer(cfg, device, model_cfg=mcfg,
                          severity_source="oracle", kl_lookup=kl_lookup)

    # ── Check 1: severity sourcing ──
    s = trainer._severity_for(subjects[2], None)
    assert tuple(s.squeeze(0).tolist()) == (0.0, 0.0, 1.0), s
    print(f"✅ Check 1 — oracle severity (KL3→severe) = {s.squeeze(0).tolist()}")

    # ── Check 2: 2-epoch train loop runs and loss is finite ──
    res = trainer.train(subjects, num_epochs=2, lr=1e-3)
    assert np.isfinite(res["epoch_losses"]).all(), res["epoch_losses"]
    assert os.path.exists(res["checkpoint_path"])
    print(f"✅ Check 2 — train 2 epochs, losses={np.round(res['epoch_losses'],3)}")

    # ── Check 3: inference returns pred + thickness + uncertainty ──
    out = trainer.predict(subjects[0])
    assert out["pred_mask"].shape == (S, S, S)
    assert out["thickness"].shape == (3, S, S, S)
    assert out["uncertainty"] is not None
    print(f"✅ Check 3 — predict: pred{out['pred_mask'].shape}, "
          f"labels={np.unique(out['pred_mask']).tolist()}, "
          f"sev={out['severity'].round(2).tolist()}")

    print(f"\n✅ All SACN-integration smoke-test checks passed on device={device}")

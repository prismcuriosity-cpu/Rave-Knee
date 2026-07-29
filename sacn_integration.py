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
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _pad_at_least(vol: torch.Tensor, target: Tuple[int, int, int],
                  value: float = 0.0) -> torch.Tensor:
    """Pad (never crop) so every spatial dim is >= target."""
    shape = vol.shape[2:]
    pad: List[int] = []
    for i in range(2, -1, -1):
        pad.extend([0, max(0, target[i] - shape[i])])
    if any(p > 0 for p in pad):
        vol = F.pad(vol, pad, value=value)
    return vol


def sample_patch(
    vol: torch.Tensor, msk: torch.Tensor, patch: Tuple[int, int, int],
    cartilage_classes: Sequence[int], fg_prob: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random foreground-biased crop of (vol, msk) to `patch` size.

    3D segmentation is trained on patches, not whole volumes — this bounds the
    Mamba scan length (and every activation) to the patch, which is the fix for
    the full-volume CUDA OOM. With probability `fg_prob` the patch is centred on
    a random cartilage voxel so thin structures are seen often enough.

    vol, msk: (1, 1, D, H, W). Returns crops of the same rank.
    """
    vol = _pad_at_least(vol, patch)
    msk = _pad_at_least(msk, patch)
    D, H, W = vol.shape[2:]
    pd, ph, pw = patch

    z0 = y0 = x0 = None
    if random.random() < fg_prob:
        fg = torch.zeros(D, H, W, dtype=torch.bool, device=msk.device)
        for c in cartilage_classes:
            fg |= (msk[0, 0] == c)
        nz = torch.nonzero(fg, as_tuple=False)
        if nz.numel() > 0:
            cz, cy, cx = nz[random.randrange(nz.shape[0])].tolist()
            z0 = min(max(cz - pd // 2, 0), D - pd)
            y0 = min(max(cy - ph // 2, 0), H - ph)
            x0 = min(max(cx - pw // 2, 0), W - pw)
    if z0 is None:
        z0 = random.randint(0, D - pd)
        y0 = random.randint(0, H - ph)
        x0 = random.randint(0, W - pw)

    vp = vol[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    mp = msk[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    return vp, mp


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
              grad_clip: float = 1.0,
              patch_size: Optional[Tuple[int, int, int]] = None,
              patches_per_subject: int = 1,
              empty_cache_every: int = 0) -> Dict[str, Any]:
        """Patch-based training loop (memory-bounded).

        Trains on random foreground-biased crops of ``patch_size`` rather than
        whole volumes, so activation memory (and the Mamba scan length) is
        bounded by the patch — the fix for full-volume CUDA OOM. Inference still
        stitches the whole volume with a sliding window (see ``predict``).

        Args:
            patch_size: crop size; defaults to ``cfg.patch_size`` or (96,96,96).
                Shrink this first if you still hit OOM.
            patches_per_subject: random crops per subject per epoch.
            empty_cache_every: call ``torch.cuda.empty_cache`` every N steps
                (0 disables) to fight fragmentation on tight cards.
        """
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        patch = tuple(patch_size or getattr(self.cfg, "patch_size", (96, 96, 96)))
        cart = self.model_cfg.cartilage_classes
        history: List[float] = []
        step = 0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            losses: List[float] = []
            for subj in train_subjects:
                vol, msk = _subject_tensors(subj, self.device, self.transform_fn)
                if msk is None:
                    continue
                sev = self._severity_for(subj, vol)   # global gate (pre-crop)
                msk = msk.float()
                ramp = min(1.0, epoch / max(1, num_epochs // 5))

                for _ in range(patches_per_subject):
                    vp, mp = sample_patch(vol, msk, patch, cart)
                    mp = mp.long()
                    gt_boundary, gt_thickness = self._targets(mp)

                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self._dev_type,
                                        dtype=torch.bfloat16,
                                        enabled=(self._dev_type == "cuda")):
                        out = self.model(vp, sev)
                        loss, _ = sacn_loss(
                            out, mp, self.model_cfg,
                            gt_boundary=gt_boundary, gt_thickness=gt_thickness,
                            evidential_ramp=ramp)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   grad_clip)
                    opt.step()
                    losses.append(float(loss.detach()))
                    step += 1
                    if (empty_cache_every and self._dev_type == "cuda"
                            and step % empty_cache_every == 0):
                        torch.cuda.empty_cache()

                del vol, msk
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
    # use_checkpoint exercises the gradient-checkpointing path.
    mcfg = SACNConfig(num_classes=6, base_channels=8, stage_depths=(1, 1, 1, 1),
                      use_evidential=True, deep_supervision=True,
                      encoder_block="mamba", mamba_stages=(1, 2, 3),
                      use_checkpoint=True)
    trainer = SACNTrainer(cfg, device, model_cfg=mcfg,
                          severity_source="oracle", kl_lookup=kl_lookup)

    # ── Check 1: severity sourcing ──
    s = trainer._severity_for(subjects[2], None)
    assert tuple(s.squeeze(0).tolist()) == (0.0, 0.0, 1.0), s
    print(f"✅ Check 1 — oracle severity (KL3→severe) = {s.squeeze(0).tolist()}")

    # ── Check 2: patch-based train loop runs and loss is finite ──
    res = trainer.train(subjects[:2], num_epochs=1, lr=1e-3,
                        patch_size=(S, S, S), patches_per_subject=1)
    assert np.isfinite(res["epoch_losses"]).all(), res["epoch_losses"]
    assert os.path.exists(res["checkpoint_path"])
    print(f"✅ Check 2 — patch train, losses={np.round(res['epoch_losses'],3)}")

    # ── Check 2b: sample_patch crops a larger volume down to patch size ──
    big = torch.randn(1, 1, 24, 20, 18)
    bigm = torch.zeros(1, 1, 24, 20, 18); bigm[0, 0, 5:12, 5:12, 5:12] = 2
    vp, mp = sample_patch(big, bigm, (16, 16, 16), (2, 4, 5))
    assert vp.shape == (1, 1, 16, 16, 16) and mp.shape == (1, 1, 16, 16, 16)
    print(f"✅ Check 2b — sample_patch 24×20×18 → {tuple(vp.shape[2:])}")

    # ── Check 3: inference returns pred + thickness + uncertainty ──
    out = trainer.predict(subjects[0])
    assert out["pred_mask"].shape == (S, S, S)
    assert out["thickness"].shape == (3, S, S, S)
    assert out["uncertainty"] is not None
    print(f"✅ Check 3 — predict: pred{out['pred_mask'].shape}, "
          f"labels={np.unique(out['pred_mask']).tolist()}, "
          f"sev={out['severity'].round(2).tolist()}")

    print(f"\n✅ All SACN-integration smoke-test checks passed on device={device}")

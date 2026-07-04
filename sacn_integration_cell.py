# === NOTEBOOK CELL: SACN Integration ===
# Paste this cell into rave_knee_nifti_csv_final_corrected_2.0.ipynb
# after Cell 21D (RGSSPD). Trains the Severity-Aware Cartilage Network (SACN,
# 3D Mamba encoder + FiLM severity conditioning + boundary/thickness heads)
# and evaluates it, KL-stratified, alongside the RGSSPD/RAVE baselines.
#
# Prerequisite notebook state:
#   model          — SwinUNETRModule (trained, on device)     [for RGSSPD votes]
#   faiss_index    — FAISS HNSW index over train embeddings
#   index_embeddings — np.ndarray (N, emb_dim)
#   train_subjects — list[tio.Subject]
#   val_subjects   — list[tio.Subject]
#   kl_lookup      — dict {subject_id: int|None}
#   cfg            — Config
#   device         — torch.device
#   get_transforms — callable (from Cell 15)
#   rgsspd_infer   — RGSSPDInference (optional; enables soft-vote severity)

import gc
import os

import numpy as np
import pandas as pd
import torch

from mamba3d import _HAS_MAMBA_KERNEL
from sacn_model import SACNConfig
from sacn_integration import SACNTrainer

cfg.num_workers = 0  # Windows: no multiprocessing DataLoader

# ─── 1. Model config — Mamba encoder, kernel-aware stage selection ───────────
# With the mamba-ssm CUDA kernel the selective scan is fast at every stage;
# without it, restrict Mamba to the deeper (low-token) stages so the
# pure-PyTorch fallback stays tractable.
mamba_stages = (0, 1, 2, 3) if _HAS_MAMBA_KERNEL else (2, 3)
print(f"mamba-ssm kernel available: {_HAS_MAMBA_KERNEL}  ->  mamba_stages={mamba_stages}")
if not _HAS_MAMBA_KERNEL:
    print("  Tip: `pip install mamba-ssm causal-conv1d` for the fast full-res scan.")

sacn_cfg = SACNConfig(
    in_channels=1,
    num_classes=cfg.num_classes,
    base_channels=32,
    stage_depths=(2, 2, 4, 2),
    use_evidential=True,
    deep_supervision=True,
    encoder_block="mamba",
    mamba_stages=mamba_stages,
)

# ─── 2. Trainer — severity from RGSSPD soft vote if available, else oracle ────
severity_source = "rgsspd" if "rgsspd_infer" in dir() and rgsspd_infer is not None \
    else "oracle"
sacn_trainer = SACNTrainer(
    cfg=cfg,
    device=device,
    model_cfg=sacn_cfg,
    severity_source=severity_source,
    kl_lookup=kl_lookup,
    rgsspd_infer=(rgsspd_infer if severity_source == "rgsspd" else None),
    transform_fn=get_transforms(train=False),
)
n_params = sum(p.numel() for p in sacn_trainer.model.parameters())
print(f"✅ SACN ready: {n_params/1e6:.2f}M params | severity_source={severity_source}")

# ─── 3. Train or load checkpoint ─────────────────────────────────────────────
ckpt_path = os.path.join(cfg.output_dir, "sacn.pth")
if os.path.exists(ckpt_path):
    print(f"\nLoading SACN checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    sacn_trainer.model.load_state_dict(state["model"])
    print("✅ SACN loaded from checkpoint")
else:
    print("\nNo checkpoint — training SACN …")
    res = sacn_trainer.train(train_subjects, num_epochs=100)
    print(f"✅ Training done. final loss = {res['epoch_losses'][-1]:.4f}")

torch.cuda.empty_cache()
gc.collect()

# ─── 4. Evaluate on val subjects (KL-stratified cartilage Dice + HD95) ───────
from rgsspd_module import _dice_binary, _hd95  # reuse metric helpers

CART = {2: "FC", 4: "MTC", 5: "LTC"}  # OAI-ZIB cartilage compartments
rows = []
print("\nEvaluating SACN on val_subjects …")
for subj in val_subjects:
    out = sacn_trainer.predict(subj)
    pred, gt = out["pred_mask"], out["gt_mask"]
    if gt is None:
        continue
    sid = getattr(subj, "subject_id", "")
    row = {"subject_id": sid, "kl_grade": kl_lookup.get(sid)}
    dices = []
    for c, name in CART.items():
        d = _dice_binary(pred == c, gt == c)
        h = _hd95(pred == c, gt == c)
        row[f"dice_{name}"] = d
        row[f"hd95_{name}"] = h
        dices.append(d)
    row["mean_cart_dice"] = float(np.mean(dices))
    rows.append(row)
    del out
    torch.cuda.empty_cache()

sacn_df = pd.DataFrame(rows)

# ─── 5. KL-stratified summary ────────────────────────────────────────────────
print("\n" + "=" * 64)
print("SACN — KL-stratified mean cartilage Dice")
print("=" * 64)
strata = {"mild": [0, 1], "moderate": [2], "severe": [3, 4]}
for st, grades in strata.items():
    sub = sacn_df[sacn_df["kl_grade"].isin(grades)]
    if sub.empty:
        continue
    print(f"  {st:>9s} (n={len(sub):3d}): "
          f"FC={sub['dice_FC'].mean():.3f}  MTC={sub['dice_MTC'].mean():.3f}  "
          f"LTC={sub['dice_LTC'].mean():.3f}  mean={sub['mean_cart_dice'].mean():.3f}")
print(f"\n  Overall mean cartilage Dice: {sacn_df['mean_cart_dice'].mean():.4f}")

csv_path = os.path.join(cfg.output_dir, "sacn_results.csv")
sacn_df.to_csv(csv_path, index=False)
print(f"✅ SACN results saved → {csv_path}")

torch.cuda.empty_cache()
gc.collect()
print("\n✅ SACN Integration Cell complete.")

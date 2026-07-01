# === NOTEBOOK CELL: RGSSPD Integration ===
# Paste this cell into rave_knee_nifti_csv_final_corrected_2.0.ipynb
# after Cell 50 (train_rave_v2_adapter / load_rave_v2_adapter).
#
# Prerequisite notebook state:
#   model         — SwinUNETRModule (trained, on device)
#   faiss_index   — FAISS HNSW index over train embeddings
#   index_embeddings — np.ndarray (N, emb_dim)
#   train_subjects — list[tio.Subject]
#   val_subjects   — list[tio.Subject]
#   kl_lookup      — dict {subject_id: int|None}
#   cfg            — Config
#   device         — torch.device
#   rave_v2        — RAVEKneeV2 (for A2 baseline)
#   get_transforms — callable (from Cell 15)

import gc
import os

import numpy as np
import pandas as pd
import torch

from rgsspd_module import (
    RGSSPDAblationRunner,
    RGSSPDInference,
    RGSSPDTrainer,
    SeveritySpecialistHead,
    _stage3_channels,
    plot_blend_weight_heatmap,
    plot_blend_weight_vs_true_kl,
    plot_kl_stratified_hd95,
)

cfg.num_workers = 0   # Windows: no multiprocessing DataLoader

# ─── 1. Instantiate RGSSPDTrainer ────────────────────────────────────────────
print("Instantiating RGSSPDTrainer …")
try:
    trainer = RGSSPDTrainer(
        backbone=model,
        faiss_index=faiss_index,
        index_embeddings=index_embeddings,
        train_subjects=train_subjects,
        kl_lookup=kl_lookup,
        cfg=cfg,
        device=device,
        transform_fn=get_transforms(train=False),
        output_dir=cfg.output_dir,
    )
    print(f"  embed_dim   = {_stage3_channels(cfg)}")
    print(f"  num_classes = {cfg.num_classes}")
    for st in ["mild", "moderate", "severe"]:
        _, subs, _ = trainer.build_stratum_subindex(st)
        print(f"  {st:>10s}: {len(subs)} subjects")
    print("✅ RGSSPDTrainer ready")

except Exception as e:
    print(f"❌ RGSSPDTrainer init failed: {e}")
    raise

# ─── 2. Train all specialists (or load cached checkpoint) ────────────────────
ckpt_path = os.path.join(cfg.output_dir, "specialist_heads.pth")
if os.path.exists(ckpt_path):
    print(f"\nLoading existing specialist checkpoint: {ckpt_path}")
    try:
        loaded = trainer.load_checkpoints(ckpt_path)
        if not loaded:
            raise RuntimeError("load_checkpoints returned False")
        print("✅ Specialist heads loaded from checkpoint")
    except Exception as e:
        print(f"❌ Checkpoint load failed: {e} — retraining …")
        trainer.train_all(num_epochs=50)
else:
    print("\nNo checkpoint found — training all specialists …")
    try:
        training_results = trainer.train_all(num_epochs=50)
        for st, res in training_results.items():
            print(f"  {st}: best_val_dice={res['best_dice']:.4f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

torch.cuda.empty_cache()
gc.collect()

# ─── 3. Build RGSSPDInference from trained heads ─────────────────────────────
print("\nBuilding RGSSPDInference …")
try:
    specialist_heads = {
        "mild":     trainer.heads["mild"].eval(),
        "moderate": trainer.heads["moderate"].eval(),
        "severe":   trainer.heads["severe"].eval(),
    }

    rgsspd_infer = RGSSPDInference(
        backbone=model,
        faiss_index=faiss_index,
        index_embeddings=index_embeddings,
        train_subjects=train_subjects,
        kl_lookup=kl_lookup,
        specialist_heads=specialist_heads,
        fusion_logit=trainer.fusion_logit,
        cfg=cfg,
        device=device,
        temperature=1.0,
        transform_fn=get_transforms(train=False),
    )
    print("✅ RGSSPDInference ready")

    # Quick smoke test on first val subject
    if val_subjects:
        print("\nSmoke test on val_subjects[0] …")
        _demo = rgsspd_infer.predict(val_subjects[0], {})
        print(f"  pred_mask shape : {_demo['pred_mask'].shape}")
        print(f"  unique labels   : {np.unique(_demo['pred_mask']).tolist()}")
        print(f"  blend_weights   : {_demo['blend_weights'].round(3)}")
        print(f"  neighbor_kl     : {_demo['neighbor_kl_grades']}")
        del _demo
        torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ RGSSPDInference build failed: {e}")
    raise

# ─── 4. Run ablation on first 30 eval subjects (smoke test) ──────────────────
print("\nRunning ablation smoke test (n=30) …")
eval_subjects_smoke = val_subjects[:30]

try:
    ablation_runner = RGSSPDAblationRunner(
        backbone=model,
        faiss_index=faiss_index,
        index_embeddings=index_embeddings,
        train_subjects=train_subjects,
        eval_subjects=eval_subjects_smoke,
        kl_lookup=kl_lookup,
        specialist_heads=specialist_heads,
        fusion_logit=trainer.fusion_logit,
        cfg=cfg,
        device=device,
        rave_v2=rave_v2,
        transform_fn=get_transforms(train=False),
    )

    ablation_df = ablation_runner.run_all_ablations(n_eval=30)
    print("\n✅ Ablation complete")

except Exception as e:
    print(f"❌ Ablation runner failed: {e}")
    raise

# ─── 5. Print KL-stratified summary per condition ─────────────────────────────
print("\n" + "=" * 70)
print("KL-Stratified Mean Dice and HD95 per Condition")
print("=" * 70)

strata_map = {"mild": [0, 1], "moderate": [2], "severe": [3, 4]}
metric_cols = ["mean_dice", "hd95_FC"]

for cond in ablation_runner.ALL_CONDITIONS:
    sub = ablation_df[ablation_df["condition"] == cond]
    if sub.empty:
        continue
    parts = [f"[{cond}]"]
    for st_name, kl_grades in strata_map.items():
        st_sub = sub[sub["kl_grade"].isin(kl_grades)]
        if st_sub.empty:
            continue
        md = st_sub["mean_dice"].mean()
        hd = st_sub["hd95_FC"].mean()
        parts.append(f"{st_name}: Dice={md:.3f} HD95={hd:.2f}")
    print("  " + " | ".join(parts))

# ─── 6. Save ablation DataFrame ───────────────────────────────────────────────
csv_path = os.path.join(cfg.output_dir, "ablation_results.csv")
ablation_df.to_csv(csv_path, index=False)
print(f"\n✅ Ablation results saved → {csv_path}")

# ─── 7. Visualizations ────────────────────────────────────────────────────────
print("\nGenerating figures …")
fig_dir = os.path.join(cfg.output_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

try:
    plot_blend_weight_vs_true_kl(
        ablation_df, os.path.join(fig_dir, "blend_weight_vs_kl.pdf")
    )
    plot_kl_stratified_hd95(
        ablation_df, os.path.join(fig_dir, "kl_stratified_hd95.pdf")
    )
    plot_blend_weight_heatmap(
        ablation_df, os.path.join(fig_dir, "blend_weight_heatmap.pdf")
    )
except Exception as e:
    print(f"⚠ Figure generation error (non-fatal): {e}")

# LaTeX table
try:
    latex = ablation_runner.generate_latex_table(ablation_df)
    latex_path = os.path.join(cfg.output_dir, "ablation_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"✅ LaTeX table saved → {latex_path}")
except Exception as e:
    print(f"⚠ LaTeX table generation error (non-fatal): {e}")

torch.cuda.empty_cache()
gc.collect()
print("\n✅ RGSSPD Integration Cell complete.")

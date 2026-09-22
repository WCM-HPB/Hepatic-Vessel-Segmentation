# Hepatic vessel model (Dataset092) — Colab release bundle

| File | What it is |
|---|---|
| `Dataset092_Hepatic_Vessels_25_nnUNetv2.zip` | Standard nnU-Net v2 model (381 MB). SHA-256 `6a9feb21305b9d4a2eca8c9dbe5923f84f754e71306d6558d826179b3ffbdb6f` |
| `Hepatic_Vessel_calculator.ipynb` | Colab notebook. Downloads the zip from the release URL below and checks the hash |
| `Dataset092_Hepatic_Vessels_25_export_info.json` | Export provenance (cluster paths only). Keep internal |
| `Dataset092_Hepatic_Vessels_25/` | Unzipped copy of the model, for local use |

## Publish

1. On `WCM-HPB/Hepatic-Vessel-Segmentation`, create release tag **`hepatic-vessels-20260921`** and upload the zip as an
   asset **without renaming it**. The notebook downloads
   `https://github.com/WCM-HPB/Hepatic-Vessel-Segmentation/releases/download/hepatic-vessels-20260921/Dataset092_Hepatic_Vessels_25_nnUNetv2.zip`.
   With the GitHub CLI:

   ```bash
   gh release create hepatic-vessels-20260921 Dataset092_Hepatic_Vessels_25_nnUNetv2.zip \
     --repo WCM-HPB/Hepatic-Vessel-Segmentation --title "Hepatic vessel segmentation (nnU-Net v2)" \
     --notes "Standard nnU-Net v2 model for Hepatic_Vessel_calculator.ipynb."
   ```

2. Commit `Hepatic_Vessel_calculator.ipynb` to the repo root on `main`. Colab link:
   `https://colab.research.google.com/github/WCM-HPB/Hepatic-Vessel-Segmentation/blob/main/Hepatic_Vessel_calculator.ipynb`
3. Add a line to the README's notebook list, for example:

   > - **`Hepatic_Vessel_calculator.ipynb`** — Segments the **portal vein, IVC, portal-splenic confluence, and hepatic
   >   veins** on contrast-enhanced CT and reports their volumes. A standard nnU-Net v2 model (`pip install nnunetv2`),
   >   so it also runs on your own server. Colab: [Hepatic Vessels](https://colab.research.google.com/github/WCM-HPB/Hepatic-Vessel-Segmentation/blob/main/Hepatic_Vessel_calculator.ipynb)

If the model is ever re-exported, the zip hash changes: update `MODEL_SHA256` in the notebook's download cell.

## Model card

- **Source:** `nnUNetTrainerNuSA__nnUNetPlannerResEncL__3d_fullres/fold_all/checkpoint_final.pth` (epoch 500), trained
  in CLAP on all 25 cases of `Dataset092_Hepatic_Vessels_25` (living liver donor CT).
- **Network:** nnU-Net ResEnc-L `3d_fullres`, patch 64×256×256, spacing 3.0×1.406×1.406 mm, per-image z-score
  normalization, 102.3 M parameters.
- **Labels:** 1 portal vein, 2 IVC, 3 portal-splenic confluence, 4 hepatic veins.
- **Export** (`tools/export_stock_nnunet.py` in `MFM/cl_web`): NuSA adapters folded into plain conv weights, trainer
  renamed to `nnUNetTrainer`, reader switched from the fork-only `SimpleITKIOWithReorient` to stock `SimpleITKIO`.
  Callers reorient inputs to RAS (the notebook does this). Only `plans.json`, `dataset.json` and the checkpoint are
  shipped. Training logs and validation outputs, whose file names carry donor initials and scan dates, are not.
- **Accuracy:** there is no held-out estimate. The model was trained on `fold_all`, so nnU-Net's validation ran on the
  training cases: Dice 0.876 portal vein, 0.978 IVC, 0.954 confluence, 0.877 hepatic veins (mean 0.921). That measures
  fit, not generalization. Do not quote it as accuracy.

## Verification (2026-09-21)

- Exported network vs original NuSA network, float64 probe: max |logit diff| 4.6e-6 on logits up to 22.8, 100% argmax
  agreement. Forcing the stale `conv` alias instead gives a logit diff of 22.2, so the check catches that failure.
- Notebook run headless with `nnunetv2==2.8.1` from PyPI, in an env matching Colab's current preinstalled packages
  (Python 3.13, torch 2.11.0, numpy 2.1.3), on an A40 limited to 2 CPUs and 12 GB RAM:
  - reproduces the fork's own predictions on 3 training scans (Dice ≥ 0.9995 per label; 28-104 voxels differ out of
    31-177 M, from fp16 rounding);
  - LPS and axis-permuted (PIL) copies of a scan give exactly the same labels;
  - a DICOM series gives exactly the same labels as its NIfTI, and a 3-slice localizer is skipped;
  - peak RAM 8.2 GiB for a 677-slice scan.
- On Colab's package set, the notebook's install step adds 14 small packages and downgrades `timm` 1.0.29 → 1.0.22.
  torch is not reinstalled.

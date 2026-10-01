# Hepatic Vessel Segmentation

nnU-Net v2 notebook and model for segmenting the portal vein, inferior vena cava (IVC), portal-splenic confluence, and hepatic veins in contrast-enhanced CT.

- [Notebook](Hepatic_Vessel_calculator.ipynb)
- [Open in Colab](https://colab.research.google.com/github/WCM-HPB/Hepatic-Vessel-Segmentation/blob/main/Hepatic_Vessel_calculator.ipynb)
- [Model download](https://github.com/WCM-HPB/Hepatic-Vessel-Segmentation/releases/tag/hepatic-vessels-20260921)
- [Model card and export verification record](RELEASE.md)

## Usage

No preprocessing is needed anywhere: inputs can be NIfTI (`.nii.gz`, `.nii`), `.nrrd`, `.mha`, DICOM folders, or `.zip`
archives of these, in any orientation. DICOM conversion and reorientation are automatic, and the label maps are written
on each scan's own voxel grid.

**Google Colab:** open the notebook, upload your scans from your computer (or pick a Google Drive folder instead), and
run the cells. Results download as one zip.

**Your own GPU server:**

```bash
pip install nnunetv2==2.8.1
python hepatic_vessel_seg.py -i /path/to/scans -o /path/to/results
```

The model downloads on first use. On machines without internet access, pass `--model` with the release zip.

For each scan the output folder gets `<scan>_vessels.nii.gz` (1 portal vein, 2 IVC, 3 portal-splenic confluence,
4 hepatic veins), a quality-control image `<scan>_qc.png`, and for DICOM input `<scan>_ct.nii.gz`.
`hepatic_vessel_volumes.csv` lists the volumes in mL.

## Model files

The model archive is distributed as a GitHub Release asset; the uncompressed checkpoint is excluded from Git. The archive SHA-256 is `6a9feb21305b9d4a2eca8c9dbe5923f84f754e71306d6558d826179b3ffbdb6f`.

## Intended use

Research only. The model was trained on all 25 available cases; performance on held-out cases has not been established. Review segmentations before interpreting the reported volumes.

The validation results in `RELEASE.md` are the supplied export record; they were not rerun during repository publication.

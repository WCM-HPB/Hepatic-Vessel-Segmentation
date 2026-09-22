# Hepatic Vessel Segmentation

nnU-Net v2 notebook and model for segmenting the portal vein, inferior vena cava (IVC), portal-splenic confluence, and hepatic veins in contrast-enhanced CT.

- [Notebook](Hepatic_Vessel_calculator.ipynb)
- [Open in Colab](https://colab.research.google.com/github/WCM-HPB/Hepatic-Vessel-Segmentation/blob/main/Hepatic_Vessel_calculator.ipynb)
- [Model download](https://github.com/WCM-HPB/Hepatic-Vessel-Segmentation/releases/tag/hepatic-vessels-20260921)
- [Model card and export verification record](RELEASE.md)

The model archive is distributed as a GitHub Release asset; the uncompressed checkpoint is excluded from Git. The archive SHA-256 is `6a9feb21305b9d4a2eca8c9dbe5923f84f754e71306d6558d826179b3ffbdb6f`.

## Private repository access

This repository is private. Viewing the notebook and downloading the release asset require repository access. The notebook currently uses an unauthenticated download URL, which will not download a private release asset in Colab. Download the archive using an authorized GitHub session and use the local installation instructions in step 8; Colab automatic downloading requires an authenticated download workflow.

## Intended use

Research only. The model was trained on all 25 available cases; performance on held-out cases has not been established. Review segmentations before interpreting the reported volumes.

The validation results in `RELEASE.md` are the supplied export record; they were not rerun during repository publication.

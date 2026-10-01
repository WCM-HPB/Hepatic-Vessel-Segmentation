#!/usr/bin/env python3
"""Segment hepatic vessels on contrast-enhanced CT with a standard nnU-Net v2 model.

Labels: 1 portal vein, 2 inferior vena cava (IVC), 3 portal-splenic confluence, 4 hepatic veins.

    pip install nnunetv2==2.8.1
    python hepatic_vessel_seg.py -i scans/ -o results/

The input can be a NIfTI / NRRD / MHA file, a DICOM folder, a .zip of either, or a folder holding
any mix of these. No preprocessing is needed: DICOM series are converted, every scan is reoriented
to RAS for the model, and the labels are written back on the input's own voxel grid. The model
(about 380 MB) is downloaded on first use; pass --model to use a local copy instead.

For each scan the output folder gets <scan>_vessels.nii.gz (labels), <scan>_qc.png (overview
image) and, for DICOM input, <scan>_ct.nii.gz. hepatic_vessel_volumes.csv lists the volumes (mL)
of every label map in the folder.

Research use only. Not for clinical, diagnostic, or treatment use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MODEL_URL = (
    "https://github.com/WCM-HPB/Hepatic-Vessel-Segmentation/releases/download/"
    "hepatic-vessels-20260921/Dataset092_Hepatic_Vessels_25_nnUNetv2.zip"
)
MODEL_SHA256 = "6a9feb21305b9d4a2eca8c9dbe5923f84f754e71306d6558d826179b3ffbdb6f"
MODEL_SUBDIR = Path("Dataset092_Hepatic_Vessels_25", "nnUNetTrainer__nnUNetPlannerResEncL__3d_fullres")
CHECKPOINT = "checkpoint_final.pth"

LABELS = {1: "Portal vein", 2: "IVC", 3: "Portal-splenic confluence", 4: "Hepatic veins"}
COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
IMAGE_ENDINGS = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")
LABEL_SUFFIX = "_vessels.nii.gz"
CT_SUFFIX = "_ct.nii.gz"
VOLUME_CSV = "hepatic_vessel_volumes.csv"
NOT_DICOM_ENDINGS = (".png", ".jpg", ".jpeg", ".csv", ".txt", ".json", ".md", ".pdf", ".xlsx", ".docx", ".gz", ".npy", ".npz")
MIN_SLICES = 20  # skips scouts and localizers
LOW_TOTAL_ML = 10.0  # typical scans give 150-200 mL across the four labels


@dataclass
class Scan:
    name: str
    source: str | list[str]  # image file, or the files of one DICOM series
    is_dicom: bool


# ---------------------------------------------------------------- model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hepatic_vessel_seg"


def ensure_model(cache_dir: str | Path | None = None) -> Path:
    """Return the installed model folder, downloading and checking the release zip if needed."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache()
    model_dir = cache_dir / MODEL_SUBDIR
    if (model_dir / "fold_all" / CHECKPOINT).exists():
        return model_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "model.zip"
    print("Downloading the model (about 380 MB) ...", flush=True)
    shown = set()

    def progress(blocks: int, block_size: int, total: int) -> None:
        tenth = min(10, blocks * block_size * 10 // max(total, 1))
        if tenth not in shown:
            shown.add(tenth)
            print(f"  {tenth * 10}%", end="\n" if tenth == 10 else "", flush=True)

    urllib.request.urlretrieve(MODEL_URL, archive, progress)
    if _sha256(archive) != MODEL_SHA256:
        archive.unlink()
        raise RuntimeError("The model download is incomplete or corrupted. Please run again.")
    return install_model(archive, cache_dir)


def install_model(archive: Path, cache_dir: Path) -> Path:
    # Same layout nnUNetv2_install_pretrained_model_from_zip produces.
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(cache_dir)
    if archive.name == "model.zip" and archive.parent == cache_dir:
        archive.unlink()
    return cache_dir / MODEL_SUBDIR


def resolve_model(path: str | None) -> Path:
    """--model may be omitted (download), the release zip, or a folder containing the model."""
    if not path:
        return ensure_model()
    path = Path(path)
    if path.is_file() and path.suffix == ".zip":
        return install_model(path, default_cache())
    for candidate in (path, path / MODEL_SUBDIR):
        if (candidate / "fold_all" / CHECKPOINT).exists():
            return candidate
    raise FileNotFoundError(f"No model found at {path}")


def load_predictor(model_dir: str | Path, mirroring: bool = True):
    """Load the standard nnU-Net v2 predictor (GPU if available)."""
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(name, str(Path(model_dir).parents[1]))  # silences nnU-Net's path warnings
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found; segmentation will take a long time.", flush=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=mirroring,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=("all",), checkpoint_name=CHECKPOINT)
    return predictor


# ---------------------------------------------------------------- input


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", text).strip("_") or "scan"


def _skip(name: str) -> bool:
    # Hidden files and the debris macOS adds to zips are never inputs.
    return name.startswith(".") or name == "__MACOSX"


def _image_stem(filename: str) -> str | None:
    lower = filename.lower()
    ending = next((e for e in IMAGE_ENDINGS if lower.endswith(e)), None)
    return filename[: -len(ending)] if ending else None


def _dicom_scans(folder: Path, prefix: str) -> list[Scan]:
    sitk.ProcessObject.SetGlobalWarningDisplay(False)  # GDCM warns about every folder without DICOM
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
    finally:
        sitk.ProcessObject.SetGlobalWarningDisplay(True)
    scans = []
    for series in series_ids:
        files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series))
        header = sitk.ImageFileReader()
        header.SetFileName(files[0])
        header.ReadImageInformation()
        size = header.GetSize()
        slices = len(files) if len(files) > 1 else (size[2] if len(size) > 2 else 1)
        if slices < MIN_SLICES:
            continue

        def tag(key: str) -> str:
            return header.GetMetaData(key).strip() if header.HasMetaDataKey(key) else ""

        name = f"{prefix}S{tag('0020|0011')}_{tag('0008|103e')}"
        scans.append(Scan(name, files if len(files) > 1 else files[0], True))  # one file = multi-frame DICOM
    return scans


def find_scans(path: Path, unzip_dir: Path, skip_dir: Path | None = None) -> list[Scan]:
    """Every image file and DICOM series in path (a file, zip, or folder; zips are unpacked into unzip_dir)."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".zip":
            return find_scans(_unzip(path, unzip_dir), unzip_dir, skip_dir)
        stem = _image_stem(path.name)
        scans = [Scan(stem, str(path), False)] if stem else _dicom_scans(path.parent, "")
        return _unique(scans)

    skip_dir = Path(skip_dir).resolve() if skip_dir else None
    scans = []
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        dirs[:] = sorted(d for d in dirs if not _skip(d) and (root_path / d).resolve() != skip_dir)
        in_output = root_path.resolve() == skip_dir  # results written next to the inputs
        rel = root_path.relative_to(path)
        where = "" if rel == Path(".") else "_".join(rel.parts) + "_"
        other_files = False
        for name in sorted(files):
            if _skip(name) or (in_output and name.endswith((LABEL_SUFFIX, CT_SUFFIX))):
                continue
            stem = _image_stem(name)
            if stem:
                scans.append(Scan(where + stem, str(root_path / name), False))
            elif name.lower().endswith(".zip"):
                for scan in find_scans(_unzip(root_path / name, unzip_dir), unzip_dir, skip_dir):
                    scans.append(Scan(where + scan.name, scan.source, scan.is_dicom))
            elif not name.lower().endswith(NOT_DICOM_ENDINGS):
                other_files = True  # DICOM files often have no extension at all
        if other_files:
            scans.extend(_dicom_scans(root_path, where or path.name + "_"))
    return _unique(scans)


def _unzip(archive: Path, unzip_dir: Path) -> Path:
    target = Path(unzip_dir) / _safe_name(archive.stem)
    k = 2
    while target.exists():
        target, k = Path(unzip_dir) / f"{_safe_name(archive.stem)}_{k}", k + 1
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target)  # zipfile drops absolute paths and ".." components
    # A zip holding a single folder is named after that folder, not after the zip.
    entries = [p for p in target.iterdir() if not _skip(p.name)]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else target


def _unique(scans: list[Scan]) -> list[Scan]:
    seen, out = set(), []
    for scan in scans:
        base = candidate = _safe_name(scan.name)
        k = 2
        while candidate in seen:
            candidate, k = f"{base}_{k}", k + 1
        seen.add(candidate)
        out.append(Scan(candidate, scan.source, scan.is_dicom))
    return out


def read_scan(source: str | list[str]) -> sitk.Image:
    if isinstance(source, list):
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(source)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(source)
    while image.GetDimension() > 3 and image.GetSize()[-1] == 1:  # some converters add a time axis of length 1
        image = image[(slice(None),) * (image.GetDimension() - 1) + (0,)]
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(
            f"expected a 3D grayscale CT volume, got size {image.GetSize()} "
            f"with {image.GetNumberOfComponentsPerPixel()} channel(s)"
        )
    # nnU-Net computes in float32; float64 CTs would only double the memory footprint.
    return sitk.Cast(image, sitk.sitkFloat32) if image.GetPixelID() == sitk.sitkFloat64 else image


def describe(image: sitk.Image) -> str:
    sample = sitk.GetArrayViewFromImage(image)[::2, ::4, ::4]
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
    size = "x".join(map(str, image.GetSize()))
    spacing = "x".join(f"{s:.2f}" for s in image.GetSpacing())
    return f"{size} voxels, {spacing} mm, {orientation}, intensity {sample.min():.0f} to {sample.max():.0f}"


# ---------------------------------------------------------------- segmentation


def segment(predictor, image: sitk.Image) -> sitk.Image:
    """Run the model on one CT; return labels on the same voxel grid as the input.

    The model was trained on arrays reoriented with SimpleITK.DICOMOrient(image, "RAS"), so every
    scan is reoriented before prediction and the labels are reoriented back afterwards.
    """
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
    ras = sitk.DICOMOrient(image, "RAS")
    data = sitk.GetArrayFromImage(ras).astype(np.float32, copy=False)[None]
    origin, spacing, direction = ras.GetOrigin(), ras.GetSpacing(), ras.GetDirection()
    del ras  # free the copy before nnU-Net allocates its own buffers
    labels = predictor.predict_single_npy_array(data, {"spacing": list(spacing)[::-1]})
    seg = sitk.GetImageFromArray(labels.astype(np.uint8))
    seg.SetOrigin(origin)
    seg.SetSpacing(spacing)
    seg.SetDirection(direction)
    seg = sitk.DICOMOrient(seg, orientation)
    seg.CopyInformation(image)
    return seg


def volumes_ml(seg: sitk.Image) -> dict[str, float]:
    counts = np.bincount(sitk.GetArrayViewFromImage(seg).ravel(), minlength=len(LABELS) + 1)
    voxel_ml = float(np.prod(seg.GetSpacing())) / 1000.0
    return {f"{name} (mL)": round(float(counts[k]) * voxel_ml, 1) for k, name in LABELS.items()}


def warnings_for(image: sitk.Image, volumes: dict[str, float]) -> list[str]:
    notes = []
    if np.allclose(image.GetSpacing(), 1.0):
        notes.append("voxel spacing is exactly 1 x 1 x 1 mm. If that is not the scan's real spacing, the file "
                     "lost it during conversion; re-export it from DICOM (e.g. with ITK-SNAP or dcm2niix)")
    sample = sitk.GetArrayViewFromImage(image)[::2, ::4, ::4]
    if sample.max() < 500 or sample.min() > -500:
        notes.append(f"intensities run from {sample.min():.0f} to {sample.max():.0f}, which does not look like CT "
                     "in Hounsfield units. The model only works on CT")
    total = sum(volumes.values())
    if total < LOW_TOTAL_ML:
        notes.append(f"almost no vessels found ({total:.1f} mL in total; typical scans give over 100 mL). The model "
                     "expects a contrast-enhanced CT of the abdomen that covers the liver")
    return notes


def save_qc(image: sitk.Image, seg: sitk.Image, title: str, path: Path) -> None:
    """Axial and coronal slices through the portal-splenic confluence, plus a coronal projection of all labels."""
    import matplotlib

    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ct = sitk.DICOMOrient(image, "LPS")  # LPS arrays display anterior-up with the patient's right on the left
    arr = sitk.GetArrayViewFromImage(ct)
    lab = sitk.GetArrayFromImage(sitk.DICOMOrient(seg, "LPS"))
    sx, _, sz = ct.GetSpacing()
    focus = np.argwhere(lab == 3) if (lab == 3).any() else np.argwhere(lab > 0)
    z, y, _ = focus.mean(0).round().astype(int) if len(focus) else np.array(lab.shape) // 2
    projection = np.zeros((lab.shape[0], lab.shape[2]), np.uint8)
    for k in (2, 4, 1, 3):  # small structures last so they stay visible
        projection[(lab == k).any(axis=1)] = k

    cmap = ListedColormap([COLORS[k] for k in LABELS])
    norm = BoundaryNorm(np.arange(0.5, len(LABELS) + 1), cmap.N)
    panels = [  # title, CT, labels, pixel aspect, HU window max, label opacity
        ("Axial", arr[z], lab[z], 1.0, 300, 0.6),
        ("Coronal", arr[:, y][::-1], lab[:, y][::-1], sz / sx, 300, 0.6),
        ("Coronal projection", arr.max(axis=1)[::-1], projection[::-1], sz / sx, 1000, 0.8),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, (name, img, lbl, aspect, vmax, alpha) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=-100, vmax=vmax, aspect=aspect)
        ax.imshow(np.ma.masked_equal(lbl, 0), cmap=cmap, norm=norm, alpha=alpha, aspect=aspect, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in LABELS]
    fig.legend(handles, LABELS.values(), loc="outside lower center", ncol=len(LABELS), frameon=False)
    fig.suptitle(title)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def process(predictor, scan: Scan, output_dir: Path) -> tuple[dict[str, float], list[str], str]:
    # One scan per call, so its arrays are released before the next scan is read.
    image = read_scan(scan.source)
    info = describe(image)
    seg = segment(predictor, image)
    sitk.WriteImage(seg, str(output_dir / f"{scan.name}{LABEL_SUFFIX}"), True)
    if scan.is_dicom:
        sitk.WriteImage(image, str(output_dir / f"{scan.name}{CT_SUFFIX}"), True)
    save_qc(image, seg, scan.name, output_dir / f"{scan.name}_qc.png")
    volumes = volumes_ml(seg)
    return volumes, warnings_for(image, volumes), info


def volume_table(output_dir: Path) -> list[dict]:
    """Volumes of every label map in output_dir, also written to hepatic_vessel_volumes.csv."""
    rows = []
    for path in sorted(Path(output_dir).glob(f"*{LABEL_SUFFIX}")):
        rows.append({"Scan": path.name[: -len(LABEL_SUFFIX)], **volumes_ml(sitk.ReadImage(str(path)))})
    if rows:
        with open(Path(output_dir) / VOLUME_CSV, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run(predictor, input_path: str | Path, output_dir: str | Path) -> int:
    """Segment every scan under input_path into output_dir; return how many succeeded."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as unzip_dir:
        scans = find_scans(input_path, Path(unzip_dir), skip_dir=output_dir)
        if not scans:
            print(f"No scans found in {input_path}.")
            print("Expected NIfTI (.nii.gz, .nii), .nrrd, .mha files, DICOM folders with at least "
                  f"{MIN_SLICES} slices, or .zip archives of these.")
            if input_path.is_dir():
                listing = sorted(p.name for p in input_path.iterdir())
                print("The folder contains:", listing[:20] if listing else "nothing")
            return 0
        print(f"Found {len(scans)} scan(s) in {input_path}\n", flush=True)
        done = 0
        for i, scan in enumerate(scans, 1):
            start = time.time()
            try:
                volumes, notes, info = process(predictor, scan, output_dir)
            except Exception as exc:  # one bad scan should not stop the batch
                print(f"[{i}/{len(scans)}] {scan.name}: FAILED - {exc}", flush=True)
                continue
            done += 1
            summary = ", ".join(f"{name} {volumes[f'{name} (mL)']:.1f}" for name in LABELS.values())
            print(f"[{i}/{len(scans)}] {scan.name} ({time.time() - start:.0f} s)\n"
                  f"    input:   {info}\n    volumes: {summary} mL", flush=True)
            for note in notes:
                print(f"    WARNING: {note}", flush=True)
    volume_table(output_dir)
    print(f"\n{done} of {len(scans)} scan(s) segmented. Results are in {output_dir}")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="image file, DICOM folder, .zip, or a folder of these")
    parser.add_argument("-o", "--output", required=True, help="folder for the results")
    parser.add_argument("--model", help="model folder or release zip (default: download on first use)")
    parser.add_argument("--no-mirroring", action="store_true", help="faster, without test-time mirroring")
    args = parser.parse_args(argv)

    predictor = load_predictor(resolve_model(args.model), mirroring=not args.no_mirroring)
    return 0 if run(predictor, args.input, args.output) else 1


if __name__ == "__main__":
    sys.exit(main())

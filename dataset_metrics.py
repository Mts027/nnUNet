#!/usr/bin/env python3
from __future__ import annotations

"""
nnUNet connected components & volume analysis (GPU with CuPy, CPU fallback with SciPy)

Defaults now:
- Log-scaled x-axis ON by default.
- No percentile zoom by default (plot full range).
- Volumes reported/visualized in mm³ (not mL).
- 25 bins by default.
- When saving, figures are SVG.
- Custom plot title via --title.

Change in this version:
- The 'Components: <total>' stat is replaced with four numbers describing
  components PER SCAN: mean | p50 | p25 | p75.

NEW:
- Falls back to SciPy/NumPy (CPU) if CuPy is not installed. If GPU is available,
  it uses CuPy + cupyx.scipy.ndimage automatically.

Examples
--------
# Show interactively (no files written), log-x default, full range, 25 bins:
python nnunet_cc_volume.py /path/to/Dataset

# Linear x-axis instead of log:
python nnunet_cc_volume.py /path/to/Dataset --no-logx

# Save CSV + SVG (overall + per-class), custom title:
python nnunet_cc_volume.py /path/to/Dataset --save --save-per-class --title "My Task Volumes"

# Manually zoom x-range and change bin rule:
python nnunet_cc_volume.py /path/to/Dataset --xlim 1,5000 --bins fd
"""

import os
import sys
import glob
import argparse
import csv
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = "deep"
sns.set_theme(context="notebook", style="whitegrid", palette=PALETTE)

BLUE = sns.color_palette(PALETTE)[0]
ORANGE = sns.color_palette(PALETTE)[1]
GREEN = sns.color_palette(PALETTE)[2]
RED   = sns.color_palette(PALETTE)[3]
GRAY  = sns.color_palette(PALETTE)[7]

# ---- Backend selection (CuPy if available, else NumPy/SciPy) ----
USE_GPU = False
xp: Any = np      # numpy or cupy
nd_label = None   # labeling function from cupyx.scipy.ndimage or scipy.ndimage
cp = None         # will be set if CuPy is available

try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _cnd
    cp = _cp
    xp = cp
    nd_label = _cnd.label
    USE_GPU = True
except Exception as e:
    # Fall back to SciPy
    try:
        import scipy.ndimage as _snd
        nd_label = _snd.label
        USE_GPU = False
        print(
            "WARNING: CuPy not available; falling back to SciPy/NumPy (CPU). "
            "For GPU, install a CUDA-matching wheel, e.g.: pip install 'cupy-cuda12x'\n"
            f"CuPy import error was: {e}",
            file=sys.stderr,
        )
    except Exception as ee:  # pragma: no cover
        sys.exit(
            "Neither CuPy nor SciPy could be imported. Please install at least SciPy:\n"
            "  pip install scipy\n"
            f"Import errors:\n  CuPy -> {e}\n  SciPy -> {ee}"
        )

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


# --------------------------- I/O & core -------------------------------------
def get_spacing_mm(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError("NIfTI image does not have 3 spatial dimensions")
    sx, sy, sz = zooms[:3]
    if sx == 0 or sy == 0 or sz == 0:
        raise ValueError(f"Invalid voxel spacing found: {zooms}")
    return float(sx), float(sy), float(sz)


def _structure_for_connectivity(connectivity: int):
    """Return a 3x3x3 structuring element for the chosen connectivity in the active backend."""
    if connectivity == 6:
        structure = xp.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=xp.int8,
        )
    elif connectivity == 18:
        structure = xp.ones((3, 3, 3), dtype=xp.int8)
        # zero out corners
        structure[0, 0, 0] = structure[0, 0, 2] = structure[0, 2, 0] = structure[0, 2, 2] = 0
        structure[2, 0, 0] = structure[2, 0, 2] = structure[2, 2, 0] = structure[2, 2, 2] = 0
    elif connectivity == 26:
        structure = xp.ones((3, 3, 3), dtype=xp.int8)
    else:
        raise ValueError("connectivity must be one of {6, 18, 26}")
    return structure


def connected_components(mask: Any, connectivity: int = 26) -> Tuple[Any, int]:
    """Label connected components on a boolean 3D mask using the selected backend (GPU if available)."""
    if mask.dtype != xp.bool_:
        mask = mask.astype(xp.bool_)
    structure = _structure_for_connectivity(connectivity)
    labels, num = nd_label(mask, structure=structure)
    return labels, int(num)


def analyze_label_file(
    nii_path: str,
    connectivity: int = 26,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[float]]]:
    img = nib.load(nii_path)
    sx, sy, sz = get_spacing_mm(img)
    voxel_mm3 = sx * sy * sz

    data = np.asanyarray(img.dataobj)
    if data.dtype.kind == "f":
        data = np.rint(data).astype(np.int32, copy=False)
    else:
        data = data.astype(np.int32, copy=False)

    if data.ndim != 3:
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D labels, got shape {data.shape} for {os.path.basename(nii_path)}")

    g = xp.asarray(data)
    rows: List[Dict[str, Any]] = []
    per_class_vols_mm3: Dict[int, List[float]] = {}

    classes = xp.unique(g)
    classes = classes[classes != 0]  # ignore background

    for cls_val in classes.tolist():
        cls_id = int(cls_val)
        mask = (g == cls_id)

        labels, num = connected_components(mask, connectivity=connectivity)
        if num == 0:
            continue

        counts = xp.bincount(labels.ravel())[1:]  # drop background 0

        # bring counts to host as int64
        if USE_GPU:
            counts_host = cp.asnumpy(counts).astype(np.int64, copy=False)
        else:
            counts_host = counts.astype(np.int64, copy=False)

        volumes_mm3 = counts_host * voxel_mm3  # mm³
        per_class_vols_mm3.setdefault(cls_id, []).extend(volumes_mm3.tolist())

        for comp_idx, (vox, vmm3) in enumerate(zip(counts_host, volumes_mm3), start=1):
            rows.append(
                {
                    "file": os.path.basename(nii_path),
                    "class_id": cls_id,
                    "component_id": comp_idx,
                    "voxel_count": int(vox),
                    "voxel_spacing_x_mm": sx,
                    "voxel_spacing_y_mm": sy,
                    "voxel_spacing_z_mm": sz,
                    "voxel_volume_mm3": voxel_mm3,
                    "component_volume_mm3": float(vmm3),
                }
            )

        # free intermediates (GPU only)
        if USE_GPU:
            del labels, counts, mask
            try:
                cp._default_memory_pool.free_all_blocks()
            except Exception:
                pass

    # free big array (GPU only)
    if USE_GPU:
        del g
        try:
            cp._default_memory_pool.free_all_blocks()
        except Exception:
            pass

    return rows, per_class_vols_mm3


# --------------------------- plotting ---------------------------------------
def _parse_xlim(xlim_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if not xlim_str:
        return None
    parts = [p.strip() for p in xlim_str.split(",")]
    if len(parts) != 2:
        raise ValueError("--xlim must be in the form MIN,MAX (e.g., 1,5000)")
    lo = float(parts[0]) if parts[0] != "" else 0.0
    hi = float(parts[1])
    if hi <= lo:
        raise ValueError("--xlim MAX must be greater than MIN")
    return (lo, hi)

import matplotlib as mpl

def plot_distribution(
    volumes_mm3: List[float],
    out_path: Optional[str] = None,
    show: bool = True,
    logx: bool = True,
    title_suffix: str = "",
    title_base: str = "Connected Component Volume Distribution",
    bins: Union[int, str] = 25,
    xlim: Optional[Tuple[float, float]] = None,
    zoom_pctl: float = 100.0,
    hist_color: Optional[Union[str, tuple]] = None,  # override if you want
) -> None:
    """
    Seaborn histogram that matches your script's look exactly:
    - Uses the same palette color (BLUE) without desaturation or edges.
    - Respects log-x and custom bin edges.
    """
    if not volumes_mm3:
        print("No volumes to plot.")
        return

    v = np.asarray([float(x) for x in volumes_mm3 if x > 0.0], dtype=float)
    if v.size == 0:
        print("No positive volumes to plot.")
        return

    # ----- decide plotting range -----
    if xlim is not None:
        lo, hi = max(np.min(v), float(xlim[0])), float(xlim[1])
    else:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if not logx and 0 < zoom_pctl < 100:
            hi = float(np.percentile(v, zoom_pctl))

    # avoid degenerate range
    if hi <= lo:
        eps = 1e-12
        lo, hi = max(lo - eps, np.nextafter(0, 1)), hi + eps

    # Count clipped & nonpositive for title note
    clipped = int(np.sum(v < lo) + np.sum(v > hi))
    nonpos = int(len(volumes_mm3) - v.size)

    # ----- choose bins -----
    bins_spec = bins
    if logx:
        lo = max(lo, np.nextafter(0, 1))  # strictly positive
        if isinstance(bins_spec, str):
            log_edges = np.histogram_bin_edges(
                np.log10(v[(v >= lo) & (v <= hi)]),
                bins=bins_spec.lower(),
                range=(np.log10(lo), np.log10(hi)),
            )
            use_bins = 10 ** log_edges
        else:
            n = max(int(bins_spec), 1)
            use_bins = np.logspace(np.log10(lo), np.log10(hi), n + 1)
    else:
        if isinstance(bins_spec, str):
            use_bins = np.histogram_bin_edges(v, bins=bins_spec.lower(), range=(lo, hi))
        else:
            use_bins = max(int(bins_spec), 1)

    # subset to plotting window (since seaborn has no range=)
    v_plot = v[(v >= lo) & (v <= hi)]

    # ----- draw -----
    fig, ax = plt.subplots(figsize=(8, 6))  # matches your other plots

    g = sns.histplot(
        v_plot,
        bins=use_bins,
        stat="count",
        ax=ax,
        color=BLUE,
        edgecolor="none",
        linewidth=0,
        alpha=1,
    )

    exact_rgba = mpl.colors.to_rgba(BLUE, alpha=1)
    for p in ax.patches:
        p.set_facecolor(exact_rgba)
        p.set_edgecolor("none")
        p.set_alpha(1)

    if logx:
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        xlabel = "Component volume (mm³, log scale)"
    else:
        if xlim is not None:
            ax.set_xlim(lo, hi)
        xlabel = "Component volume (mm³)"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    title = f"{title_base} {title_suffix}".strip()
    if clipped > 0 or nonpos > 0:
        note_bits = []
        if clipped > 0:
            note_bits.append(f"clipped {clipped} outlier{'s' if clipped != 1 else ''}")
        if nonpos > 0:
            note_bits.append(f"excluded {nonpos} nonpositive")
        title += "  (" + ", ".join(note_bits) + ")"
    ax.set_title(title)

    sns.despine(ax=ax)
    fig.tight_layout()

    if out_path:
        if not out_path.lower().endswith(".svg"):
            out_path = os.path.splitext(out_path)[0] + ".svg"
        fig.savefig(out_path, dpi=150, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved histogram to: {out_path}")
    elif show:
        plt.show()
    else:
        plt.close(fig)



# --------------------------- CLI --------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="nnU-Net label connected components (GPU if available) + volume analysis")
    parser.add_argument("root", help="Path to nnU-Net dataset root (must contain labelsTr/)")
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26],
                        help="3D connectivity (default: 26)")

    # Plot behavior
    parser.add_argument("--no-logx", dest="logx", action="store_false",
                        help="Disable log-scaled x-axis (default is log-scale ON)")
    parser.set_defaults(logx=True)
    parser.add_argument("--zoom-pctl", type=float, default=100.0,
                        help="Upper percentile to zoom to for linear x-axis. "
                             "Default 100: disabled (plot full range).")
    parser.add_argument("--xlim", type=str, default=None,
                        help="Explicit x-axis limits as 'MIN,MAX' (overrides --zoom-pctl).")
    parser.add_argument("--bins", default="25",
        help="Number of bins or rule: 'fd', 'auto', 'sturges', 'sqrt'. Default: 25")
    parser.add_argument("--title", type=str, default="Connected Component Volume Distribution",
                        help="Custom plot title (default shown)")

    # Saving/showing
    parser.add_argument("--save", action="store_true",
                        help="Opt-in: save CSV and figures (SVG) to disk instead of showing")
    parser.add_argument("--out", default="cc_volume_out",
                        help="Output directory when --save is used (default: cc_volume_out)")
    parser.add_argument("--save-per-class", action="store_true",
                        help="(Requires --save) Also save one histogram per class")
    parser.add_argument("--csv-name", default="components.csv",
                        help="CSV filename when --save is used (default: components.csv)")

    args = parser.parse_args()

    # Parse user-specified options
    xlim_tuple: Optional[Tuple[float, float]] = None
    if args.xlim:
        xlim_tuple = _parse_xlim(args.xlim)

    try:
        bins_spec: Union[int, str] = int(args.bins)
    except ValueError:
        bins_spec = str(args.bins).lower()

    labels_dir = os.path.join(args.root, "labelsTr")
    if not os.path.isdir(labels_dir):
        print(f"ERROR: {labels_dir} not found. This should be the nnU-Net labels directory.", file=sys.stderr)
        sys.exit(1)

    if args.save:
        os.makedirs(args.out, exist_ok=True)
        csv_path = os.path.join(args.out, args.csv_name)

    nii_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    if len(nii_files) == 0:
        print(f"No .nii.gz files found in {labels_dir}", file=sys.stderr)
        sys.exit(2)

    all_rows: List[Dict[str, Any]] = []
    per_class: Dict[int, List[float]] = {}
    counts_per_scan: List[int] = []

    iterator = tqdm(nii_files, desc="Processing") if TQDM else nii_files
    for f in iterator:
        try:
            rows, per_cls = analyze_label_file(f, connectivity=args.connectivity)
            all_rows.extend(rows)
            for k, v in per_cls.items():
                per_class.setdefault(k, []).extend(v)

            # count of components for this scan (across all classes)
            counts_per_scan.append(len(rows))

        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(f)} due to error: {e}", file=sys.stderr)

    if len(all_rows) == 0:
        print("No components found in dataset (only background?). Exiting.")
        sys.exit(0)

    # Save CSV only if opted-in
    if args.save:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved component table: {csv_path}")

    all_vols_mm3 = [float(row["component_volume_mm3"]) for row in all_rows]

    # Components-per-scan summary (mean|p50|p25|p75)
    cc_mean = float(np.mean(counts_per_scan)) if counts_per_scan else 0.0
    cc_p50  = float(np.percentile(counts_per_scan, 50)) if counts_per_scan else 0.0
    cc_p25  = float(np.percentile(counts_per_scan, 25)) if counts_per_scan else 0.0
    cc_p75  = float(np.percentile(counts_per_scan, 75)) if counts_per_scan else 0.0

    total_mm3 = float(np.sum(all_vols_mm3))
    p50 = float(np.median(all_vols_mm3))
    p5 = float(np.percentile(all_vols_mm3, 5))
    p95 = float(np.percentile(all_vols_mm3, 95))
    std = float(np.std(all_vols_mm3))
    mean = float(np.mean(all_vols_mm3))
    cov = std / mean

    print(
        f"Components: mean {cc_mean:.2f} | p50 {cc_p50:.3f} | p25 {cc_p25:.3f} | p75 {cc_p75:.3f} | "
        f"Total volume: {total_mm3:.2f} mm³ | "
        f"Median: {p50:.3f} mm³ | P5: {p5:.3f} mm³ | P95: {p95:.3f} mm³ | "
        f"Mean: {mean:.3f} mm³ | Std: ±{std:.3f} mm³ | CoV: {cov:.3f} mm³"
    )

    # -------------------- Plotting --------------------
    # Overall histogram
    if args.save:
        plot_distribution(
            all_vols_mm3,
            out_path=os.path.join(args.out, "volumes_overall_hist.svg"),
            show=False,
            logx=args.logx,
            bins=bins_spec,
            xlim=xlim_tuple,
            zoom_pctl=args.zoom_pctl,
            title_base=args.title,
        )
    else:
        plot_distribution(
            all_vols_mm3,
            out_path=None,
            show=True,
            logx=args.logx,
            bins=bins_spec,
            xlim=xlim_tuple,
            zoom_pctl=args.zoom_pctl,
            title_base=args.title,
        )

    # Optional per-class histograms (when saving)
    if args.save and args.save_per_class:
        for cls_id, vols in per_class.items():
            out_svg = os.path.join(args.out, f"volumes_class{cls_id}_hist.svg")
            plot_distribution(
                vols,
                out_path=out_svg,
                show=False,
                logx=args.logx,
                bins=bins_spec,
                xlim=xlim_tuple,
                zoom_pctl=args.zoom_pctl,
                title_suffix=f"(class {cls_id})",
                title_base=args.title,
            )

if __name__ == "__main__": 
    main()

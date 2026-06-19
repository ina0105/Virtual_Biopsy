

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_semisupervised_regression import (
    UNet3D,
    load_metadata,
    _normalize_channel,
    extract_roi_mean,
)

DEFAULT_SRI24   = Path(__file__).resolve().parents[2] / "SRI24_modalities"
PATCH_SIZE      = (160, 160, 128)
BASE_CHANNELS   = 32
DEPTH           = 4
BRAIN_THRESHOLD = 100.0



def load_brain_mask(pid: str, sri24_dir: Path) -> np.ndarray | None:
    t1g = sri24_dir / pid / f"{pid}T1G_SRI24.nii.gz"
    if not t1g.exists():
        return None
    return nib.load(str(t1g)).get_fdata(dtype=np.float32) > BRAIN_THRESHOLD


def apply_brain_mask_uint8(pred: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    pred_clipped = np.clip(pred, 0.0, 1.0)
    if mask is not None:
        pred_clipped = np.where(mask, pred_clipped, 0.0)
    return (pred_clipped * 255).astype(np.uint8)



def _pad_to_patch(vol: np.ndarray, patch: tuple):
    pads = [(0, 0)]
    for s, p in zip(vol.shape[1:], patch):
        diff = max(0, p - s)
        pads.append((diff // 2, diff - diff // 2))
    return np.pad(vol, pads, mode="constant"), pads


def sliding_window_inference(model: nn.Module, image: np.ndarray, patch_size: tuple,
                              overlap: float = 0.5, device: torch.device = torch.device("cpu")) -> np.ndarray:
    model.eval()
    padded, pads = _pad_to_patch(image, patch_size)
    C, X, Y, Z = padded.shape
    px, py, pz = patch_size
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)

    pred_sum   = np.zeros((X, Y, Z), dtype=np.float32)
    weight_sum = np.zeros((X, Y, Z), dtype=np.float32)

    gx = np.exp(-((np.arange(px) - px / 2) ** 2) / (2 * (px / 4) ** 2))
    gy = np.exp(-((np.arange(py) - py / 2) ** 2) / (2 * (py / 4) ** 2))
    gz = np.exp(-((np.arange(pz) - pz / 2) ** 2) / (2 * (pz / 4) ** 2))
    gauss = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]

    xs = sorted(set(list(range(0, max(1, X - px + 1), stride[0])) + [max(0, X - px)]))
    ys = sorted(set(list(range(0, max(1, Y - py + 1), stride[1])) + [max(0, Y - py)]))
    zs = sorted(set(list(range(0, max(1, Z - pz + 1), stride[2])) + [max(0, Z - pz)]))

    with torch.no_grad():
        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    patch = padded[:, x0:x0+px, y0:y0+py, z0:z0+pz]
                    t = torch.from_numpy(patch[None]).to(device)
                    out = model(t).squeeze().cpu().numpy()
                    pred_sum[x0:x0+px, y0:y0+py, z0:z0+pz]   += out * gauss
                    weight_sum[x0:x0+px, y0:y0+py, z0:z0+pz] += gauss

    pred = pred_sum / np.clip(weight_sum, 1e-8, None)
    orig = image.shape[1:]
    sx, sy, sz = pads[1][0], pads[2][0], pads[3][0]
    return pred[sx:sx+orig[0], sy:sy+orig[1], sz:sz+orig[2]]



def find_loo_fold(pid: str, dataset_dir: Path, patients_info: dict) -> int:
    from sklearn.model_selection import LeaveOneOut
    import glob as _glob
    labels_dir = dataset_dir / "labels"
    images_dir = dataset_dir / "images"


    candidates = ["_biopsy_purity_absneg_innertumor_enhancing"]
    first_p = next(iter(patients_info))
    extra = [
        Path(f).stem.replace(first_p, "").replace(".nii", "")
        for f in _glob.glob(str(labels_dir / f"{first_p}_biopsy_purity_absneg*.nii.gz"))
    ]
    candidates = extra + candidates  # prefer dataset-specific suffix

    combined_suffix = None
    for suf in candidates:
        if any((labels_dir / f"{p}{suf}.nii.gz").exists() for p in patients_info):
            combined_suffix = suf
            break

    if combined_suffix is None:
        raise ValueError(f"No combined label files found in {labels_dir}")

    fully_labeled = sorted(
        p for p, info in patients_info.items()
        if info.get("has_labels", False)
        and (images_dir / f"{p}.nii.gz").exists()
        and (labels_dir / f"{p}{combined_suffix}.nii.gz").exists()
    )
    for fi, (_, va_idx) in enumerate(LeaveOneOut().split(range(len(fully_labeled)))):
        if fully_labeled[va_idx[0]] == pid:
            return fi
    raise ValueError(f"{pid} not found in fully-labeled list: {fully_labeled}")



def _best_slices(biopsy_df, vol_shape, plane, n=6):
    ax_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    size = vol_shape[ax_map[plane]]
    if biopsy_df is not None and not biopsy_df.empty:
        col = {"axial": "sri24_vox_z", "coronal": "sri24_vox_y", "sagittal": "sri24_vox_x"}[plane]
        if col in biopsy_df.columns:
            center = int(biopsy_df[col].median())
            half = (n // 2) * max(1, size // (n * 2))
            return list(np.linspace(max(0, center - half), min(size - 1, center + half), n, dtype=int))
    return list(np.linspace(size // 8, 7 * size // 8, n, dtype=int))


def _get_slice(vol, idx, plane):
    if plane == "axial":    return vol[:, :, idx]
    if plane == "coronal":  return vol[:, idx, :]
    if plane == "sagittal": return vol[idx, :, :]


def visualize(pid, t1_vol, pred_vol, biopsy_df, seg_vol, out_dir, planes, n_slices, suffix):
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_map = {
        "axial":    ("sri24_vox_x", "sri24_vox_y", "sri24_vox_z"),
        "coronal":  ("sri24_vox_x", "sri24_vox_z", "sri24_vox_y"),
        "sagittal": ("sri24_vox_y", "sri24_vox_z", "sri24_vox_x"),
    }
    for plane in planes:
        slices = _best_slices(biopsy_df, t1_vol.shape, plane, n_slices)
        fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 7))
        fig.suptitle(f"{pid} — {plane} | {suffix} purity", fontsize=13)
        bx_h, bx_v, bx_sl = coord_map[plane]
        im = None
        for col, sl_idx in enumerate(slices):
            t1_sl   = _get_slice(t1_vol,   sl_idx, plane)
            pred_sl = _get_slice(pred_vol, sl_idx, plane)
            ax = axes[0, col]
            t1_norm = np.clip(t1_sl, 0, None)
            t1_norm = t1_norm / (t1_norm.max() + 1e-8)
            ax.imshow(t1_norm.T, cmap="gray", origin="lower", aspect="equal")
            brain_mask = t1_norm > 0.02
            pred_masked = np.where(brain_mask, pred_sl, np.nan)
            im = ax.imshow(pred_masked.T, cmap="jet", vmin=0, vmax=1,
                           alpha=0.55, origin="lower", aspect="equal")
            if biopsy_df is not None and not biopsy_df.empty:
                if all(c in biopsy_df.columns for c in [bx_h, bx_v, bx_sl]):
                    tol = max(2, int(t1_vol.shape[0] / 40))
                    nearby = biopsy_df[np.abs(biopsy_df[bx_sl] - sl_idx) <= tol]
                    if not nearby.empty:
                        ax.scatter(nearby[bx_h], nearby[bx_v],
                                   c=nearby["PAMES_purity"], cmap="RdYlGn",
                                   vmin=0, vmax=1, s=60, edgecolors="white",
                                   linewidths=0.8, zorder=5)
            ax.set_title(f"slice {sl_idx}", fontsize=8)
            ax.axis("off")
            ax2 = axes[1, col]
            ax2.imshow(pred_sl.T, cmap="jet", vmin=0, vmax=1, origin="lower", aspect="equal")
            if seg_vol is not None:
                seg_sl = _get_slice(seg_vol, sl_idx, plane)
                ax2.contour((seg_sl > 0).astype(float).T, levels=[0.5],
                            colors=["cyan"], linewidths=0.8, alpha=0.8)
            ax2.set_title("pred only", fontsize=8)
            ax2.axis("off")
        if im is not None:
            plt.colorbar(im, ax=axes[0, :].tolist(), shrink=0.6, label="Predicted purity")
        handles = [mpatches.Patch(facecolor="cyan", label="tumor seg (contour)")]
        if biopsy_df is not None and not biopsy_df.empty:
            handles += [mpatches.Patch(facecolor="red",   label="GT purity high"),
                        mpatches.Patch(facecolor="green", label="GT purity low")]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{pid}_{plane}_{suffix}_prediction.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {pid}_{plane}_{suffix}_prediction.png")


def plot_biopsy_locations(pid, t1_vol, pred_vol, biopsy_df, out_dir, suffix, crop=30):
    if biopsy_df is None or biopsy_df.empty:
        return
    needed = {"sri24_vox_x", "sri24_vox_y", "sri24_vox_z", "PAMES_purity"}
    if not needed.issubset(biopsy_df.columns):
        return
    df = biopsy_df.copy().reset_index(drop=True)
    X, Y, Z = pred_vol.shape
    n = len(df)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"{pid} — biopsy locations ({suffix})", fontsize=18)
    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        cx = int(np.clip(round(row["sri24_vox_x"]), 0, X - 1))
        cy = int(np.clip(round(row["sri24_vox_y"]), 0, Y - 1))
        cz = int(np.clip(round(row["sri24_vox_z"]), 0, Z - 1))
        t1_sl   = t1_vol[:, :, cz]
        pred_sl = pred_vol[:, :, cz]
        x0 = max(0, cx - crop); x1 = min(X, cx + crop)
        y0 = max(0, cy - crop); y1 = min(Y, cy + crop)
        t1_crop = t1_sl[x0:x1, y0:y1]
        pred_crop = pred_sl[x0:x1, y0:y1]
        t1_norm = np.clip(t1_crop, 0, None)
        t1_norm = t1_norm / (t1_norm.max() + 1e-8)
        ax.imshow(t1_norm.T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(pred_crop.T, cmap="hot", vmin=0, vmax=1, alpha=0.55, origin="lower", aspect="equal")
        ax.scatter([cx - x0], [cy - y0], s=80, c="cyan", edgecolors="white",
                   linewidths=1.0, zorder=5, marker="+")
        true_val = row["PAMES_purity"]
        pred_val = float(pred_vol[cx, cy, cz])
        lbl = row.get("biopsy_id", str(i))
        color = "#d73027" if true_val > 0.5 else "#4575b4"
        ax.set_title(f"{lbl}  true={true_val:.2f}  pred={pred_val:.2f}", fontsize=13, color=color)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{pid}_{suffix}_biopsy_locations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pid}_{suffix}_biopsy_locations.png")


def plot_biopsy_predictions(pid, pred_vol, biopsy_df, out_dir, suffix):
    if biopsy_df is None or biopsy_df.empty:
        print("  [skip biopsy plot] no biopsy data")
        return
    needed = {"sri24_vox_x", "sri24_vox_y", "sri24_vox_z", "PAMES_purity"}
    if not needed.issubset(biopsy_df.columns):
        return
    df = biopsy_df.copy().reset_index(drop=True)
    X, Y, Z = pred_vol.shape
    preds = []
    for _, row in df.iterrows():
        x = int(np.clip(round(row["sri24_vox_x"]), 0, X - 1))
        y = int(np.clip(round(row["sri24_vox_y"]), 0, Y - 1))
        z = int(np.clip(round(row["sri24_vox_z"]), 0, Z - 1))
        preds.append(float(pred_vol[x, y, z]))
    df["pred_purity"] = preds
    true_vals = df["PAMES_purity"].values
    pred_vals = df["pred_purity"].values
    labels    = df["biopsy_id"].values if "biopsy_id" in df.columns else [str(i) for i in range(len(df))]
    colors    = ["#d73027" if v > 0.5 else "#4575b4" for v in true_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{pid} — biopsy predictions ({suffix})", fontsize=13)
    ax1.scatter(true_vals, pred_vals, c=colors, s=80, edgecolors="k", linewidths=0.6, zorder=3)
    for i, lbl in enumerate(labels):
        ax1.annotate(lbl, (true_vals[i], pred_vals[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    lim = (-0.05, 1.05)
    ax1.plot(lim, lim, "--", color="gray", linewidth=1, label="perfect")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("True purity (PAMES)"); ax1.set_ylabel("Predicted purity")
    ax1.set_title("Predicted vs True"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    order = np.argsort(true_vals)
    x_pos = np.arange(len(order)); width = 0.35
    ax2.bar(x_pos - width/2, true_vals[order], width, label="True (PAMES)",
            color=[colors[i] for i in order], alpha=0.85, edgecolor="k", linewidth=0.5)
    ax2.bar(x_pos + width/2, pred_vals[order], width, label="Predicted",
            color="gray", alpha=0.75, edgecolor="k", linewidth=0.5)
    ax2.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([labels[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Purity")
    ax2.set_title("Per-biopsy (sorted by true purity)"); ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{pid}_{suffix}_biopsy_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pid}_{suffix}_biopsy_predictions.png")
    csv_path = out_dir / f"{pid}_{suffix}_biopsy_predictions.csv"
    df[["biopsy_id", "PAMES_purity", "pred_purity"]].to_csv(csv_path, index=False)
    print(f"  Saved: {pid}_{suffix}_biopsy_predictions.csv")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient",       required=True)
    p.add_argument("--work-dir",      type=Path, required=True)
    p.add_argument("--dataset-dir",   type=Path, required=True)
    p.add_argument("--out-dir",       type=Path, required=True)
    p.add_argument("--sri24-dir",     type=Path, default=DEFAULT_SRI24)
    p.add_argument("--dropout",       type=float, default=0.0,
                   help="Dropout rate. Auto-detected from checkpoint if not set.")
    p.add_argument("--attention",     default="cbam",
                   choices=["none", "channel", "spatial", "cbam"])
    p.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    p.add_argument("--depth",         type=int, default=DEPTH)
    p.add_argument("--fold",          type=int, default=None,
                   help="Explicit fold index. Use --loo to auto-detect.")
    p.add_argument("--loo",           action="store_true",
                   help="Auto-detect held-out fold for this patient.")
    p.add_argument("--overlap",       type=float, default=0.5)
    p.add_argument("--device",        default=None)
    p.add_argument("--no-save-nifti", action="store_true")
    p.add_argument("--planes",        nargs="+", default=["axial", "coronal", "sagittal"],
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--n-slices",      type=int, default=6)
    p.add_argument("--suffix",        default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pid = args.patient
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    img_path = args.dataset_dir / "images" / f"{pid}.nii.gz"
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")
    nii   = nib.load(str(img_path))
    raw   = nii.get_fdata(dtype=np.float32)
    image = np.moveaxis(raw, -1, 0)
    print(f"Loaded {pid}: shape={image.shape}")

    global_mods, patients_info = load_metadata(args.dataset_dir)
    norm = np.empty_like(image)
    for ch in range(image.shape[0]):
        norm[ch] = _normalize_channel(image[ch])

    # Resolve fold
    if args.loo:
        fold = find_loo_fold(pid, args.dataset_dir, patients_info)
        print(f"  LOO mode: using fold_{fold} (held-out fold for {pid})")
    elif args.fold is not None:
        fold = args.fold
    else:
        sys.exit("Specify --loo or --fold N")

    ckpt_name = f"best_model_attn-{args.attention}.pt"
    ckpt_path = args.work_dir / f"fold_{fold}" / ckpt_name
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    # Build model and load checkpoint
    in_channels = image.shape[0]
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    # Auto-detect dropout: if second Conv is at block.4 (not block.3), dropout was used
    mstate = state["model_state"]
    has_dropout = any("block.4.weight" in k for k in mstate.keys())
    dropout = args.dropout if not has_dropout else max(args.dropout, 0.2)
    if has_dropout and args.dropout == 0.0:
        print(f"  Auto-detected dropout in checkpoint — building model with dropout=0.2")
    model = UNet3D(in_channels=in_channels, base_channels=args.base_channels,
                   depth=args.depth, dropout=dropout, attention=args.attention).to(device)
    model.load_state_dict(mstate)
    print(f"  Loaded fold_{fold} (val_loss={state.get('val_loss', float('nan')):.5f})")

    pred_vol = sliding_window_inference(model, norm, PATCH_SIZE, args.overlap, device)
    print(f"  Pred range [{pred_vol.min():.3f}, {pred_vol.max():.3f}]")

    # Brain mask + segmentation
    brain_mask = load_brain_mask(pid, args.sri24_dir)
    seg_path = Path(__file__).resolve().parents[2] / "TumorSynth_Outputs" / f"{pid}_tumor_mask.nii.gz"
    seg_vol = nib.load(str(seg_path)).get_fdata(dtype=np.float32) if seg_path.exists() else None

    # Biopsy GT
    biopsy_df = None
    biopsy_csv = args.dataset_dir / "biopsy_metadata.csv"
    if biopsy_csv.exists():
        df = pd.read_csv(biopsy_csv)
        sub = df[df["mri_patient"] == pid]
        if not sub.empty:
            biopsy_df = sub.reset_index(drop=True)
            print(f"  Found {len(biopsy_df)} biopsy points")

    suffix = args.suffix or f"unet_{args.attention}"

    # Save NIfTI
    if not args.no_save_nifti:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        affine = nii.affine
        nib.save(nib.Nifti1Image(np.clip(pred_vol, 0, 1).astype(np.float32), affine),
                 str(args.out_dir / f"{pid}_purity_{suffix}.nii.gz"))
        uint8 = apply_brain_mask_uint8(pred_vol, brain_mask)
        nib.save(nib.Nifti1Image(uint8, affine),
                 str(args.out_dir / f"{pid}_purity_{suffix}_uint8.nii.gz"))
        print(f"  Saved NIfTIs")

        if biopsy_df is not None and not biopsy_df.empty:
            needed = {"sri24_vox_x", "sri24_vox_y", "sri24_vox_z"}
            if needed.issubset(biopsy_df.columns):
                marker_vol = np.zeros(pred_vol.shape, dtype=np.int16)
                X, Y, Z = pred_vol.shape
                radius = 4
                for b_idx, (_, row) in enumerate(biopsy_df.iterrows(), start=1):
                    cx = int(np.clip(round(row["sri24_vox_x"]), 0, X - 1))
                    cy = int(np.clip(round(row["sri24_vox_y"]), 0, Y - 1))
                    cz = int(np.clip(round(row["sri24_vox_z"]), 0, Z - 1))
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                xi = cx + dx; yi = cy + dy
                                if 0 <= xi < X and 0 <= yi < Y:
                                    marker_vol[xi, yi, cz] = b_idx
                nib.save(nib.Nifti1Image(marker_vol, affine),
                         str(args.out_dir / f"{pid}_biopsy_markers_{suffix}.nii.gz"))

    t1_idx = global_mods.index("T01") if "T01" in global_mods else 0
    visualize(pid, image[t1_idx], pred_vol, biopsy_df, seg_vol,
              args.out_dir, args.planes, args.n_slices, suffix)
    plot_biopsy_locations(pid, image[t1_idx], pred_vol, biopsy_df, args.out_dir, suffix)
    plot_biopsy_predictions(pid, pred_vol, biopsy_df, args.out_dir, suffix)
    print("Done.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_semisupervised_regression import (
    UNet3D,
    load_metadata,
    _normalize_channel,
)
from predict_and_visualize_unet import (
    find_loo_fold,
    _pad_to_patch,
)

DEFAULT_TUMORSYNTH = Path(__file__).resolve().parents[2] / "TumorSynth_Outputs"
PATCH_SIZE         = (160, 160, 128)
BASE_CHANNELS      = 32
DEPTH              = 4



def _sliding_window_mean(model, padded, patch_size, stride, tm_pad, device):
    X, Y, Z = padded.shape[1:]
    px, py, pz = patch_size
    xs = sorted(set(range(0, max(1, X - px + 1), stride[0])) | {max(0, X - px)})
    ys = sorted(set(range(0, max(1, Y - py + 1), stride[1])) | {max(0, Y - py)})
    zs = sorted(set(range(0, max(1, Z - pz + 1), stride[2])) | {max(0, Z - pz)})
    total, count = 0.0, 0
    with torch.no_grad():
        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    patch = padded[:, x0:x0+px, y0:y0+py, z0:z0+pz].copy()
                    t = torch.from_numpy(patch[None]).float().to(device)
                    pred = model(t).squeeze().cpu().numpy()
                    if tm_pad is not None:
                        roi = tm_pad[x0:x0+px, y0:y0+py, z0:z0+pz] > 0
                        if roi.sum() > 0:
                            total += float(pred[roi].mean())
                            count += 1
                            continue
                    total += float(pred.mean())
                    count += 1
    return total / max(count, 1)


def compute_channel_importance(
    model: torch.nn.Module,
    norm_image: np.ndarray,
    patch_size: tuple,
    tumor_mask: np.ndarray | None,
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model.eval()
    C = norm_image.shape[0]
    padded, pads = _pad_to_patch(norm_image, patch_size)
    stride = tuple(max(1, int(p * (1 - ov))) for p, ov in zip(patch_size, [overlap] * 3))

    tm_pad = None
    if tumor_mask is not None:
        tm_pad = np.pad(
            tumor_mask,
            [(pads[i + 1][0], pads[i + 1][1]) for i in range(3)],
            mode="constant",
        )

    baseline = _sliding_window_mean(model, padded, patch_size, stride, tm_pad, device)

    importance = np.zeros(C, dtype=np.float64)
    for c in range(C):
        ablated = padded.copy()
        ablated[c] = 0.0
        ablated_purity = _sliding_window_mean(
            model, ablated, patch_size, stride, tm_pad, device
        )
        importance[c] = abs(baseline - ablated_purity)

    return importance



def _bar_chart(
    importance: np.ndarray,
    channel_names: List[str],
    title: str,
    out_path: Path,
    yerr: np.ndarray | None = None,
) -> None:
    n_ch = len(channel_names)
    fig, ax = plt.subplots(figsize=(max(7, n_ch * 0.75), 5))
    x = np.arange(n_ch)
    rel = importance / (importance.max() + 1e-12)
    colors = plt.cm.viridis(rel)
    ax.bar(
        x, importance,
        color=colors, edgecolor="k", linewidth=0.5,
        yerr=yerr, capsize=4 if yerr is not None else 0,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |∂purity / ∂channel|", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Relative importance", fraction=0.03, pad=0.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir",       type=Path, required=True,
                   help="Work dir containing fold_N/best_model_attn-*.pt checkpoints.")
    p.add_argument("--dataset-dir",    type=Path, required=True)
    p.add_argument("--out-dir",        type=Path, required=True)
    p.add_argument("--tumorsynth-dir", type=Path, default=DEFAULT_TUMORSYNTH)
    p.add_argument("--patients",       nargs="+", default=None,
                   help="Patient IDs to process (default: all with LOO checkpoints).")
    p.add_argument("--attention",      default="cbam",
                   choices=["none", "channel", "spatial", "cbam"])
    p.add_argument("--base-channels",  type=int, default=BASE_CHANNELS)
    p.add_argument("--depth",          type=int, default=DEPTH)
    p.add_argument("--dropout",        type=float, default=0.0)
    p.add_argument("--overlap",        type=float, default=0.5)
    p.add_argument("--patch-size",     type=int, nargs=3, default=list(PATCH_SIZE))
    p.add_argument("--device",         default=None)
    p.add_argument("--mean-only",      action="store_true",
                   help="Skip per-patient PNG charts; only save group mean + CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    channel_names, patients_info = load_metadata(args.dataset_dir)
    patch_size = tuple(args.patch_size)
    ckpt_name = f"best_model_attn-{args.attention}.pt"

    fully_labeled = sorted(
        p for p, info in patients_info.items()
        if info.get("has_labels", False)
        and (args.dataset_dir / "images" / f"{p}.nii.gz").exists()
    )
    patients = args.patients or fully_labeled
    print(f"Patients : {patients}")
    print(f"Channels : {channel_names}")
    print(f"Attention: {args.attention}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_importance: Dict[str, np.ndarray] = {}

    for pid in patients:
        print(f"\n── {pid} ──")

        img_path = args.dataset_dir / "images" / f"{pid}.nii.gz"
        if not img_path.exists():
            print(f"  [SKIP] image not found: {img_path}")
            continue

        raw = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        image = np.moveaxis(raw, -1, 0)  # (C, X, Y, Z)
        norm = np.empty_like(image)
        for ch in range(image.shape[0]):
            norm[ch] = _normalize_channel(image[ch])
        print(f"  Image: {image.shape}")

        try:
            fold = find_loo_fold(pid, args.dataset_dir, patients_info)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        ckpt_path = args.work_dir / f"fold_{fold}" / ckpt_name
        if not ckpt_path.exists():
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        mstate = state["model_state"]
        has_dropout = any("block.4.weight" in k for k in mstate.keys())
        dropout = max(args.dropout, 0.2) if has_dropout else args.dropout

        model = UNet3D(
            in_channels=image.shape[0],
            base_channels=args.base_channels,
            depth=args.depth,
            dropout=dropout,
            attention=args.attention,
        ).to(device)
        model.load_state_dict(mstate)
        print(f"  Loaded fold_{fold}  val_loss={state.get('val_loss', float('nan')):.5f}")

        tm_path = args.tumorsynth_dir / f"{pid}_tumor_mask.nii.gz"
        if tm_path.exists():
            tumor_mask = nib.load(str(tm_path)).get_fdata(dtype=np.float32)
            print(f"  Tumor ROI: {int((tumor_mask > 0).sum())} voxels")
        else:
            tumor_mask = None
            print(f"  No tumor mask — using full volume as ROI")

        imp = compute_channel_importance(
            model, norm, patch_size, tumor_mask, args.overlap, device
        )
        all_importance[pid] = imp

        imp_str = "  ".join(f"{cn}={v:.4f}" for cn, v in zip(channel_names, imp))
        print(f"  {imp_str}")

        if not args.mean_only:
            _bar_chart(
                imp, channel_names,
                title=f"{pid} — modality importance  (attn={args.attention})",
                out_path=args.out_dir / f"channel_importance_{pid}.png",
            )

    if not all_importance:
        print("\nNo patients processed — check work-dir and checkpoint names.")
        return

    # CSV summary
    df = pd.DataFrame(all_importance, index=channel_names).T
    df.index.name = "patient"
    csv_path = args.out_dir / "channel_importance_per_patient.csv"
    df.to_csv(csv_path)
    print(f"\nSaved: {csv_path.name}")

    # Group-level chart
    arr = np.array(list(all_importance.values()))  # (N, C)
    mean_imp = arr.mean(axis=0)
    std_imp  = arr.std(axis=0)
    _bar_chart(
        mean_imp, channel_names,
        title=(
            f"Group mean ± std — modality importance\n"
            f"n={len(all_importance)} patients  |  attn={args.attention}"
        ),
        out_path=args.out_dir / "channel_importance_group.png",
        yerr=std_imp,
    )
    print("Done.")


if __name__ == "__main__":
    main()

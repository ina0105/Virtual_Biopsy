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
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_semisupervised_regression import (
    UNet3D,
    SpatialAttention,
    load_metadata,
    _normalize_channel,
)
from predict_and_visualize_unet import (
    find_loo_fold,
    _pad_to_patch,
)

DEFAULT_SRI24 = Path(__file__).resolve().parents[2] / "SRI24_modalities"
PATCH_SIZE    = (160, 160, 128)
BASE_CHANNELS = 32
DEPTH         = 4



class SpatialAttnCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        self._buffers: Dict[str, torch.Tensor] = {}
        self._handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, SpatialAttention):
                handle = mod.sigmoid.register_forward_hook(
                    self._make_hook(name)
                )
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(mod, inp, out):
            self._buffers[name] = out.detach().cpu()
        return hook

    def clear(self) -> None:
        self._buffers.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def combined_map(self, target_shape: tuple) -> np.ndarray | None:
        if not self._buffers:
            return None
        maps = []
        for t in self._buffers.values():
            # t: (1, 1, H', W', D')
            if t.shape[-3:] != target_shape:
                t = F.interpolate(
                    t, size=target_shape, mode="trilinear", align_corners=False
                )
            maps.append(t.squeeze().numpy())  # (H', W', D')
        return np.mean(maps, axis=0)



def sliding_window_attn(
    model: torch.nn.Module,
    norm_image: np.ndarray,
    patch_size: tuple,
    capture: SpatialAttnCapture,
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model.eval()
    padded, pads = _pad_to_patch(norm_image, patch_size)
    X, Y, Z = padded.shape[1:]
    px, py, pz = patch_size
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)

    attn_sum  = np.zeros((X, Y, Z), dtype=np.float64)
    weight_sum = np.zeros((X, Y, Z), dtype=np.float64)

    gx = np.exp(-((np.arange(px) - px / 2) ** 2) / (2 * (px / 4) ** 2))
    gy = np.exp(-((np.arange(py) - py / 2) ** 2) / (2 * (py / 4) ** 2))
    gz = np.exp(-((np.arange(pz) - pz / 2) ** 2) / (2 * (pz / 4) ** 2))
    gauss = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]

    xs = sorted(set(range(0, max(1, X - px + 1), stride[0])) | {max(0, X - px)})
    ys = sorted(set(range(0, max(1, Y - py + 1), stride[1])) | {max(0, Y - py)})
    zs = sorted(set(range(0, max(1, Z - pz + 1), stride[2])) | {max(0, Z - pz)})

    with torch.no_grad():
        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    patch = padded[:, x0:x0+px, y0:y0+py, z0:z0+pz].copy()
                    t = torch.from_numpy(patch[None]).float().to(device)
                    capture.clear()
                    model(t)

                    amap = capture.combined_map((px, py, pz))
                    if amap is None:
                        continue

                    attn_sum[x0:x0+px, y0:y0+py, z0:z0+pz]   += amap * gauss
                    weight_sum[x0:x0+px, y0:y0+py, z0:z0+pz] += gauss

    result = attn_sum / np.clip(weight_sum, 1e-8, None)
    orig = norm_image.shape[1:]
    sx, sy, sz = pads[1][0], pads[2][0], pads[3][0]
    return result[sx:sx+orig[0], sy:sy+orig[1], sz:sz+orig[2]].astype(np.float32)



def _get_slice(vol: np.ndarray, idx: int, plane: str) -> np.ndarray:
    if plane == "axial":    return vol[:, :, idx]
    if plane == "coronal":  return vol[:, idx, :]
    return vol[idx, :, :]


def _best_center(attn_vol: np.ndarray, plane: str, n: int = 6) -> List[int]:
    ax = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
    proj = attn_vol.mean(axis=tuple(a for a in range(3) if a != ax))
    center = int(np.argmax(proj))
    size = attn_vol.shape[ax]
    half = (n // 2) * max(1, size // (n * 2))
    return list(np.linspace(
        max(0, center - half), min(size - 1, center + half), n, dtype=int
    ))


def visualize_attn(
    pid: str,
    t1_vol: np.ndarray,
    attn_vol: np.ndarray,
    out_dir: Path,
    plane: str,
    n_slices: int = 6,
    suffix: str = "spatial_attn",
) -> None:
    slices = _best_center(attn_vol, plane, n_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3.5))
    fig.suptitle(f"{pid} — spatial attention ({plane})", fontsize=12)

    for col, sl_idx in enumerate(slices):
        t1_sl   = _get_slice(t1_vol,   sl_idx, plane)
        attn_sl = _get_slice(attn_vol, sl_idx, plane)
        ax = axes[col]
        t1_norm = np.clip(t1_sl, 0, None)
        t1_norm = t1_norm / (t1_norm.max() + 1e-8)
        ax.imshow(t1_norm.T, cmap="gray", origin="lower", aspect="equal")
        brain_mask = t1_norm > 0.02
        attn_masked = np.where(brain_mask, attn_sl, np.nan)
        ax.imshow(attn_masked.T, cmap="jet", vmin=0, vmax=1,
                  alpha=0.6, origin="lower", aspect="equal")
        ax.set_title(f"sl {sl_idx}", fontsize=8)
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label="Attention weight", fraction=0.05)
    plt.tight_layout()
    out_path = out_dir / f"{suffix}_{pid}_{plane}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir",       type=Path, required=True)
    p.add_argument("--dataset-dir",    type=Path, required=True)
    p.add_argument("--out-dir",        type=Path, required=True)
    p.add_argument("--sri24-dir",      type=Path, default=DEFAULT_SRI24)
    p.add_argument("--patients",       nargs="+", default=None)
    p.add_argument("--attention",      default="spatial",
                   choices=["spatial", "cbam"],
                   help="Attention type of the checkpoint (must include SpatialAttention).")
    p.add_argument("--base-channels",  type=int, default=BASE_CHANNELS)
    p.add_argument("--depth",          type=int, default=DEPTH)
    p.add_argument("--dropout",        type=float, default=0.0)
    p.add_argument("--overlap",        type=float, default=0.5)
    p.add_argument("--patch-size",     type=int, nargs=3, default=list(PATCH_SIZE))
    p.add_argument("--planes",         nargs="+",
                   default=["axial", "coronal", "sagittal"],
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--n-slices",       type=int, default=6)
    p.add_argument("--device",         default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    _, patients_info = load_metadata(args.dataset_dir)
    patch_size = tuple(args.patch_size)
    ckpt_name  = f"best_model_attn-{args.attention}.pt"

    fully_labeled = sorted(
        p for p, info in patients_info.items()
        if info.get("has_labels", False)
        and (args.dataset_dir / "images" / f"{p}.nii.gz").exists()
    )
    patients = args.patients or fully_labeled
    print(f"Patients : {patients}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_attn: Dict[str, np.ndarray] = {}
    reference_affine = None

    for pid in patients:
        print(f"\n── {pid} ──")

        img_path = args.dataset_dir / "images" / f"{pid}.nii.gz"
        if not img_path.exists():
            print(f"  [SKIP] image not found")
            continue

        nii  = nib.load(str(img_path))
        raw  = nii.get_fdata(dtype=np.float32)
        image = np.moveaxis(raw, -1, 0)  # (C, X, Y, Z)
        norm = np.empty_like(image)
        for ch in range(image.shape[0]):
            norm[ch] = _normalize_channel(image[ch])
        if reference_affine is None:
            reference_affine = nii.affine

        try:
            fold = find_loo_fold(pid, args.dataset_dir, patients_info)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        ckpt_path = args.work_dir / f"fold_{fold}" / ckpt_name
        if not ckpt_path.exists():
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        state  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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

        capture = SpatialAttnCapture(model)
        attn_vol = sliding_window_attn(
            model, norm, patch_size, capture, args.overlap, device
        )
        capture.remove()

        print(f"  Attention range [{attn_vol.min():.3f}, {attn_vol.max():.3f}]")
        all_attn[pid] = attn_vol

        nib.save(
            nib.Nifti1Image(attn_vol, nii.affine),
            str(args.out_dir / f"spatial_attn_{pid}.nii.gz"),
        )
        print(f"  Saved: spatial_attn_{pid}.nii.gz")

        t1g_path = args.sri24_dir / pid / f"{pid}T1G_SRI24.nii.gz"
        if t1g_path.exists():
            t1_vol = nib.load(str(t1g_path)).get_fdata(dtype=np.float32)
        else:
            t1_vol = image[1] if image.shape[0] > 1 else image[0]

        for plane in args.planes:
            visualize_attn(pid, t1_vol, attn_vol, args.out_dir,
                           plane, args.n_slices)

    if not all_attn:
        print("\nNo patients processed.")
        return

    arr = np.stack(list(all_attn.values()), axis=0)  # (N, X, Y, Z)
    mean_attn = arr.mean(axis=0).astype(np.float32)

    if reference_affine is not None:
        nib.save(
            nib.Nifti1Image(mean_attn, reference_affine),
            str(args.out_dir / "spatial_attn_group_mean.nii.gz"),
        )
        print(f"\nSaved: spatial_attn_group_mean.nii.gz")

    t1g_p01 = args.sri24_dir / "P01" / "P01T1G_SRI24.nii.gz"
    if t1g_p01.exists():
        t1_atlas = nib.load(str(t1g_p01)).get_fdata(dtype=np.float32)
        for plane in args.planes:
            visualize_attn(
                f"group_mean (n={len(all_attn)})", t1_atlas, mean_attn,
                args.out_dir, plane, args.n_slices,
                suffix="spatial_attn_group",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

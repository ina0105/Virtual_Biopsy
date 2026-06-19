

import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import re
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parents[1] / 'Virtual_biopsy'
METADATA_CSV = BASE_DIR / 'FRONTIER.QC.filtered.metadata_mult_20200911.csv'
ROIVOX_DIR = BASE_DIR / 'ROIVox'
PIPELINE_DIR = BASE_DIR / 'SRI24_modalities'
MRI_ORIG_DIR = BASE_DIR / 'MRI'
DEFAULT_CNN_DIR = BASE_DIR / 'CNN_dataset'

ALL_MODALITIES = ['T01', 'T1G', 'T02', 'FLR', 'ADC', 'BFa', 'BFd', 'BVd', 'C18', 'dFA', 'FET', 'S24', 'SR9']

NON_PET_MODALITIES = ['T01', 'T1G', 'T02', 'FLR', 'ADC', 'BFa', 'BFd', 'BVd', 'dFA']

ALL_PATIENTS = [f'P{i:02d}' for i in range(1, 21)]

SRI24_SHAPE = (240, 240, 155)


def load_metadata():
    df = pd.read_csv(METADATA_CSV)
    def parse_bnum(b):
        m = re.match(r'S(\d+)', str(b))
        return int(m.group(1)) if m else None
    df['biopsy_num'] = df['Biopsy'].apply(parse_bnum)
    return df


def build_patient_mapping(metadata_df):
    metadata_patients = set(metadata_df['Patient'].dropna().astype(str))
    mapping = {}
    for mri_patient_id in ALL_PATIENTS:
        vumc_id = f"VUmc-{int(mri_patient_id[1:]):02d}"
        if vumc_id in metadata_patients:
            mapping[mri_patient_id] = vumc_id
    return mapping


def parse_lta(lta_path):
    with open(lta_path) as f:
        lines = f.readlines()
    result = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '1 4 4':
            mat = []
            for j in range(4):
                i += 1
                mat.append([float(x) for x in lines[i].strip().split()])
            result['matrix'] = np.array(mat)
        if 'src volume info' in line:
            result['src'] = _parse_vol_info(lines, i)
        if 'dst volume info' in line:
            result['dst'] = _parse_vol_info(lines, i)
        i += 1
    return result


def _parse_vol_info(lines, start_i):
    info = {}
    i = start_i + 1
    while i < len(lines) and i < start_i + 20:
        line = lines[i].strip()
        for key in ['volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras']:
            if line.startswith(key):
                parts = line.split('=')[1].strip().split()
                if key == 'volume':
                    info[key] = [int(x) for x in parts]
                else:
                    info[key] = [float(x) for x in parts]
                if key == 'cras':
                    return info
        i += 1
    return info


def build_vox2ras(vol_info):
    vol = np.array(vol_info['volume'], dtype=float)
    vs = np.array(vol_info['voxelsize'], dtype=float)
    xr = np.array(vol_info['xras'], dtype=float)
    yr = np.array(vol_info['yras'], dtype=float)
    zr = np.array(vol_info['zras'], dtype=float)
    cras = np.array(vol_info['cras'], dtype=float)

    MdcD = np.column_stack([xr, yr, zr]) @ np.diag(vs)
    P0 = cras - MdcD @ ((vol - 1) / 2.0)
    vox2ras = np.eye(4)
    vox2ras[:3, :3] = MdcD
    vox2ras[:3, 3] = P0
    return vox2ras


def native_voxel_to_sri24_voxel(mri_patient_id, native_coords):
    lta_path = PIPELINE_DIR / mri_patient_id / f'{mri_patient_id}_T1G_reg.lta'
    if not lta_path.exists():
        return None

    lta = parse_lta(str(lta_path))

    orig_t1g = MRI_ORIG_DIR / f'{mri_patient_id}T1G.nii.gz'
    if orig_t1g.exists():
        src_vox2ras = nib.load(str(orig_t1g)).affine
    else:
        src_vox2ras = build_vox2ras(lta['src'])

    dst_vox2ras = build_vox2ras(lta['dst'])
    dst_ras2vox = np.linalg.inv(dst_vox2ras)
    ras2ras = lta['matrix']

    vox_h = np.array([native_coords[0], native_coords[1], native_coords[2], 1.0])
    ras = src_vox2ras @ vox_h
    sri24_ras = ras2ras @ ras
    sri24_vox = dst_ras2vox @ sri24_ras

    return (sri24_vox[0], sri24_vox[1], sri24_vox[2])


def get_biopsy_coordinates(mri_patient_id, biopsy_num, modality='T1G'):
    fname = f'{mri_patient_id}{modality}T{biopsy_num:02d}.csv'
    fpath = ROIVOX_DIR / fname
    if not fpath.exists():
        return None
    with open(fpath) as f:
        parts = f.read().strip().split()
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def create_biopsy_label_volume(mri_patient_id, metadata_df, patient_mapping):
    if mri_patient_id not in patient_mapping:
        return None, []

    meta_patient = patient_mapping[mri_patient_id]
    patient_data = metadata_df[metadata_df['Patient'] == meta_patient]

    ref_file = PIPELINE_DIR / mri_patient_id / f'{mri_patient_id}T1G_SRI24.nii.gz'
    if not ref_file.exists():
        print(f"  No SRI-24 ref for {mri_patient_id}")
        return None, []

    ref_img = nib.load(str(ref_file))
    label_vol = np.zeros(ref_img.shape[:3], dtype=np.float32)

    biopsy_records = []

    for _, row in patient_data.iterrows():
        biopsy_num = row['biopsy_num']
        purity = row.get('PAMES (tumor purity)')

        if pd.isna(purity) or biopsy_num is None or pd.isna(biopsy_num):
            continue
        biopsy_num = int(biopsy_num)

        native_coords = get_biopsy_coordinates(mri_patient_id, biopsy_num)
        if native_coords is None:
            continue

        sri24_coords = native_voxel_to_sri24_voxel(mri_patient_id, native_coords)
        if sri24_coords is None:
            continue

        xi = int(round(sri24_coords[0]))
        yi = int(round(sri24_coords[1]))
        zi = int(round(sri24_coords[2]))

        if (0 <= xi < label_vol.shape[0] and
            0 <= yi < label_vol.shape[1] and
            0 <= zi < label_vol.shape[2]):
            label_vol[xi, yi, zi] = purity

            biopsy_records.append({
                'mri_patient': mri_patient_id,
                'meta_patient': meta_patient,
                'biopsy_id': row['Biopsy'],
                'biopsy_num': biopsy_num,
                'native_x': native_coords[0],
                'native_y': native_coords[1],
                'native_z': native_coords[2],
                'sri24_x': sri24_coords[0],
                'sri24_y': sri24_coords[1],
                'sri24_z': sri24_coords[2],
                'sri24_vox_x': xi,
                'sri24_vox_y': yi,
                'sri24_vox_z': zi,
                'PAMES_purity': purity,
                'TvsN_Predict': row.get('TvsN_Predict'),
                'TvsN_proba_Tumor': row.get('TvsN_proba_Tumor'),
                'Cellularity_mean': row.get('Cellularity_mean'),
                'IDH': row.get('IDH'),
                'Histology': row.get('Histology'),
                'Grade': row.get('Grade'),
                'Cell_Predict': row.get('Cell_Predict'),
            })

    label_img = nib.Nifti1Image(label_vol, ref_img.affine, ref_img.header)
    return label_img, biopsy_records


def create_multichannel_image(mri_patient_id, modalities):
    ref_file = PIPELINE_DIR / mri_patient_id / f'{mri_patient_id}T1G_SRI24.nii.gz'
    if not ref_file.exists():
        return None, []

    ref_img = nib.load(str(ref_file))
    shape = ref_img.shape[:3]
    n_channels = len(modalities)
    stack = np.zeros((*shape, n_channels), dtype=np.float32)
    available = []

    for ch_idx, mod in enumerate(modalities):
        sri24_file = PIPELINE_DIR / mri_patient_id / f'{mri_patient_id}{mod}_SRI24.nii.gz'
        if sri24_file.exists():
            img = nib.load(str(sri24_file))
            data = img.get_fdata(dtype=np.float32)
            # Ensure shape match (should already match... but be safe)
            if data.shape[:3] == shape:
                stack[..., ch_idx] = data
                available.append(mod)
            else:
                print(f"  WARNING: {mod} shape {data.shape} != expected {shape}")
        else:
            # Channel stays zero (missing modality)
            pass

    img_4d = nib.Nifti1Image(stack, ref_img.affine)
    return img_4d, available


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build CNN-ready dataset with selectable modalities and biopsy purity labels.'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=DEFAULT_CNN_DIR,
        help='Output dataset directory (default: Virtual_biopsy/CNN_dataset)'
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=NON_PET_MODALITIES,
        help=(
            'Modalities to stack as channels in output image '
            '(default: all non-PET MRI modalities; PET channels '
            'C18, FET, S24, SR9 can be added explicitly if desired)'
        )
    )
    parser.add_argument(
        '--biopsy-roi-mm',
        type=float,
        default=0.0,
        help=(
            'If > 0, also create an enlarged biopsy purity volume where each '
            'biopsy voxel is expanded to a cubic ROI of this side length in '
            'millimetres. The file will be named '
            "<PID>_biopsy_purity_roi<mm>mm.nii.gz."
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cnn_dir = args.dataset_dir
    selected_modalities = args.modalities
    biopsy_roi_mm = float(getattr(args, 'biopsy_roi_mm', 0.0) or 0.0)

    invalid_modalities = [m for m in selected_modalities if m not in ALL_MODALITIES]
    if invalid_modalities:
        print(f"ERROR: Unknown modalities: {invalid_modalities}")
        print(f"Allowed modalities: {ALL_MODALITIES}")
        sys.exit(1)

    if len(selected_modalities) == 0:
        print('ERROR: At least one modality must be selected.')
        sys.exit(1)

    print("=" * 70)
    print("BUILDING CNN DATASET WITH BIOPSY PURITY LABELS")
    print("=" * 70)
    print(f"Output dir: {cnn_dir}")
    print(f"Selected modalities ({len(selected_modalities)}): {selected_modalities}")

    images_dir = cnn_dir / 'images'
    labels_dir = cnn_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata()
    patient_mapping = build_patient_mapping(metadata_df)
    print(f"Metadata-mapped patients: {len(patient_mapping)} -> {sorted(patient_mapping.keys())}")

    all_biopsy_records = []
    dataset_info = {
        'channel_order': selected_modalities,
        'label_type': 'PAMES_purity_continuous',
        'label_description': 'Single-voxel PAMES tumor purity (0-1) at biopsy locations in SRI-24 space',
        'space': 'SRI-24',
        'voxel_size_mm': [1.0, 1.0, 1.0],
        'volume_shape': list(SRI24_SHAPE),
        'patients': {},
    }

    for pat in ALL_PATIENTS:
        print(f"\n--- {pat} ---")

        has_labels = pat in patient_mapping
        if has_labels:
            label_img, records = create_biopsy_label_volume(pat, metadata_df, patient_mapping)
            if label_img is not None and records:
                label_path = labels_dir / f'{pat}_biopsy_purity.nii.gz'
                nib.save(label_img, str(label_path))
                print(f"  Labels: {len(records)} biopsies saved to {label_path.name}")
                all_biopsy_records.extend(records)

                if biopsy_roi_mm > 0.0:
                    label_data = label_img.get_fdata(dtype=np.float32)
                    vol_shape = label_data.shape
                    voxel_size = label_img.header.get_zooms()[:3]
                    half_vox = [
                        max(1, int(round((biopsy_roi_mm / 2.0) / float(vs))))
                        for vs in voxel_size
                    ]

                    roi_vol = np.zeros(vol_shape, dtype=np.float32)
                    for r in records:
                        xi = int(r['sri24_vox_x'])
                        yi = int(r['sri24_vox_y'])
                        zi = int(r['sri24_vox_z'])
                        purity = float(r['PAMES_purity'])

                        if (
                            xi < 0
                            or yi < 0
                            or zi < 0
                            or xi >= vol_shape[0]
                            or yi >= vol_shape[1]
                            or zi >= vol_shape[2]
                        ):
                            continue

                        x0 = max(0, xi - half_vox[0])
                        x1 = min(vol_shape[0], xi + half_vox[0] + 1)
                        y0 = max(0, yi - half_vox[1])
                        y1 = min(vol_shape[1], yi + half_vox[1] + 1)
                        z0 = max(0, zi - half_vox[2])
                        z1 = min(vol_shape[2], zi + half_vox[2] + 1)

                        roi_vol[x0:x1, y0:y1, z0:z1] = np.maximum(
                            roi_vol[x0:x1, y0:y1, z0:z1], purity
                        )

                    roi_name = f"{pat}_biopsy_purity_roi{int(round(biopsy_roi_mm))}mm.nii.gz"
                    roi_path = labels_dir / roi_name
                    roi_img = nib.Nifti1Image(roi_vol.astype(np.float32), label_img.affine, label_img.header)
                    nib.save(roi_img, str(roi_path))
                    print(f"  ROI labels: enlarged biopsy purity saved to {roi_name}")
            else:
                has_labels = False
                print(f"  Labels: no biopsy data available")
        else:
            print(f"  Labels: no metadata mapping (unlabeled, can use for test)")

        img_4d, available_mods = create_multichannel_image(pat, selected_modalities)
        if img_4d is not None:
            img_path = images_dir / f'{pat}.nii.gz'
            nib.save(img_4d, str(img_path))
            print(f"  Image: {len(available_mods)}/{len(selected_modalities)} channels -> {img_path.name}")
        else:
            available_mods = []
            print(f"  Image: FAILED (no SRI-24 data)")

        dataset_info['patients'][pat] = {
            'n_channels': len(available_mods),
            'available_modalities': available_mods,
            'has_labels': has_labels,
            'n_biopsies': len([r for r in all_biopsy_records if r['mri_patient'] == pat]),
            'meta_patient': patient_mapping.get(pat, 'unknown'),
        }

    biopsy_df = pd.DataFrame(all_biopsy_records)
    biopsy_csv = cnn_dir / 'biopsy_metadata.csv'
    biopsy_df.to_csv(biopsy_csv, index=False)
    print(f"\nBiopsy metadata: {len(biopsy_df)} records -> {biopsy_csv.name}")

    info_path = cnn_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Dataset info: {info_path.name}")

    labeled_count = sum(1 for p in dataset_info['patients'].values() if p['has_labels'])
    total_biopsies = len(biopsy_df)
    print(f"\n{'=' * 70}")
    print(f"CNN DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total patients:   {len(ALL_PATIENTS)}")
    print(f"  With labels:      {labeled_count}")
    print(f"  Without labels:   {len(ALL_PATIENTS) - labeled_count} (usable for test/unlabeled)")
    print(f"  Total biopsies:   {total_biopsies}")
    print(f"  Image shape:      {SRI24_SHAPE} x {len(selected_modalities)} channels")
    print(f"  Label type:       continuous PAMES purity (single voxel)")
    print(f"  Output dir:       {cnn_dir}")
    print(f"\nChannel order:")
    for i, mod in enumerate(selected_modalities):
        print(f"    ch{i:02d}: {mod}")
    print(f"\nPAMES purity stats:")
    print(f"    mean:  {biopsy_df['PAMES_purity'].mean():.3f}")
    print(f"    std:   {biopsy_df['PAMES_purity'].std():.3f}")
    print(f"    range: {biopsy_df['PAMES_purity'].min():.3f} - {biopsy_df['PAMES_purity'].max():.3f}")

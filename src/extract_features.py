"""
Extract 64x64 patches from LUNA16 CT volumes and compute handcrafted texture
features (GLCM + LBP + first-order statistics).

Requires the LUNA16 dataset. Only subset0 is needed to reproduce the reported
result. The script auto-detects the dataset location by searching for
candidates.csv.

Usage:
    python src/extract_features.py --data-root /path/to/luna16 --subset 0

Output:
    results/luna_features.npz  containing X (n, 86), y (n,), uids (n,)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

PATCH_HALF = 32          # -> 64 x 64 patches
NEG_RATIO = 3            # negatives sampled per positive
GLCM_LEVELS = 32         # grey-level quantisation
HU_MIN, HU_MAX = -1000, 400   # lung window


def compute_features(patch, levels=GLCM_LEVELS):
    """86 features: 72 GLCM + 10 LBP histogram bins + 4 first-order stats."""
    # Window Hounsfield Units to the lung range, then scale to 8-bit.
    windowed = np.clip(patch, HU_MIN, HU_MAX)
    windowed = ((windowed - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    quantised = (windowed // (256 // levels)).astype(np.uint8)

    glcm = graycomatrix(
        quantised,
        distances=[1, 2, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels, symmetric=True, normed=True)

    features = []
    for prop in ["contrast", "homogeneity", "energy",
                 "correlation", "dissimilarity", "ASM"]:
        features.extend(graycoprops(glcm, prop).ravel())   # 12 values each

    lbp = local_binary_pattern(windowed, P=8, R=1, method="uniform")
    features.extend(np.histogram(lbp, bins=10, range=(0, 10), density=True)[0])

    features.extend([
        patch.mean(), patch.std(),
        np.percentile(patch, 25), np.percentile(patch, 75),
    ])
    return features


def find_data_root(explicit=None):
    if explicit:
        return explicit
    hits = glob.glob("/kaggle/input/**/candidates.csv", recursive=True)
    if not hits:
        raise FileNotFoundError(
            "Could not locate candidates.csv. Pass --data-root explicitly.")
    return os.path.dirname(hits[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None,
                        help="Directory containing candidates.csv and subsetN/")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = find_data_root(args.data_root)
    print("data root:", root)

    # Map seriesuid -> .mhd path. Mirrors nest subsetN differently, so search.
    pattern = os.path.join(root, f"subset{args.subset}", "**", "*.mhd")
    uid2path = {os.path.basename(p).replace(".mhd", ""): p
                for p in glob.glob(pattern, recursive=True)}
    if not uid2path:
        raise FileNotFoundError(f"No .mhd files found under {pattern}")

    candidates = pd.read_csv(os.path.join(root, "candidates.csv"))
    positive = candidates[(candidates["class"] == 1)
                          & (candidates.seriesuid.isin(uid2path))]
    negative = candidates[(candidates["class"] == 0)
                          & (candidates.seriesuid.isin(uid2path))]
    negative = negative.sample(n=len(positive) * NEG_RATIO, random_state=42)

    work = pd.concat([positive, negative]).sort_values("seriesuid")
    work = work.reset_index(drop=True)
    print(f"scans={len(uid2path)}  positives={len(positive)}  "
          f"negatives={len(negative)}")

    patches, labels, uids = [], [], []
    for uid, group in work.groupby("seriesuid"):
        image = sitk.ReadImage(uid2path[uid])
        volume = sitk.GetArrayFromImage(image)          # (z, y, x)
        origin = np.array(image.GetOrigin())            # (x, y, z) in mm
        spacing = np.array(image.GetSpacing())          # (x, y, z) in mm

        for _, row in group.iterrows():
            world = np.array([row.coordX, row.coordY, row.coordZ])
            # World millimetres -> voxel indices. Note the axis order flip:
            # voxel is (x, y, z) but the array is indexed [z, y, x].
            vx, vy, vz = ((world - origin) / spacing).astype(int)

            in_bounds = (PATCH_HALF <= vx < volume.shape[2] - PATCH_HALF
                         and PATCH_HALF <= vy < volume.shape[1] - PATCH_HALF
                         and 0 <= vz < volume.shape[0])
            if not in_bounds:
                continue

            patch = volume[vz,
                           vy - PATCH_HALF:vy + PATCH_HALF,
                           vx - PATCH_HALF:vx + PATCH_HALF]
            if patch.shape != (2 * PATCH_HALF, 2 * PATCH_HALF):
                continue

            patches.append(patch.astype(np.int16))
            labels.append(int(row["class"]))
            uids.append(uid)

    patches = np.array(patches)
    y = np.array(labels)
    uids = np.array(uids)
    print(f"patches={patches.shape}  patients={len(np.unique(uids))}")

    X = np.array([compute_features(p) for p in patches])
    print("X:", X.shape)

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(
        os.path.dirname(here), "results", "luna_features.npz")
    np.savez_compressed(out, X=X, y=y, uids=uids)
    print("wrote", out)


if __name__ == "__main__":
    main()

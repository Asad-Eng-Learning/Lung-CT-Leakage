"""
Audit a medical imaging dataset for patient-level contamination.

Answers one question: if you split this dataset randomly, what fraction of the
test set would be patients the model already trained on?

Usage:
    python audit_dataset.py --dicom-dir /path/to/dicom_dir
    python audit_dataset.py --csv manifest.csv --id-col PatientID

Output: a contamination report, and a warning if random splitting is unsafe.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

TEST_SIZE = 0.2
N_SEEDS = 20


def index_dicom(dicom_dir):
    """Read headers only. Fast, no pixel data."""
    import pydicom

    files = sorted(glob.glob(os.path.join(dicom_dir, "**", "*.dcm"),
                             recursive=True))
    if not files:
        sys.exit(f"No .dcm files found under {dicom_dir}")

    rows = []
    for f in files:
        d = pydicom.dcmread(f, stop_before_pixels=True)
        rows.append({
            "file": os.path.basename(f),
            "PatientID": getattr(d, "PatientID", None),
            "SeriesInstanceUID": getattr(d, "SeriesInstanceUID", None),
            "SliceLocation": getattr(d, "SliceLocation", None),
            "SliceThickness": getattr(d, "SliceThickness", None),
            "ConvolutionKernel": getattr(d, "ConvolutionKernel", None),
            "Manufacturer": getattr(d, "Manufacturer", None),
        })
    return pd.DataFrame(rows)


def audit(df, id_col="PatientID"):
    n_files = len(df)
    n_patients = df[id_col].nunique()

    print("=" * 62)
    print("PATIENT-LEVEL CONTAMINATION AUDIT")
    print("=" * 62)
    print(f"files                : {n_files}")
    print(f"unique patients      : {n_patients}")
    if "SeriesInstanceUID" in df:
        print(f"unique series        : {df.SeriesInstanceUID.nunique()}")

    if n_patients == n_files:
        print("\nEvery file is a distinct patient. Random splitting is safe.")
        return

    print(f"files per patient    : {n_files / n_patients:.2f}")

    # exact duplicates: same patient, same anatomical location
    if "SliceLocation" in df and df.SliceLocation.notna().any():
        dup = df.groupby([id_col, "SliceLocation"]).size()
        print(f"exact duplicate slices: {(dup > 1).sum()}")

    print("\nMost repeated patients:")
    counts = df[id_col].value_counts()
    print(counts[counts > 1].head(10).to_string())

    # how bad is a random split, empirically
    idx = np.arange(n_files)
    groups = df[id_col].values

    contaminated, leaked_frac = [], []
    for seed in range(N_SEEDS):
        tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed)
        train_patients = set(groups[tr])
        contaminated.append(len(train_patients & set(groups[te])))
        leaked_frac.append(
            sum(1 for i in te if groups[i] in train_patients) / len(te) * 100)

    grp_contaminated = []
    for seed in range(N_SEEDS):
        tr, te = next(GroupShuffleSplit(
            1, test_size=TEST_SIZE, random_state=seed).split(idx, groups=groups))
        grp_contaminated.append(len(set(groups[tr]) & set(groups[te])))

    print(f"\nOver {N_SEEDS} random {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} splits:")
    print(f"  patients on both sides : {np.mean(contaminated):.1f} of {n_patients}")
    print(f"  test images whose patient is also in train : "
          f"{np.mean(leaked_frac):.1f}%")
    print(f"\nWith GroupShuffleSplit on {id_col}:")
    print(f"  patients on both sides : {np.mean(grp_contaminated):.1f}")

    print("\n" + "=" * 62)
    if np.mean(leaked_frac) > 5:
        print("WARNING: random splitting on this dataset is not defensible.")
        print(f"Split on {id_col} using GroupShuffleSplit or StratifiedGroupKFold.")
    print("=" * 62)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dicom-dir", help="directory of .dcm files")
    p.add_argument("--csv", help="manifest CSV instead of DICOM headers")
    p.add_argument("--id-col", default="PatientID")
    p.add_argument("--save", help="write the index to this CSV")
    args = p.parse_args()

    if args.dicom_dir:
        df = index_dicom(args.dicom_dir)
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        sys.exit("Pass either --dicom-dir or --csv")

    if args.save:
        df.to_csv(args.save, index=False)
        print(f"index written to {args.save}\n")

    audit(df, args.id_col)


if __name__ == "__main__":
    main()

"""
Random vs patient-level splitting on LUNA16.

Reproduces the headline result from cached features in results/luna_features.npz.
Runs in under a minute. No CT data download required.

To regenerate the features from raw LUNA16 volumes, use extract_features.py.

Usage:
    python src/run_experiment.py
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = 5
TEST_SIZE = 0.25

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEATURES = os.path.join(ROOT, "results", "luna_features.npz")
OUTPUT = os.path.join(ROOT, "results", "luna_5seed_results.json")


def build_model():
    """Identical model in both arms. Only the split differs."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=0)),
    ])


def evaluate(X, y, train_idx, test_idx):
    model = build_model()
    model.fit(X[train_idx], y[train_idx])
    prob = model.predict_proba(X[test_idx])[:, 1]
    pred = model.predict(X[test_idx])
    return roc_auc_score(y[test_idx], prob), f1_score(y[test_idx], pred)


def main():
    data = np.load(FEATURES, allow_pickle=True)
    X, y, uids = data["X"], data["y"], data["uids"]

    print(f"patches   : {X.shape[0]}")
    print(f"features  : {X.shape[1]}")
    print(f"patients  : {len(np.unique(uids))}")
    print(f"positives : {(y == 1).sum()}   negatives: {(y == 0).sum()}")
    print()

    rand_auc, rand_f1 = [], []
    grp_auc, grp_f1 = [], []
    overlaps = []

    for seed in range(SEEDS):
        # ---- Arm A: random split. Patients may appear on both sides. ----
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(splitter.split(X, y))
        auc, f1 = evaluate(X, y, tr, te)
        rand_auc.append(auc)
        rand_f1.append(f1)
        overlaps.append(len(set(uids[tr]) & set(uids[te])))

        # ---- Arm B: patient-level split. Zero overlap by construction. ----
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(splitter.split(X, y, groups=uids))
        auc, f1 = evaluate(X, y, tr, te)
        grp_auc.append(auc)
        grp_f1.append(f1)

    inflation = np.mean(rand_auc) - np.mean(grp_auc)

    print(f"RANDOM   split  AUC {np.mean(rand_auc):.4f} +/- {np.std(rand_auc):.4f}"
          f"   F1 {np.mean(rand_f1):.4f}")
    print(f"PATIENT  split  AUC {np.mean(grp_auc):.4f} +/- {np.std(grp_auc):.4f}"
          f"   F1 {np.mean(grp_f1):.4f}")
    print()
    print(f"AUC inflation from leakage : {inflation:+.4f}")
    print(f"mean patients on both sides: {np.mean(overlaps):.1f} of {len(np.unique(uids))}")
    print()
    print("random  folds:", np.round(rand_auc, 4))
    print("patient folds:", np.round(grp_auc, 4))

    results = {
        "random_auc_mean": float(np.mean(rand_auc)),
        "random_auc_std": float(np.std(rand_auc)),
        "patient_auc_mean": float(np.mean(grp_auc)),
        "patient_auc_std": float(np.std(grp_auc)),
        "inflation": float(inflation),
        "random_f1_mean": float(np.mean(rand_f1)),
        "patient_f1_mean": float(np.mean(grp_f1)),
        "random_folds": [float(v) for v in rand_auc],
        "patient_folds": [float(v) for v in grp_auc],
        "mean_leaking_patients_random_split": float(np.mean(overlaps)),
        "n_patches": int(len(y)),
        "n_patients": int(len(np.unique(uids))),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_features": int(X.shape[1]),
        "subset": "subset0",
        "seeds": SEEDS,
        "model": "RandomForest(n_estimators=300, class_weight=balanced)",
    }
    with open(OUTPUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()

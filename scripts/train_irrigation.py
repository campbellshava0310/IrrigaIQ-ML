"""Train and select the irrigation-need model.

    .venv/bin/python scripts/train_irrigation.py [--data ../dataset/train.csv]

Compares a linear baseline, a random forest and gradient boosting with 3-fold cross-validation on
a stratified sample, fits the winner on the full training split, and scores it once on a held-out
20% it never saw. Writes models/irrigation.joblib and models/irrigation.card.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from irrigaiq_ml.irrigation.features import CATEGORICAL, CLASSES, FEATURES, NUMERIC, STAGES, TARGET  # noqa: E402
from irrigaiq_ml.irrigation.model import IrrigationNeedModel, reference_values  # noqa: E402

SEED = 7
VERSION = "irrigation-need-1.0"


ALGORITHM_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting (scikit-learn HistGradientBoosting)",
}


def candidates() -> dict[str, Pipeline]:
    one_hot = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )
    ordinal = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                OrdinalEncoder(categories=[STAGES, ["No", "Yes"]], handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL,
            ),
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [("prep", one_hot), ("model", LogisticRegression(max_iter=2000))]
        ),
        "random_forest": Pipeline(
            [
                ("prep", ordinal),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200, min_samples_leaf=5, max_features=0.7, n_jobs=-1, random_state=SEED
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", ordinal),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        categorical_features=[4, 5],
                        max_iter=400,
                        learning_rate=0.08,
                        max_leaf_nodes=63,
                        l2_regularization=1.0,
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def scores(y_true: pd.Series, proba: pd.DataFrame, priors: dict[str, float]) -> dict[str, float]:
    decided = (proba / pd.Series(priors)[CLASSES]).idxmax(axis=1)
    recalls = recall_score(y_true, decided, labels=CLASSES, average=None)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, decided)),
        "macro_f1": float(f1_score(y_true, decided, average="macro")),
        # log_loss reads probability columns in sorted label order.
        "log_loss": float(log_loss(y_true, proba[sorted(CLASSES)].to_numpy(), labels=sorted(CLASSES))),
        **{f"recall_{c.lower()}": float(r) for c, r in zip(CLASSES, recalls)},
        "calibration_error": expected_calibration_error(y_true, proba),
    }


def expected_calibration_error(y_true: pd.Series, proba: pd.DataFrame, bins: int = 10) -> float:
    """Gap between stated confidence and accuracy of the top class (lower is better)."""
    confidence = proba.max(axis=1).to_numpy()
    correct = (proba.idxmax(axis=1).to_numpy() == y_true.to_numpy()).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def proba_frame(pipeline: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(pipeline.predict_proba(x), columns=list(pipeline.classes_), index=x.index)[CLASSES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT.parent / "dataset" / "train.csv"))
    parser.add_argument("--cv-sample", type=int, default=150_000)
    args = parser.parse_args()

    data_path = Path(args.data)
    frame = pd.read_csv(data_path)
    x, y = frame[FEATURES], frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    priors = y_train.value_counts(normalize=True).to_dict()

    # 1. Model selection by cross-validation on a stratified sample of the training split.
    sample_x, _, sample_y, _ = train_test_split(
        x_train, y_train, train_size=args.cv_sample, stratify=y_train, random_state=SEED
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    comparison: dict[str, dict[str, float]] = {}
    for name, pipeline in candidates().items():
        fold_scores = []
        started = time.time()
        for train_idx, valid_idx in folds.split(sample_x, sample_y):
            pipeline.fit(sample_x.iloc[train_idx], sample_y.iloc[train_idx])
            proba = proba_frame(pipeline, sample_x.iloc[valid_idx])
            fold_scores.append(scores(sample_y.iloc[valid_idx], proba, priors))
        mean = {k: float(np.mean([s[k] for s in fold_scores])) for k in fold_scores[0]}
        mean["seconds"] = round(time.time() - started, 1)
        comparison[name] = mean
        print(f"{name:20s} balanced acc {mean['balanced_accuracy']:.4f}  macro F1 {mean['macro_f1']:.4f}  "
              f"log loss {mean['log_loss']:.4f}  ECE {mean['calibration_error']:.4f}  ({mean['seconds']}s)")

    # Highest balanced accuracy; log loss breaks near-ties (within 0.002) in favour of better probabilities.
    best_ba = max(m["balanced_accuracy"] for m in comparison.values())
    contenders = {n: m for n, m in comparison.items() if best_ba - m["balanced_accuracy"] <= 0.002}
    winner = min(contenders, key=lambda n: contenders[n]["log_loss"])
    print(f"selected: {winner}")

    # 2. Fit the winner on the full training split; score once on the untouched test split.
    pipeline = candidates()[winner]
    pipeline.fit(x_train, y_train)
    test = scores(y_test, proba_frame(pipeline, x_test), priors)
    print("held-out test:", json.dumps({k: round(v, 4) for k, v in test.items()}))

    model = IrrigationNeedModel(
        pipeline=pipeline,
        priors=priors,
        reference=reference_values(x_train),
        ranges={n: (float(x_train[n].min()), float(x_train[n].max())) for n in NUMERIC},
        version=VERSION,
        metrics=test,
    )
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    model.save(out / "irrigation.joblib")

    card = {
        "name": "Irrigation need",
        "version": VERSION,
        "task": "Classify a field's irrigation need as Low, Medium or High from today's conditions.",
        "algorithm": f"{ALGORITHM_NAMES[winner]}, chosen over the other candidates by cross-validated balanced accuracy",
        "selectedCandidate": winner,
        "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "file": data_path.name,
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "rows": int(len(frame)),
            "trainRows": int(len(x_train)),
            "testRows": int(len(x_test)),
            "classBalance": {k: round(v, 4) for k, v in priors.items()},
        },
        "features": FEATURES,
        "featureSelection": (
            "Six of twenty candidate columns; the rest changed held-out balanced accuracy by less than "
            "0.001 under permutation, so they are left out rather than filled with guesses."
        ),
        "decisionRule": "Class probability divided by training frequency, largest wins (optimises balanced accuracy).",
        "modelComparison": comparison,
        "heldOutTest": test,
        "limitations": [
            "Trained on a public synthetic benchmark dataset, not on measurements from IrrigaIQ farms.",
            "Seasons and regions in the data are Indian; season and region carry no signal and are not used.",
            "Rainfall in the data is annual-scale and, below about 300 mm, marks dry conditions; IrrigaIQ "
            "passes the last 90 days of rain scaled to a year, so a dry spell reads as dry even where the "
            "yearly total is high.",
            "The dataset does not define its soil-moisture scale. IrrigaIQ treats it as a probe-style "
            "percentage with about 15 % at wilting point and 40 % at field capacity, which places the "
            "model's learnt dry threshold (about 25 %) at 60 % depletion, the stress threshold of typical "
            "field crops (FAO-56, p = 0.5-0.6).",
            "Volumes, run times and timing still come from the FAO-56 water balance; this model answers "
            "whether irrigation is needed and how confident it is.",
            "Retrain on sensor-labelled data from IrrigaIQ farms before relying on it for decisions.",
        ],
    }
    (out / "irrigation.card.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {out / 'irrigation.joblib'} ({(out / 'irrigation.joblib').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Train the pump failure-risk model on a simulated fleet and compare it with rules.

    .venv/bin/python scripts/train_equipment.py

Pumps are split into training and test groups (no pump appears in both), so the score reflects
pumps the model has never seen. Writes models/equipment.joblib and models/equipment.card.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from irrigaiq_ml.equipment.model import EquipmentRiskModel  # noqa: E402
from irrigaiq_ml.equipment.simulator import FEATURES, HORIZON_DAYS, simulate_fleet  # noqa: E402

VERSION = "equipment-risk-1.0"

# Physics as constraints: worse readings can never lower the predicted risk. This keeps the model
# sensible for pumps worse than any it saw, and keeps its explanations pointing the right way.
MONOTONIC = {
    "energy_ratio": 1,
    "energy_trend": 1,
    "flow_ratio": -1,
    "pressure_ratio": -1,
    "vibration_mm_s": 1,
    "motor_temp_c": 1,
    "runtime_h_per_day": 1,
    "starts_per_day": 1,
    "service_overdue_ratio": 1,
    "age_ratio": 1,
}
SEED = 11


def rules_flag(frame) -> np.ndarray:
    """What a sensible rule set would flag: IrrigaIQ's current health-factor thresholds."""
    return (
        (frame["energy_ratio"] >= 1.2)
        | (frame["vibration_mm_s"].fillna(0) >= 7.1)
        | (frame["service_overdue_ratio"] >= 1.0)
        | (frame["flow_ratio"] <= 0.85)
    ).to_numpy()


def precision_at_recall(y_true: np.ndarray, proba: np.ndarray, recall: float) -> dict[str, float]:
    """The model's precision when its threshold is set to catch as many failures as the rules do."""
    order = np.argsort(-proba)
    caught = np.cumsum(y_true[order])
    needed = recall * y_true.sum()
    k = int(np.searchsorted(caught, needed)) + 1
    return {"recall": float(caught[k - 1] / y_true.sum()), "precision": float(caught[k - 1] / k), "threshold": float(proba[order][k - 1])}


def oracle(test, y_test) -> dict[str, float]:
    """Best achievable: scoring with each pump's true hidden wear (failures stay random events)."""
    hazard = 1 / (1 + np.exp(-(-8.2 + 7.5 * test["wear"] + 0.6 * np.maximum(0, test["service_overdue_ratio"] - 1))))
    score = 1 - (1 - hazard) ** HORIZON_DAYS
    return {"rocAuc": float(roc_auc_score(y_test, score)), "averagePrecision": float(average_precision_score(y_test, score))}


def main() -> None:
    fleet = simulate_fleet(pumps=700, seed=SEED)
    x, y, groups = fleet[FEATURES], fleet["fails_within_horizon"].astype(int), fleet["pump"]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(x, y, groups))
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
    x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

    classifier = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        monotonic_cst=[MONOTONIC[name] for name in FEATURES],
        early_stopping=True,
        random_state=SEED,
    ).fit(x_train, y_train)
    proba = classifier.predict_proba(x_test)[:, 1]

    # Operating point used for "HIGH" and above (probability >= 0.15).
    flagged = proba >= 0.15
    rules = rules_flag(fleet.iloc[test_idx])
    metrics = {
        "rocAuc": float(roc_auc_score(y_test, proba)),
        "averagePrecision": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "baseRate": float(y_test.mean()),
        "atHighThreshold": {
            "threshold": 0.15,
            "recall": float(recall_score(y_test, flagged)),
            "precision": float(precision_score(y_test, flagged)),
        },
        "rulesBaseline": {
            "recall": float(recall_score(y_test, rules)),
            "precision": float(precision_score(y_test, rules)),
        },
        "modelAtRulesRecall": precision_at_recall(y_test.to_numpy(), proba, float(recall_score(y_test, rules))),
        "ceiling": oracle(fleet.iloc[test_idx], y_test),
        "trainPumps": int(groups.iloc[train_idx].nunique()),
        "testPumps": int(groups.iloc[test_idx].nunique()),
        "rows": int(len(fleet)),
    }
    print(json.dumps(metrics, indent=2))

    healthy = x_train[fleet.iloc[train_idx]["wear"] < 0.2]
    reference = {name: float(healthy[name].median()) for name in FEATURES}
    model = EquipmentRiskModel(classifier=classifier, reference=reference, version=VERSION, metrics=metrics)
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    model.save(out / "equipment.joblib")

    card = {
        "name": "Equipment failure risk",
        "version": VERSION,
        "task": f"Probability that a pump fails within {HORIZON_DAYS} days, with the readings driving it.",
        "algorithm": (
            "Gradient boosting (scikit-learn HistGradientBoosting) with monotonic constraints from pump "
            "physics; native handling of missing sensors"
        ),
        "monotonicConstraints": MONOTONIC,
        "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES,
        "data": "Simulated fleet: 700 pumps x 540 days of wear, servicing and failures (irrigaiq_ml/equipment/simulator.py).",
        "evaluation": metrics,
        "riskBands": {"LOW": "< 5%", "MODERATE": "5-15%", "HIGH": "15-35%", "CRITICAL": ">= 35%"},
        "interpretation": (
            "Failures are random events even for a worn pump, so no model can foresee most of them. "
            "The fair yardstick is the ceiling (scoring with each pump's true hidden wear): the model "
            f"reaches {metrics['rocAuc'] / metrics['ceiling']['rocAuc']:.0%} of its ROC AUC. At the rules' "
            f"catch rate it is {metrics['modelAtRulesRecall']['precision'] / metrics['rulesBaseline']['precision'] - 1:.0%} "
            "more precise, and adds a calibrated 14-day probability, a smooth ranking and tolerance of "
            "missing sensors."
        ),
        "limitations": [
            "Trained on simulated degradation, not on failures of real IrrigaIQ pumps.",
            "Physically plausible but idealised: real pumps fail in ways the simulation does not model.",
            "Complements, not replaces, the explainable health score; retrain on real telemetry and "
            "maintenance records once a season of data exists.",
        ],
    }
    (out / "equipment.card.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {out / 'equipment.joblib'} ({(out / 'equipment.joblib').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

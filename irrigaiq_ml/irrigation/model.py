"""Irrigation-need model at run time: probabilities, a decision, and the reasons behind it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import CATEGORICAL, CLASSES, FEATURES, LABELS, NUMERIC, UNITS, FieldConditions


@dataclass
class IrrigationNeedModel:
    """A fitted classifier plus what is needed to use it honestly.

    Probabilities come from a model trained without class weights, so they stay calibrated for
    display ("87% likely"). The decision divides each probability by the class's training
    frequency before taking the largest (the Bayes rule for balanced accuracy), so the rare "High"
    class is not drowned out by "Low".
    """

    pipeline: Any
    priors: dict[str, float]
    reference: dict[str, Any]
    ranges: dict[str, tuple[float, float]]
    version: str
    metrics: dict[str, Any] = field(default_factory=dict)

    # ---- inference -------------------------------------------------------------------------

    def probabilities(self, rows: pd.DataFrame) -> pd.DataFrame:
        proba = self.pipeline.predict_proba(rows[FEATURES])
        classes = list(self.pipeline.classes_)
        return pd.DataFrame(proba, columns=classes, index=rows.index)[CLASSES]

    def decide(self, proba: pd.DataFrame) -> pd.Series:
        adjusted = proba / pd.Series(self.priors)[CLASSES]
        return adjusted.idxmax(axis=1)

    def predict(self, conditions: FieldConditions) -> dict[str, Any]:
        row = pd.DataFrame([conditions.as_row()])
        proba = self.probabilities(row).iloc[0]
        need = self.decide(proba.to_frame().T).iloc[0]
        required = need != "Low"
        # Confidence in the yes/no answer, not in the exact class.
        p_required = float(proba["Medium"] + proba["High"])
        confidence = p_required if required else 1.0 - p_required
        return {
            "need": need,
            "irrigationRequired": bool(required),
            "probabilities": {k: round(float(v), 4) for k, v in proba.items()},
            "probabilityRequired": round(p_required, 4),
            "confidence": round(float(confidence), 4),
            "needIndex": round(self._need_index(row), 4),
            "drivers": self.explain(row),
            "outOfRange": self.out_of_range(row.iloc[0]),
            "modelVersion": self.version,
        }

    # ---- explanation -----------------------------------------------------------------------

    def explain(self, row: pd.DataFrame, top: int = 4) -> list[dict[str, Any]]:
        """How much each input moves the field's irrigation need.

        For each feature, swap in its typical training value and measure the change in the need
        index (Low 0, Medium 0.5, High 1, weighted by probability). Positive impact means this
        field's value pushes towards irrigating. The index is used rather than P(irrigate) because
        that probability saturates near 1, hiding what separates Medium from High. Model-agnostic
        and cheap (six extra predictions), and it reads as a sentence.
        """
        base = self._need_index(row)
        drivers = []
        for name in FEATURES:
            swapped = row.copy()
            swapped[name] = self.reference[name]
            impact = base - self._need_index(swapped)
            value = row.iloc[0][name]
            drivers.append(
                {
                    "feature": name,
                    "label": LABELS[name],
                    "value": _display(name, value),
                    "typical": _display(name, self.reference[name]),
                    "impact": round(float(impact), 4),
                    "direction": "raises" if impact > 0 else "lowers",
                }
            )
        drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return [d for d in drivers[:top] if abs(d["impact"]) >= 0.01]

    def _need_index(self, rows: pd.DataFrame) -> float:
        proba = self.probabilities(rows).iloc[0]
        return float(0.5 * proba["Medium"] + proba["High"])

    def out_of_range(self, row: pd.Series) -> list[str]:
        """Numeric inputs outside what the model saw in training: its answer is an extrapolation."""
        flags = []
        for name in NUMERIC:
            low, high = self.ranges[name]
            if not low <= float(row[name]) <= high:
                flags.append(f"{LABELS[name]} {_display(name, row[name])} is outside the training range")
        return flags

    # ---- persistence -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "IrrigationNeedModel":
        model = joblib.load(path)
        if not isinstance(model, IrrigationNeedModel):
            raise TypeError(f"{path} does not hold an IrrigationNeedModel")
        return model


def reference_values(frame: pd.DataFrame) -> dict[str, Any]:
    """Typical values: medians for numbers, most common category otherwise."""
    ref: dict[str, Any] = {}
    for name in NUMERIC:
        ref[name] = float(frame[name].median())
    for name in CATEGORICAL:
        ref[name] = str(frame[name].mode().iloc[0])
    return ref


def _display(name: str, value: Any) -> str:
    if name in NUMERIC:
        digits = 0 if name == "Rainfall_mm" else 1
        return f"{float(value):.{digits}f}{UNITS[name]}"
    return str(value)


def as_probability_frame(values: np.ndarray, classes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, columns=classes)[CLASSES]

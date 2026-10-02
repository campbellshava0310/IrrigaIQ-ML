"""Equipment failure-risk model at run time: probability, risk level, and the indicators behind it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .simulator import FEATURES, HORIZON_DAYS

LABELS = {
    "energy_ratio": "Energy per m³",
    "energy_trend": "Energy per m³, 14-day change",
    "flow_ratio": "Flow",
    "pressure_ratio": "Pressure",
    "vibration_mm_s": "Vibration",
    "motor_temp_c": "Motor temperature",
    "runtime_h_per_day": "Daily runtime",
    "starts_per_day": "Starts per day",
    "service_overdue_ratio": "Service interval used",
    "age_ratio": "Share of design life used",
}

# Risk bands on the 14-day failure probability.
LEVELS = [(0.35, "CRITICAL"), (0.15, "HIGH"), (0.05, "MODERATE"), (0.0, "LOW")]

ACTIONS = {
    "CRITICAL": "Stop and inspect the pump now; plan a standby.",
    "HIGH": "Inspect the pump within 7 days.",
    "MODERATE": "Check the pump at the next visit and watch the trend.",
    "LOW": "No action beyond the service plan.",
}


@dataclass
class EquipmentRiskModel:
    classifier: Any
    reference: dict[str, float]
    version: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict(self, readings: dict[str, float | None]) -> dict[str, Any]:
        row = pd.DataFrame([{name: _value(readings.get(name)) for name in FEATURES}])
        probability = self._probability(row)
        level = next(name for floor, name in LEVELS if probability >= floor)
        return {
            "failureProbability": round(probability, 4),
            "horizonDays": HORIZON_DAYS,
            "riskLevel": level,
            "healthScore": int(round(100 * (1 - min(1.0, probability / 0.5)))),
            "recommendation": ACTIONS[level],
            "indicators": self.indicators(row),
            "missing": [LABELS[n] for n in FEATURES if pd.isna(row.iloc[0][n])],
            "modelVersion": self.version,
        }

    def indicators(self, row: pd.DataFrame, top: int = 4) -> list[dict[str, Any]]:
        """Which readings push the risk up: swap each for a healthy pump's value and compare."""
        base = self._probability(row)
        out = []
        for name in FEATURES:
            value = row.iloc[0][name]
            if pd.isna(value):
                continue
            swapped = row.copy()
            swapped[name] = self.reference[name]
            impact = base - self._probability(swapped)
            out.append(
                {
                    "feature": name,
                    "label": LABELS[name],
                    "value": _display(name, value),
                    "healthy": _display(name, self.reference[name]),
                    "impact": round(float(impact), 4),
                    "trend": _trend(name, float(value), self.reference[name]),
                }
            )
        out.sort(key=lambda d: d["impact"], reverse=True)
        return [d for d in out[:top] if d["impact"] >= 0.005]

    def _probability(self, rows: pd.DataFrame) -> float:
        return float(self.classifier.predict_proba(rows[FEATURES])[0, 1])

    def save(self, path: Path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "EquipmentRiskModel":
        model = joblib.load(path)
        if not isinstance(model, EquipmentRiskModel):
            raise TypeError(f"{path} does not hold an EquipmentRiskModel")
        return model


def _value(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _trend(name: str, value: float, healthy: float) -> str:
    return "up" if value > healthy else "down"


def _display(name: str, value: float) -> str:
    if name == "energy_trend":
        return f"{value * 100:+.0f} pts"
    if name.endswith("_ratio"):
        return f"{value * 100:.0f}%"
    if name == "vibration_mm_s":
        return f"{value:.1f} mm/s"
    if name == "motor_temp_c":
        return f"{value:.0f} °C"
    if name == "runtime_h_per_day":
        return f"{value:.1f} h"
    return f"{value:.1f}"

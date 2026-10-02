"""Water anomaly detection for a flow meter's hourly readings.

Two complementary views, both fitted on the meter's own history:

1. A robust baseline: for the same hour of day and the same irrigation state (scheduled or not),
   the median and median absolute deviation of past flow. A robust z-score says how far the
   latest hours sit from that pattern, in units the median cannot be dragged by.
2. An Isolation Forest over (flow, hour of day, scheduled, change from the previous hour), which
   catches combinations the per-hour baseline does not model.

An hour is anomalous when its robust z-score is large, and the Isolation Forest agrees it is
unusual or the deviation is extreme. Sustained deviations count more than single hours. The
output speaks in possibilities, never certainties: a meter cannot tell a leak from a valve left
open, so both are offered as possible causes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

VERSION = "water-anomaly-1.1"
MIN_HISTORY_HOURS = 72
Z_ALERT = 4.0
Z_EXTREME = 8.0
# A smaller deviation that persists in the same direction is still a signal (slow leaks).
Z_SUSTAINED = 2.8
SUSTAINED_HOURS = 3


@dataclass(frozen=True)
class Reading:
    at: datetime
    litres: float
    scheduled: bool


def _features(readings: list[Reading]) -> np.ndarray:
    rows = []
    previous = readings[0].litres if readings else 0.0
    for r in readings:
        angle = 2 * math.pi * r.at.hour / 24
        rows.append([r.litres, math.sin(angle), math.cos(angle), float(r.scheduled), r.litres - previous])
        previous = r.litres
    return np.asarray(rows, dtype=float)


class FlowBaseline:
    """Robust expected flow from a meter's history.

    During irrigation the rate does not depend on the hour, so run-hours are compared with all
    past runs. Between runs, background use can follow the clock (troughs, a house), so those hours
    use the same hour's history when there is enough of it.
    """

    def __init__(self, history: list[Reading]):
        by_state: dict[bool, list[float]] = {True: [], False: []}
        by_slot: dict[tuple[int, bool], list[float]] = {}
        for r in history:
            by_state[r.scheduled].append(r.litres)
            by_slot.setdefault((r.at.hour, r.scheduled), []).append(r.litres)
        self._state = {k: _robust(v) for k, v in by_state.items() if v}
        self._slot = {k: _robust(v) for k, v in by_slot.items() if len(v) >= 5}
        # Spread floors: a meter that reads 0 +- 2 L at night must not flag a 12 L reading as
        # 6 sigma. Between runs the floor is 0.5 % of the run rate, near a meter's starting flow
        # (Q1 is 0.6-1 % of nominal for R100-R160 meters): anything steadier and larger is real water.
        running = self._state.get(True, (0.0, 0.0))[0]
        self._floor = {True: max(15.0, 0.03 * running), False: max(15.0, 0.005 * running)}

    def expected(self, r: Reading) -> tuple[float, float]:
        state = self._state.get(r.scheduled, (0.0, 0.0))
        median, spread = state if r.scheduled else self._slot.get((r.at.hour, False), state)
        return median, max(spread, self._floor[r.scheduled])


def _robust(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median))) * 1.4826
    return median, mad


def detect(history: list[Reading], recent: list[Reading], *, seed: int = 0) -> dict[str, Any]:
    """Score `recent` hours against the meter's `history` (which should not include them)."""
    if len(history) < MIN_HISTORY_HOURS:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "message": f"Needs at least {MIN_HISTORY_HOURS} hours of history; has {len(history)}.",
            "points": [],
            "anomaly": None,
            "modelVersion": VERSION,
        }
    baseline = FlowBaseline(history)
    forest = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(_features(history))
    history_scores = -forest.score_samples(_features(history))
    recent_scores = -forest.score_samples(_features(recent))

    points: list[dict[str, Any]] = []
    for r, raw in zip(recent, recent_scores):
        expected, spread = baseline.expected(r)
        z = (r.litres - expected) / spread
        isolation = float((history_scores < raw).mean())  # share of history that looks more normal
        flagged = abs(z) >= Z_EXTREME or (abs(z) >= Z_ALERT and isolation >= 0.97)
        points.append(
            {
                "at": r.at.isoformat(),
                "observedLitres": round(r.litres, 1),
                "expectedLitres": round(expected, 1),
                "zScore": round(float(z), 2),
                "isolation": round(isolation, 3),
                "score": round(_score(z, isolation), 3),
                "anomalous": bool(flagged),
            }
        )
    _mark_sustained(points)
    return {
        "status": "OK",
        "points": points,
        "anomaly": _summarise(recent, points),
        "modelVersion": VERSION,
    }


def _mark_sustained(points: list[dict[str, Any]]) -> None:
    """Flag runs of SUSTAINED_HOURS or more consecutive hours deviating the same way."""
    run: list[int] = []
    for i, p in enumerate(points + [None]):  # sentinel closes the last run
        sign = 0 if p is None else (1 if p["zScore"] >= Z_SUSTAINED else -1 if p["zScore"] <= -Z_SUSTAINED else 0)
        if sign != 0 and (not run or sign == (1 if points[run[-1]]["zScore"] > 0 else -1)):
            run.append(i)
            continue
        if len(run) >= SUSTAINED_HOURS:
            for j in run:
                points[j]["anomalous"] = True
                points[j]["score"] = round(max(points[j]["score"], 0.65), 3)
        run = [i] if sign != 0 else []


def _score(z: float, isolation: float) -> float:
    """0-1: how unusual an hour is. Grows with the z-score, lifted when the forest agrees."""
    deviation = 1 - math.exp(-max(0.0, abs(z) - 2) / 4)
    return min(1.0, 0.75 * deviation + 0.25 * isolation * (1 if abs(z) >= 2 else 0))


def _summarise(recent: list[Reading], points: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The latest run of consecutive anomalous hours, if the most recent hours are in one."""
    run: list[int] = []
    for i in range(len(points) - 1, -1, -1):
        if points[i]["anomalous"]:
            run.append(i)
        elif run:
            break
        elif len(points) - i > 2:  # allow the newest hour or two to be normal again
            break
    if not run:
        return None
    run.reverse()
    hours = [points[i] for i in run]
    readings = [recent[i] for i in run]
    observed = float(np.mean([p["observedLitres"] for p in hours]))
    expected = float(np.mean([p["expectedLitres"] for p in hours]))
    score = float(np.mean([p["score"] for p in hours]))
    direction = "HIGH" if observed > expected else "LOW"
    scheduled = any(r.scheduled for r in readings)
    severity = "HIGH" if (score >= 0.8 and len(run) >= 2) or score >= 0.95 else "MEDIUM" if score >= 0.6 else "LOW"
    return {
        "from": hours[0]["at"],
        "to": hours[-1]["at"],
        "hours": len(run),
        "direction": direction,
        "observedLitresPerHour": round(observed, 1),
        "expectedLitresPerHour": round(expected, 1),
        "score": round(score, 3),
        "severity": severity,
        "duringScheduledIrrigation": scheduled,
        "headline": _headline(direction),
        "possibleCauses": _causes(direction, scheduled),
    }


def _headline(direction: str) -> str:
    if direction == "HIGH":
        return "Abnormal water consumption detected. Possible leakage or irrigation-system issue."
    return "Flow well below normal. Possible blockage or supply problem."


def _causes(direction: str, scheduled: bool) -> list[str]:
    if direction == "HIGH" and not scheduled:
        return [
            "Irrigation left running or a valve not closing",
            "Possible pipe leak or burst",
            "Unscheduled manual irrigation",
        ]
    if direction == "HIGH":
        return [
            "Possible leak or burst downstream of the meter",
            "Damaged emitters, nozzles or fittings",
            "More water applied than planned",
        ]
    if scheduled:
        return [
            "Clogged filter or emitters",
            "Pump or supply problem (low water level, power, worn pump)",
            "Valve not fully opening",
        ]
    return ["Meter reading lower than usual; check the meter and supply"]

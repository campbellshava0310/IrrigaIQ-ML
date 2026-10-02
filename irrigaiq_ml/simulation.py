"""Realistic synthetic farm signals, for evaluating detectors and for the live demo.

Hourly flow at a field's meter follows its irrigation schedule: near zero between runs, a steady
rate (with noise) while a run is on. Faults are injected on top so a detector can be scored
against known answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

FAULTS = ("leak", "slow_leak", "burst", "left_running", "blockage")


@dataclass(frozen=True)
class FlowPoint:
    at: datetime
    litres: float
    scheduled: bool
    fault: str | None = None


def schedule_for(day: int, rng: np.random.Generator) -> set[int]:
    """Hours with irrigation on: an early-morning run every other day, sometimes an evening one."""
    hours: set[int] = set()
    if day % 2 == 0:
        start = int(rng.integers(4, 7))
        hours.update(range(start, start + int(rng.integers(2, 4))))
    if rng.random() < 0.25:
        hours.update(range(18, 20))
    return hours


def normal_flow(
    start: datetime,
    days: int,
    *,
    rate_lph: float = 1800.0,
    seed: int = 0,
) -> list[FlowPoint]:
    """Healthy meter: runs at the design rate +-6%, a few litres of meter noise otherwise."""
    rng = np.random.default_rng(seed)
    points: list[FlowPoint] = []
    for day in range(days):
        on_hours = schedule_for(day, rng)
        for hour in range(24):
            scheduled = hour in on_hours
            litres = rng.normal(rate_lph, rate_lph * 0.06) if scheduled else abs(rng.normal(0, 4))
            points.append(FlowPoint(start + timedelta(days=day, hours=hour), max(0.0, float(litres)), scheduled))
    return points


def inject(points: list[FlowPoint], fault: str, at: int, hours: int, *, rate_lph: float = 1800.0, seed: int = 1) -> list[FlowPoint]:
    """Overlay a fault on `hours` points starting at index `at`."""
    rng = np.random.default_rng(seed)
    out = list(points)
    for i in range(at, min(at + hours, len(out))):
        p = out[i]
        if fault == "leak":  # a steady loss on top of whatever is flowing
            litres = p.litres + rate_lph * rng.uniform(0.18, 0.3)
        elif fault == "slow_leak":  # a small loss: a few percent of the run rate
            litres = p.litres + rate_lph * rng.uniform(0.03, 0.06)
        elif fault == "burst":  # a large uncontrolled flow
            litres = p.litres + rate_lph * rng.uniform(1.2, 1.8)
        elif fault == "left_running":  # a run that never stopped
            litres = rng.normal(rate_lph, rate_lph * 0.06) if not p.scheduled else p.litres
        elif fault == "blockage":  # scheduled runs deliver much less
            litres = p.litres * rng.uniform(0.25, 0.5) if p.scheduled else p.litres
        else:
            raise ValueError(f"Unknown fault {fault}")
        out[i] = FlowPoint(p.at, max(0.0, float(litres)), p.scheduled, fault)
    return out

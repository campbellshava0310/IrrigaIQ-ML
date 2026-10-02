"""A fleet of irrigation pumps wearing out, being serviced and failing, one day at a time.

Wear w (0 = new, 1 = worn out) grows with running hours and starts, faster when a service is
overdue; a service removes most of it, a failure is followed by a repair. Each day the pump shows
the symptoms of its wear, blurred by sensor noise, and fails with a hazard that rises steeply with
wear. This mirrors how centrifugal pumps actually degrade (impeller and wear-ring erosion, bearing
wear): more energy per cubic metre, less flow and head, rising vibration and temperature.

Rows are daily feature snapshots labelled with whether the pump fails within the next 14 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_DAYS = 14

# Sensor features are 7-day means, as IrrigaIQ computes them from telemetry and checks.
FEATURES = [
    "energy_ratio",        # kWh per m3 / baseline
    "energy_trend",        # change in energy_ratio over the last 14 days
    "flow_ratio",          # flow / rated flow
    "pressure_ratio",      # delivery pressure / rated
    "vibration_mm_s",      # RMS velocity (ISO 10816 zones: >4.5 unsatisfactory, >7.1 unacceptable)
    "motor_temp_c",
    "runtime_h_per_day",
    "starts_per_day",
    "service_overdue_ratio",  # days since service / service interval
    "age_ratio",              # age / design life
]


def simulate_fleet(pumps: int = 400, days: int = 540, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for pump in range(pumps):
        frames.append(_simulate_pump(pump, days, rng))
    return pd.concat(frames, ignore_index=True)


def _simulate_pump(pump: int, days: int, rng: np.random.Generator) -> pd.DataFrame:
    wear = rng.uniform(0.0, 0.35)
    interval = int(rng.choice([90, 120, 180]))
    since_service = int(rng.integers(0, interval))
    age_ratio = rng.uniform(0.05, 0.9)
    diligence = rng.uniform(0.5, 1.0)  # how reliably services happen on time
    duty = rng.uniform(0.6, 1.4)  # how hard this pump works
    wear_rate = rng.uniform(0.0016, 0.0032)
    rows = []
    failures = []
    for day in range(days):
        runtime = float(np.clip(rng.normal(7.5 * duty, 1.8), 0.5, 20))
        starts = float(np.clip(rng.normal(2.5 * duty, 1.0), 0, 12))
        overdue = since_service / interval
        wear += wear_rate * (runtime / 8) * (1 + 0.1 * starts) * (1 + max(0.0, overdue - 1) * 1.5)
        wear = min(wear, 1.2)

        hazard = 1 / (1 + np.exp(-(-8.2 + 7.5 * wear + 0.6 * max(0.0, overdue - 1))))
        failed = rng.random() < hazard
        failures.append(failed)

        rows.append(
            {
                "pump": pump,
                "day": day,
                "energy_ratio": 1 + 0.42 * wear + rng.normal(0, 0.03),
                "flow_ratio": 1 - 0.3 * wear + rng.normal(0, 0.03),
                "pressure_ratio": 1 - 0.22 * wear + rng.normal(0, 0.03),
                "vibration_mm_s": max(0.3, 1.4 + 7.5 * wear**2 + rng.normal(0, 0.35)),
                "motor_temp_c": 42 + 26 * wear + rng.normal(0, 3.0) + 0.4 * (runtime - 8),
                "runtime_h_per_day": runtime,
                "starts_per_day": starts,
                "service_overdue_ratio": overdue,
                "age_ratio": age_ratio + day / 3650,
                "wear": wear,
            }
        )

        if failed:  # repaired: new wear rings, bearings; counts as a service
            wear = rng.uniform(0.05, 0.15)
            since_service = 0
        elif since_service >= interval and rng.random() < diligence * 0.35:
            wear *= 0.35
            since_service = 0
        else:
            since_service += 1

    frame = pd.DataFrame(rows)
    smoothed = ["energy_ratio", "flow_ratio", "pressure_ratio", "vibration_mm_s", "motor_temp_c"]
    frame[smoothed] = frame[smoothed].rolling(7, min_periods=1).mean()
    frame["energy_trend"] = frame["energy_ratio"] - frame["energy_ratio"].shift(14).fillna(frame["energy_ratio"].iloc[0])
    fails = np.asarray(failures)
    # Label: a failure on any of the next HORIZON_DAYS days (not today's, which is already known).
    ahead = np.zeros(len(fails), dtype=bool)
    for offset in range(1, HORIZON_DAYS + 1):
        ahead[:-offset] |= fails[offset:]
    frame["fails_within_horizon"] = ahead
    frame["failed_today"] = fails
    # Real deployments rarely have every sensor: hide some readings so the model copes with gaps.
    for column, share in (("vibration_mm_s", 0.35), ("motor_temp_c", 0.5), ("pressure_ratio", 0.25), ("starts_per_day", 0.2)):
        frame.loc[rng.random(len(frame)) < share, column] = np.nan
    return frame[~frame["failed_today"]]  # a failed pump is repaired, not predicted

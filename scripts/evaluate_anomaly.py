"""Score the water anomaly detector on simulated meters with known, injected faults.

    .venv/bin/python scripts/evaluate_anomaly.py

For each trial: 21 days of healthy history, then a 48-hour window that is either clean or carries
one fault (leak, burst, irrigation left running, blockage). Reports event-level detection rate per
fault, the false-alarm rate on clean windows, and hour-level precision. Writes
models/water_anomaly.card.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from irrigaiq_ml.anomaly.detector import VERSION, Reading, detect  # noqa: E402
from irrigaiq_ml.simulation import FAULTS, inject, normal_flow  # noqa: E402

HISTORY_DAYS = 21
WINDOW_HOURS = 48
TRIALS_PER_CASE = 40


def main() -> None:
    start = datetime(2026, 8, 1)
    detected: Counter[str] = Counter()
    flagged_hours = true_hours = 0
    for case in ("clean", *FAULTS):
        for trial in range(TRIALS_PER_CASE):
            seed = hash((case, trial)) % 10_000
            rate = 900 + (trial * 97) % 2400
            series = normal_flow(start, HISTORY_DAYS + WINDOW_HOURS // 24, rate_lph=rate, seed=seed)
            cut = HISTORY_DAYS * 24
            if case != "clean":
                # Faults start somewhere in the window and last 3-12 hours; blockage needs a run to show.
                length = 3 + trial % 10
                begin = cut + (trial * 7) % (WINDOW_HOURS - length)
                if case == "blockage":
                    runs = [i for i in range(cut, len(series)) if series[i].scheduled]
                    begin = runs[0] if runs else begin
                series = inject(series, case, begin, length, rate_lph=rate, seed=seed + 1)
            history = [Reading(p.at, p.litres, p.scheduled) for p in series[:cut]]
            window = series[cut:]
            result = detect(history, [Reading(p.at, p.litres, p.scheduled) for p in window], seed=seed)
            hits = [pt["anomalous"] for pt in result["points"]]
            if any(hits):
                detected[case] += 1
            for point, hit in zip(window, hits):
                if hit:
                    flagged_hours += 1
                    if point.fault is not None:
                        true_hours += 1

    report = {
        "detectionRate": {f: round(detected[f] / TRIALS_PER_CASE, 3) for f in FAULTS},
        "falseAlarmRate": round(detected["clean"] / TRIALS_PER_CASE, 3),
        "hourPrecision": round(true_hours / flagged_hours, 3) if flagged_hours else None,
        "trialsPerCase": TRIALS_PER_CASE,
        "windowHours": WINDOW_HOURS,
        "historyDays": HISTORY_DAYS,
    }
    print(json.dumps(report, indent=2))

    card = {
        "name": "Water anomaly",
        "version": VERSION,
        "task": "Flag hours of abnormal water flow at a meter, with severity and possible causes.",
        "algorithm": "Robust per-hour baseline (median/MAD by hour and irrigation state) + Isolation Forest",
        "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fittedOn": "Each meter's own history at request time (unsupervised); no shared training data.",
        "evaluation": report,
        "evaluationData": "Simulated meters (IrrigaIQ schedules, 900-3300 L/h design rates) with injected faults.",
        "faults": {
            "leak": "18-30 % of the run rate added to every hour",
            "slow_leak": "3-6 % of the run rate added to every hour",
            "burst": "120-180 % of the run rate added",
            "left_running": "a run continuing through unscheduled hours",
            "blockage": "scheduled runs delivering 25-50 % of normal flow",
        },
        "limitations": [
            "Needs at least 72 hours of a meter's history before it can judge.",
            "A meter cannot tell a leak from a valve left open: causes are offered as possibilities.",
            "Evaluated on simulated faults; confirm thresholds against real meters once installed.",
        ],
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "water_anomaly.card.json").write_text(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()

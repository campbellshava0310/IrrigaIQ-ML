"""Simulated field devices for demos and testing: a soil probe, a water meter and a pump monitor.

They talk to the IrrigaIQ backend exactly as ESP32 devices do (POST /api/v1/telemetry with their
own X-Device-Key), and label every reading SIMULATED so synthetic data never passes as real.
Standard library only.

    # 1. Register the three devices on a farm (asks for your IrrigaIQ password; keys are saved
    #    to tools/.devices.json, which is git-ignored)
    python tools/simulate_devices.py setup --email you@example.com --farm "Mazowe Valley"

    # 2. Give the meter and pump monitor history (the anomaly check needs 3+ days)
    python tools/simulate_devices.py backfill --days 14

    # 3. Demo: the probe reads 23 % -> the field needs water
    python tools/simulate_devices.py soil 23

    # 4. Demo: 500 L/h flowing with no irrigation scheduled -> possible leak
    python tools/simulate_devices.py leak --lph 500 --hours 2

    # Back to normal (no unscheduled flow)
    python tools/simulate_devices.py normal
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import random
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEVICES_FILE = Path(__file__).resolve().parent / ".devices.json"
BATCH = 500


# ---- HTTP -------------------------------------------------------------------------------------------


def request(api: str, method: str, path: str, body=None, token: str | None = None, device_key: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(api.rstrip("/") + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if device_key:
        req.add_header("X-Device-Key", device_key)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        sys.exit(f"{method} {path} failed: HTTP {error.code} {detail}")
    except urllib.error.URLError as error:
        sys.exit(f"Cannot reach {api}: {error.reason}. Is the backend running?")


def post_readings(config: dict, device: str, readings: list[dict]) -> tuple[int, int]:
    key = config["devices"][device]["key"]
    accepted = duplicates = 0
    for start in range(0, len(readings), BATCH):
        receipt = request(
            config["api"], "POST", "/api/v1/telemetry",
            {"source": "SIMULATED", "readings": readings[start : start + BATCH]}, device_key=key,
        )
        accepted += receipt["accepted"]
        duplicates += receipt["duplicates"]
    return accepted, duplicates


def iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load() -> dict:
    if not DEVICES_FILE.exists():
        sys.exit(f"No devices yet: run `setup` first ({DEVICES_FILE.name} is missing).")
    return json.loads(DEVICES_FILE.read_text())


# ---- setup -------------------------------------------------------------------------------------------


def pick(items: list[dict], query: str, key: str, kind: str) -> dict:
    exact = [i for i in items if i[key].lower() == query.lower() or i["id"] == query]
    matches = exact or [i for i in items if query.lower() in i[key].lower()]
    if len(matches) != 1:
        names = ", ".join(i[key] for i in items)
        sys.exit(f"{kind} '{query}' matched {len(matches)}; choose one of: {names}")
    return matches[0]


def setup(args) -> None:
    password = os.environ.get("IRRIGAIQ_PASSWORD") or getpass.getpass(f"IrrigaIQ password for {args.email}: ")
    token = request(args.api, "POST", "/api/v1/auth/login", {"email": args.email, "password": password})["accessToken"]
    farms = request(args.api, "GET", "/api/v1/farms?size=100", token=token)["content"]
    farm = pick(farms, args.farm, "name", "Farm") if args.farm else farms[0]
    fid = farm["id"]
    fields = request(args.api, "GET", f"/api/v1/farms/{fid}/fields", token=token)
    assets = request(args.api, "GET", f"/api/v1/farms/{fid}/assets", token=token)
    field = pick(fields, args.field, "name", "Field")
    pumps = [a for a in assets if a["type"] == "PUMP"]
    meter_pump = next((a for a in pumps if a["id"] == field.get("pumpAssetId")), None) or pumps[0]
    monitored = pick(pumps, args.pump, "name", "Pump")

    suffix = datetime.now().strftime("%H%M%S")
    plan = {
        "probe": {"type": "SOIL_MOISTURE", "deviceId": f"sim-probe-{suffix}", "name": f"Soil probe · {field['name']}",
                  "fieldId": field["id"], "fieldCapacityPct": 40, "wiltingPointPct": 15},
        "meter": {"type": "FLOW_METER", "deviceId": f"sim-meter-{suffix}", "name": f"Water meter · {meter_pump['name']}",
                  "assetId": meter_pump["id"]},
        "monitor": {"type": "PUMP_MONITOR", "deviceId": f"sim-pump-{suffix}", "name": f"Pump monitor · {monitored['name']}",
                    "assetId": monitored["id"]},
    }
    devices = {}
    for role, body in plan.items():
        issued = request(args.api, "POST", f"/api/v1/farms/{fid}/devices", body, token=token)
        devices[role] = {"id": issued["device"]["id"], "name": issued["device"]["name"], "key": issued["deviceKey"]}
        print(f"registered {issued['device']['name']} (key {issued['device']['keyPrefix']}…)")

    # The meter and monitor imitate the real equipment, so remember its ratings and schedule.
    config = {
        "api": args.api, "farmId": fid, "farm": farm["name"],
        "field": {"id": field["id"], "name": field["name"]},
        "meterPump": {k: meter_pump[k] for k in ("id", "name", "ratedFlowM3h")},
        "monitoredPump": {k: monitored[k] for k in ("id", "name", "ratedFlowM3h", "ratedPowerKw", "ratedHeadM",
                                                    "baselineEnergyKwhPerM3", "operatingWindowStart",
                                                    "operatingWindowEnd")},
        "devices": devices,
        "events": request(args.api, "GET", f"/api/v1/farms/{fid}/irrigation/events?"
                          + urllib.parse.urlencode({"from": (datetime.now() - timedelta(days=60)).date().isoformat(),
                                                    "to": (datetime.now() + timedelta(days=1)).date().isoformat()}),
                          token=token),
    }
    DEVICES_FILE.write_text(json.dumps(config, indent=2))
    DEVICES_FILE.chmod(0o600)
    print(f"saved device keys to {DEVICES_FILE} (keep it private)")


# ---- flow model ------------------------------------------------------------------------------------------


def pump_runs(config: dict) -> list[tuple[datetime, datetime]]:
    """Intervals the metered pump ran, from the farm's irrigation log."""
    pump = config["meterPump"]["id"]
    runs = []
    for event in config["events"]:
        if event.get("pumpAssetId") == pump and event["status"] in ("COMPLETED", "PLANNED"):
            start = datetime.fromisoformat(event["startedAt"].replace("Z", "+00:00"))
            runs.append((start, start + timedelta(minutes=event["durationMinutes"])))
    return runs


def normal_flow_lpm(config: dict, at: datetime, runs, rng: random.Random) -> float:
    """What the meter reads: the pump's rated flow while it runs, a trickle of noise otherwise."""
    if any(start <= at < end for start, end in runs):
        rated = (config["meterPump"].get("ratedFlowM3h") or 12) * 1000 / 60
        return round(rated * rng.uniform(0.93, 0.98), 2)
    return round(max(0.0, rng.gauss(0.0, 0.05)), 2)


# ---- commands ----------------------------------------------------------------------------------------------


def backfill(args) -> None:
    config = load()
    rng = random.Random(7)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(days=args.days)
    runs = pump_runs(config)

    flow = []
    at = start
    while at < now - timedelta(minutes=5):
        flow.append({"recordedAt": iso(at), "flowLpm": normal_flow_lpm(config, at, runs, rng)})
        at += timedelta(minutes=15)
    accepted, dup = post_readings(config, "meter", flow)
    print(f"meter: {accepted} readings stored ({dup} already there) over {args.days} days, {len(runs)} logged pump runs")

    # Pump monitor: a pump wearing out over three weeks (energy per m3, vibration and heat rising).
    pump = config["monitoredPump"]
    rated_flow = pump.get("ratedFlowM3h") or 60
    baseline = pump.get("baselineEnergyKwhPerM3") or (pump.get("ratedPowerKw") or 15) / rated_flow
    head = pump.get("ratedHeadM") or 40
    readings = []
    days = max(args.days, 21)
    for d in range(days):
        wear = d / days
        day = (now - timedelta(days=days - d)).replace(hour=22, minute=0)
        for step in range(0, 6 * 60, 30):  # six hours a night, every 30 minutes
            at = day + timedelta(minutes=step)
            flow_ratio = 0.96 - 0.12 * wear + rng.gauss(0, 0.01)
            flow_m3h = rated_flow * flow_ratio
            energy_ratio = 1.08 + 0.24 * wear + rng.gauss(0, 0.015)
            readings.append({
                "recordedAt": iso(at),
                "flowLpm": round(flow_m3h * 1000 / 60, 1),
                "powerKw": round(flow_m3h * baseline * energy_ratio, 2),
                "pressureBar": round(head / 10.2 * (0.97 - 0.1 * wear) + rng.gauss(0, 0.03), 2),
                "vibrationMmS": round(2.2 + 4.8 * wear**1.5 + rng.gauss(0, 0.2), 2),
                "motorTempC": round(52 + 20 * wear + rng.gauss(0, 1.2), 1),
            })
    accepted, dup = post_readings(config, "monitor", readings)
    print(f"pump monitor: {accepted} readings stored ({dup} already there): {pump['name']} wearing over {days} days")


def soil(args) -> None:
    config = load()
    accepted, _ = post_readings(config, "probe", [{"soilMoisturePct": args.value, "temperatureC": args.temperature}])
    print(f"{config['devices']['probe']['name']}: {args.value} % sent ({'stored' if accepted else 'duplicate'})")


def leak(args) -> None:
    """Unscheduled flow at a steady rate for the last N hours (readings every 5 minutes)."""
    config = load()
    rng = random.Random()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    readings = []
    minutes = int(args.hours * 60)
    for back in range(minutes, -1, -5):
        at = now - timedelta(minutes=back)
        readings.append({"recordedAt": iso(at), "flowLpm": round(args.lph / 60 * rng.uniform(0.97, 1.03), 2)})
    accepted, _ = post_readings(config, "meter", readings)
    print(f"{config['devices']['meter']['name']}: {args.lph:.0f} L/h for the last {args.hours:g} h ({accepted} readings)")


def normal(args) -> None:
    config = load()
    rng = random.Random()
    runs = pump_runs(config)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    readings = [{"recordedAt": iso(now - timedelta(minutes=m)),
                 "flowLpm": normal_flow_lpm(config, now - timedelta(minutes=m), runs, rng)} for m in range(0, 60, 5)]
    accepted, _ = post_readings(config, "meter", readings)
    print(f"{config['devices']['meter']['name']}: normal flow for the last hour ({accepted} readings)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("setup", help="register a soil probe, a water meter and a pump monitor")
    p.add_argument("--api", default=os.environ.get("IRRIGAIQ_API", "http://localhost:8080"))
    p.add_argument("--email", required=True)
    p.add_argument("--farm", help="farm name or id (default: your first farm)")
    p.add_argument("--field", default="Block A", help="field for the soil probe; the meter goes on its pump")
    p.add_argument("--pump", default="P-2", help="pump to fit the monitor to")
    p.set_defaults(run=setup)

    p = commands.add_parser("backfill", help="history for the meter and pump monitor")
    p.add_argument("--days", type=int, default=14)
    p.set_defaults(run=backfill)

    p = commands.add_parser("soil", help="send one soil-moisture reading")
    p.add_argument("value", type=float, help="probe reading, percent")
    p.add_argument("--temperature", type=float, default=24.0)
    p.set_defaults(run=soil)

    p = commands.add_parser("leak", help="unscheduled flow on the meter")
    p.add_argument("--lph", type=float, default=500)
    p.add_argument("--hours", type=float, default=2)
    p.set_defaults(run=leak)

    p = commands.add_parser("normal", help="normal flow on the meter for the last hour")
    p.set_defaults(run=normal)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

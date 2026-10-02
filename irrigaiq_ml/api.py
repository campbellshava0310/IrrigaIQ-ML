"""HTTP API for the IrrigaIQ backend. Not exposed to browsers: the backend is the only client.

    .venv/bin/uvicorn irrigaiq_ml.api:app --port 8090

Set IRRIGAIQ_ML_TOKEN to require the same value in the X-Service-Token header.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from . import __version__, embeddings
from .anomaly.detector import Reading, detect
from .equipment.model import EquipmentRiskModel
from .equipment.simulator import FEATURES as EQUIPMENT_FEATURES
from .irrigation.features import PHASE_TO_STAGE, FieldConditions
from .irrigation.model import IrrigationNeedModel

log = logging.getLogger("irrigaiq_ml")
MODELS_DIR = Path(os.environ.get("IRRIGAIQ_MODELS_DIR", Path(__file__).resolve().parents[1] / "models"))


class Schema(BaseModel):
    """camelCase on the wire (the backend is Java), snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---- irrigation --------------------------------------------------------------------------------


class FieldInput(Schema):
    field_id: str
    soil_moisture_pct: float = Field(ge=0, le=100)
    temperature_c: float = Field(ge=-20, le=60)
    wind_speed_kmh: float = Field(ge=0, le=200)
    rainfall_mm_annualised: float = Field(ge=0, le=10_000)
    growth_phase: str
    mulched: bool = False


class IrrigationRequest(Schema):
    fields: list[FieldInput] = Field(min_length=1, max_length=500)


# ---- water anomaly -----------------------------------------------------------------------------


class FlowReading(Schema):
    at: datetime
    litres: float = Field(ge=0)
    scheduled: bool = False


class WaterAnomalyRequest(Schema):
    meter_id: str
    history: list[FlowReading] = Field(max_length=24 * 120)
    recent: list[FlowReading] = Field(min_length=1, max_length=24 * 7)


# ---- equipment -----------------------------------------------------------------------------------


class AssetReadings(Schema):
    energy_ratio: float | None = None
    energy_trend: float | None = None
    flow_ratio: float | None = None
    pressure_ratio: float | None = None
    vibration_mm_s: float | None = None
    motor_temp_c: float | None = None
    runtime_h_per_day: float | None = None
    starts_per_day: float | None = None
    service_overdue_ratio: float | None = None
    age_ratio: float | None = None


class AssetInput(Schema):
    asset_id: str
    readings: AssetReadings


class EquipmentRequest(Schema):
    assets: list[AssetInput] = Field(min_length=1, max_length=500)


# ---- embeddings ----------------------------------------------------------------------------------


class EmbeddingRequest(Schema):
    texts: list[str] = Field(min_length=1, max_length=256)
    query: bool = False


# ---- app -------------------------------------------------------------------------------------------

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    for name, loader in (("irrigation", IrrigationNeedModel.load), ("equipment", EquipmentRiskModel.load)):
        path = MODELS_DIR / f"{name}.joblib"
        try:
            state[name] = loader(path)
            log.info("loaded %s model %s", name, state[name].version)
        except FileNotFoundError:
            log.warning("no %s model at %s; run scripts/train_%s.py", name, path, name)
    yield
    state.clear()


app = FastAPI(title="IrrigaIQ ML", version=__version__, lifespan=lifespan)


def require_token(x_service_token: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("IRRIGAIQ_ML_TOKEN")
    if expected and not (x_service_token and secrets.compare_digest(x_service_token, expected)):
        raise HTTPException(status_code=401, detail="Missing or wrong service token")


def model(name: str) -> Any:
    if name not in state:
        raise HTTPException(status_code=503, detail=f"The {name} model is not loaded")
    return state[name]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "UP", "models": sorted(state), "version": __version__}


@app.get("/v1/models", dependencies=[Depends(require_token)])
def models() -> list[dict[str, Any]]:
    """Model cards: what each model does, how it was evaluated, and its limits."""
    cards = []
    for path in sorted(MODELS_DIR.glob("*.card.json")):
        cards.append(json.loads(path.read_text()))
    cards.append(
        {
            "name": "Knowledge embeddings",
            "version": embeddings.MODEL_NAME,
            "task": "Turn agricultural knowledge passages and questions into vectors for retrieval.",
            "algorithm": f"{embeddings.MODEL_NAME} ({embeddings.DIMENSIONS} dimensions, ONNX, runs locally)",
        }
    )
    return cards


@app.post("/v1/irrigation/predict", dependencies=[Depends(require_token)])
def predict_irrigation(request: IrrigationRequest) -> dict[str, Any]:
    irrigation = model("irrigation")
    results = []
    for field in request.fields:
        if field.growth_phase.upper() not in PHASE_TO_STAGE:
            raise HTTPException(status_code=422, detail=f"Unknown growth phase {field.growth_phase}")
        prediction = irrigation.predict(
            FieldConditions(
                soil_moisture_pct=field.soil_moisture_pct,
                temperature_c=field.temperature_c,
                wind_speed_kmh=field.wind_speed_kmh,
                rainfall_mm_annualised=field.rainfall_mm_annualised,
                growth_phase=field.growth_phase,
                mulched=field.mulched,
            )
        )
        results.append({"fieldId": field.field_id, **prediction})
    return {"predictions": results}


@app.post("/v1/anomaly/water", dependencies=[Depends(require_token)])
def water_anomaly(request: WaterAnomalyRequest) -> dict[str, Any]:
    def readings(points: list[FlowReading]) -> list[Reading]:
        return [Reading(p.at, p.litres, p.scheduled) for p in sorted(points, key=lambda p: p.at)]

    result = detect(readings(request.history), readings(request.recent))
    return {"meterId": request.meter_id, **result}


@app.post("/v1/equipment/risk", dependencies=[Depends(require_token)])
def equipment_risk(request: EquipmentRequest) -> dict[str, Any]:
    equipment = model("equipment")
    results = []
    for asset in request.assets:
        readings = asset.readings.model_dump()
        prediction = equipment.predict({name: readings.get(name) for name in EQUIPMENT_FEATURES})
        results.append({"assetId": asset.asset_id, **prediction})
    return {"assets": results}


@app.post("/v1/embeddings", dependencies=[Depends(require_token)])
def embed(request: EmbeddingRequest) -> dict[str, Any]:
    return {
        "model": embeddings.MODEL_NAME,
        "dimensions": embeddings.DIMENSIONS,
        "vectors": embeddings.embed(request.texts, query=request.query),
    }

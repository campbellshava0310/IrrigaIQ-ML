"""The irrigation-need model's inputs, and how an IrrigaIQ field maps onto them.

The training data (dataset/train.csv, 630k rows, 20 candidate columns) is synthetic. Permutation
importance on a held-out set shows only six columns carry signal; the other fourteen (soil pH,
organic carbon, conductivity, crop type, region, season, field area, ...) move balanced accuracy by
less than 0.001. The model is therefore trained on these six only, so IrrigaIQ never has to invent
values it does not measure.
"""

from __future__ import annotations

from dataclasses import dataclass

NUMERIC = ["Soil_Moisture", "Temperature_C", "Wind_Speed_kmh", "Rainfall_mm"]
CATEGORICAL = ["Crop_Growth_Stage", "Mulching_Used"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

STAGES = ["Sowing", "Vegetative", "Flowering", "Harvest"]

# IrrigaIQ's FAO-56 growth phases onto the dataset's crop stages.
PHASE_TO_STAGE = {
    "INITIAL": "Sowing",
    "DEVELOPMENT": "Vegetative",
    "MID_SEASON": "Flowering",
    "LATE_SEASON": "Harvest",
    "HARVEST": "Harvest",
}

# Plain-language names for explanations.
LABELS = {
    "Soil_Moisture": "Soil moisture",
    "Temperature_C": "Temperature",
    "Wind_Speed_kmh": "Wind",
    "Rainfall_mm": "Rainfall (last 90 days, annualised)",
    "Crop_Growth_Stage": "Growth stage",
    "Mulching_Used": "Mulch",
}

UNITS = {"Soil_Moisture": "%", "Temperature_C": "°C", "Wind_Speed_kmh": " km/h", "Rainfall_mm": " mm"}


@dataclass(frozen=True)
class FieldConditions:
    """What IrrigaIQ knows about a field today, in the model's units."""

    soil_moisture_pct: float
    temperature_c: float
    wind_speed_kmh: float
    rainfall_mm_annualised: float
    growth_phase: str
    mulched: bool

    def as_row(self) -> dict[str, object]:
        stage = PHASE_TO_STAGE.get(self.growth_phase.upper())
        if stage is None:
            raise ValueError(f"Unknown growth phase: {self.growth_phase}")
        return {
            "Soil_Moisture": float(self.soil_moisture_pct),
            "Temperature_C": float(self.temperature_c),
            "Wind_Speed_kmh": float(self.wind_speed_kmh),
            "Rainfall_mm": float(self.rainfall_mm_annualised),
            "Crop_Growth_Stage": stage,
            "Mulching_Used": "Yes" if self.mulched else "No",
        }

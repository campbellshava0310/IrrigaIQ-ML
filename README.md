# IrrigaIQ ML

The machine-learning service behind IrrigaIQ. It is a small, stateless HTTP API that the IrrigaIQ
backend calls; browsers never reach it.

| Model | Answers | Method |
| --- | --- | --- |
| **Irrigation need** | Does this field need water today (Low / Medium / High), how sure, and why? | Gradient boosting, chosen over random forest and logistic regression by cross-validation |
| **Water anomaly** | Is this meter's flow abnormal right now, how badly, and what might cause it? | Robust per-hour baselines (median/MAD, by hour and irrigation state) plus an Isolation Forest, fitted to each meter's own history |
| **Equipment failure risk** | How likely is this pump to fail in the next 14 days, and which readings say so? | Gradient boosting with monotonic constraints from pump physics; tolerates missing sensors |
| **Knowledge embeddings** | Vectors for the assistant's knowledge search | BAAI/bge-small-en-v1.5 (384 dimensions, ONNX via fastembed, runs locally) |

**Division of labour.** The models predict. The backend owns the data, the FAO-56 water balance
and the business rules. The language-model assistant explains and orchestrates, and never
replaces a model. Volumes, run times and timing always come from the water balance; the
irrigation model is a second opinion with its drivers.

## Results

Each model writes a model card (`models/*.card.json`, also served at `GET /v1/models`) with its
data, evaluation and limitations.

**Irrigation need**, on 126,000 held-out rows. Only six of twenty candidate columns carried
signal; the rest changed held-out accuracy by under 0.001 and are left out rather than guessed.

| Metric | Value |
| --- | --- |
| Balanced accuracy | 0.969 (random forest 0.965, logistic regression 0.846) |
| Recall: Low / Medium / High | 0.995 / 0.959 / 0.954 |
| Log loss, calibration error | 0.060, 0.001 |

**Water anomaly**, over 40 simulated meters per case, each with 21 days of history.

| Injected fault | Detected |
| --- | --- |
| Burst (120–180 % of the run rate) | 100 % |
| Leak (18–30 %) | 100 % |
| Slow leak (3–6 %) | 90 % |
| Irrigation left running | 100 % |
| Blockage (runs at 25–50 % of normal flow) | 100 % |
| False alarms on clean meters | 0 % |

**Equipment failure risk**, on 175 pumps the model never saw during training (group split).

| Metric | Value |
| --- | --- |
| ROC AUC | 0.782, against 0.788 for a perfect-knowledge ceiling (99 %) |
| At the rule set's catch rate | 23 % more precise than the rules |
| Brier score | 0.046 (base rate 5.2 %) |

Failures are random events even for a worn pump, so no model can foresee most of them. The fair
yardstick is the ceiling: scoring with each pump's true hidden wear.

### Limitations (read before relying on it)

- The irrigation model was trained on a public **synthetic** benchmark dataset with Indian
  seasons, not on measurements from IrrigaIQ farms. Its soil-moisture scale is undefined in the
  data; IrrigaIQ maps it to probe readings with about 15 % at wilting and 40 % at field capacity.
- The anomaly and equipment models were evaluated on **simulated** meters and pumps.
- Retrain all three on real IrrigaIQ telemetry and maintenance records once a season of data
  exists. Until then they are decision support, and the backend surfaces any disagreement with the
  physically based rules.

## Run

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/uvicorn irrigaiq_ml.api:app --port 8090    # http://localhost:8090/docs
.venv/bin/python -m pytest -q
```

The embedding model (about 64 MB) downloads on first use into `.cache/fastembed`
(`IRRIGAIQ_EMBEDDINGS_CACHE`). The Docker image downloads it at build time, so the container needs
no internet at run time:

```bash
docker build -t irrigaiq/ml .
docker run -p 127.0.0.1:8090:8090 -e IRRIGAIQ_ML_TOKEN=... irrigaiq/ml
```

Set `IRRIGAIQ_ML_TOKEN` here and in the backend to require the same `X-Service-Token` on every
call. `IRRIGAIQ_MODELS_DIR` points at a different model directory.

## API

| Endpoint | Body → result |
| --- | --- |
| `GET /health` | Loaded models and version |
| `GET /v1/models` | Model cards |
| `POST /v1/irrigation/predict` | `fields[]` of soil moisture, temperature, wind, annualised rainfall, growth phase, mulch → need, probabilities, confidence, drivers, inputs out of training range |
| `POST /v1/anomaly/water` | A meter's hourly `history[]` and `recent[]` (`at`, `litres`, `scheduled`) → per-hour scores and, when abnormal, an anomaly with severity, observed against expected and possible causes |
| `POST /v1/equipment/risk` | `assets[]` of pump readings (all optional) → 14-day failure probability, risk level, recommendation, indicators, missing inputs |
| `POST /v1/embeddings` | `texts[]`, `query` → unit vectors |

All bodies are camelCase JSON; invalid input is a 422.

## Retrain

```bash
.venv/bin/python scripts/train_irrigation.py --data ../dataset/train.csv
.venv/bin/python scripts/train_equipment.py
PYTHONHASHSEED=0 .venv/bin/python scripts/evaluate_anomaly.py
```

Each script prints its metrics and rewrites the model file and card. Bump `VERSION` in a script,
or in `irrigaiq_ml/anomaly/detector.py`, whenever behaviour changes. The version travels with
every prediction into the backend and its alerts.

## Device simulator

`tools/simulate_devices.py` imitates the ESP32 devices for demos and tests. It registers a soil
probe, a water meter and a pump monitor on a farm, backfills history, and injects a dry reading
or unscheduled flow. It needs only the Python standard library. Its readings are labelled
`SIMULATED`, and device keys are kept in the git-ignored `tools/.devices.json`. See
`IrrigaIQ-Backend/docs/summit-demo.md` for the demo it drives.

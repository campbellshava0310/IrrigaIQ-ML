# IrrigaIQ ML: working notes

Python 3.14, scikit-learn 1.9, pandas 3, FastAPI, pydantic 2, fastembed. Run everything through
`.venv/bin/…`.

- The service is stateless and called only by the backend. Wire format is camelCase
  (`Schema` in `irrigaiq_ml/api.py` sets the alias generator); Python stays snake_case.
- Models load once at startup from `models/` (`IRRIGAIQ_MODELS_DIR`). A missing model returns 503
  for its endpoint, never a guess.
- Every model has a card (`models/*.card.json`) written by its script in `scripts/`. When
  behaviour changes, bump the version constant and regenerate the card: versions travel into
  backend alerts.
- Wording is part of the contract: anomaly headlines and causes are hedged ("possible leak or
  irrigation-system issue"), never diagnoses.
- pandas 3 strings are not `object` dtype: use `pd.api.types.is_numeric_dtype`.
- LightGBM is not used (it needs libomp); use scikit-learn's HistGradientBoosting.
- `scripts/evaluate_anomaly.py` needs `PYTHONHASHSEED=0` to be reproducible.

Verify with `.venv/bin/python -m pytest -q`.

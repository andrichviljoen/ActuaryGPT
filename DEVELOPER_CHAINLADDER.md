# ActuaryGPT Chainladder Developer Notes

## Architecture

- `app.py`: Streamlit orchestration/UI workflow.
- `reserving_app/services/chainladder_demo.py`: sample dataset loading (`cl.load_sample`) + fallback snapshot + stochastic example (`BootstrapODPSample` + `Development`).
- `reserving_app/services/reserving_models.py`: chainladder model execution and conversion between app tables and `cl.Triangle`.
- `reserving_app/services/triangle_builder.py`: input normalization and lag-based transactional triangle construction.
- `reserving_app/services/json_utils.py`: robust JSON serialization for pandas/numpy/time-like objects.
- `reserving_app/services/ai_assistant.py`: AI assistant/report context packaging + OpenAI request calls.

## Supported chainladder functionality in the app

- Sample dataset loading via `cl.load_sample(...)`.
- Triangle conversion to/from app displays.
- Deterministic methods:
  - `Chainladder`
  - `MackChainladder`
  - `BornhuetterFerguson`
  - `Benktander`
  - `CapeCod`
  - `Development`
- Stochastic:
  - `BootstrapODPSample`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Reconciliation testing

- Core tests run with:
  - `pytest -q`
- Chainladder-dependent tests use `pytest.importorskip("chainladder")` so they run when chainladder runtime is available.
- Demo flow reconciliation includes a direct check that sample loading and fallback behaviors produce expected structures.

## Intentionally not fully exposed yet

- Some advanced tail/adjustment classes and full pipeline combinators are not yet fully surfaced in separate UI pages.
- Reason: current implementation prioritizes stable, high-usage methods and robust fallback behavior across environments where chainladder optional runtime dependencies may not always be present.

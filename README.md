# ActuaryGPT Reserving Studio

A production-style Streamlit web application for actuarial reserving analysis with professional UI/UX, flexible field mapping, deterministic and bootstrap chain ladder methods, diagnostics, AI commentary, and report generation.

## Features

- Upload CSV/XLSX claims data (drag-and-drop via Streamlit uploader).
- Excel sheet detection and selection.
- Data quality cleaning (blank/duplicate rows, date parsing, numeric conversion, missing value checks).
- Field mapping UI with suggested mapping.
- Flexible triangle basis: paid, incurred, reported counts, paid counts.
- User-selectable period grain: yearly, half-yearly, quarterly, monthly.
- Triangle build (incremental + cumulative) with segment filtering.
- Reserving methods:
  - Deterministic Chain Ladder
  - Bootstrap Chain Ladder (via `chainladder` where available + robust fallback)
- Assumption panel (tail factor, bootstrap sims, excluded link-ratio cells).
- Diagnostics:
  - triangle and link-ratio heatmaps
  - outlier detection on link ratios
  - latest diagonal view
  - sparse data warnings
- Results charts:
  - selected factors
  - reserve by origin
  - latest vs ultimate
  - bootstrap histogram
  - uncertainty percentile chart
- AI assistant integration (OpenAI API) with prebuilt prompts and strict non-fabrication instruction.
- Downloadables:
  - triangle CSV
  - all result tables to multi-tab Excel workbook
  - PDF reserving report with management summary + technical appendix
- One-click demo mode (`data/demo_claims.csv`)
- Actuarial notes and audit trail in session state.

## Architecture

```
app.py
reserving_app/
  core/
    config.py
    logging_config.py
  services/
    data_ingestion.py
    mapping_validation.py
    triangle_builder.py
    reserving_models.py
    diagnostics.py
    charts.py
    ai_assistant.py
    reporting.py
data/
  demo_claims.csv
  sample_template.csv
requirements.txt
.env.example
```

## Setup (Mac)

Recommended Python: **3.10.x** (validated with Python 3.10.19).  
`chainladder==0.8.25` is pinned for compatibility with this stack.

1. **Create and activate virtual environment**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Set `OPENAI_API_KEY` in `.env` or your shell environment.

4. **Run app**
   ```bash
   streamlit run app.py
   ```

## Usage workflow

1. **Upload Data**: upload CSV/XLSX, choose Excel sheet if needed, ingest and preview.
2. **Map Fields**: review suggested mapping and adjust manually.
3. **Build Triangle**: choose segment filter and generate cumulative/incremental triangles.
4. **Methods**: set assumptions, exclude selected link ratio cells, run deterministic + bootstrap.
5. **Diagnostics**: inspect heatmaps, latest diagonal, and outliers.
6. **Results**: review IBNR, ultimates, and charts.
7. **AI Assistant**: ask natural-language actuarial questions and generate narrative outputs.
8. **Reports**: download PDF report and full Excel output package.

## Data template

Use `data/sample_template.csv` as a starting point for user uploads. Mapping UI allows custom source schemas, so the app does not require fixed column names.

## Notes on security and data safety

- Uploaded data is held only in the active Streamlit session state.
- The AI assistant receives summary context (mapping, assumptions, reserves, diagnostics), not raw full datasets by default.
- API keys are read from environment variables.
- Input validation and user-friendly error handling are included throughout workflow.

## Roadmap

Planned enhancements:
- Bornhuetter-Ferguson
- Cape Cod
- GLM diagnostics
- advanced stochastic reserving extensions
- scenario comparison dashboard
- reserve movement analysis across saved runs
- multi-user projects
- database persistence
- authentication/authorization

## Testing (suggested)

```bash
python -m py_compile app.py reserving_app/core/*.py reserving_app/services/*.py
pytest -q
```

## Troubleshooting

- **OpenAI assistant returns key error**  
  Ensure `.env` contains `OPENAI_API_KEY=...` and restart Streamlit.
- **Excel upload issues**  
  Confirm file extension is `.xlsx` (not `.xls`) and sheet has headers in row 1.
- **Model run fails after exclusion input**  
  Exclusions must be `row,col` pairs separated by semicolons, e.g. `0,1;2,3`.

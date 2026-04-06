from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from reserving_app.core.logging_config import setup_logging
from reserving_app.services.ai_assistant import AIContext, ask_assistant
from reserving_app.services.charts import bootstrap_histogram
from reserving_app.services.charts import bootstrap_comparison_histogram
from reserving_app.services.charts import cumulative_vs_ultimate
from reserving_app.services.charts import development_factor_chart
from reserving_app.services.charts import heatmap_from_triangle
from reserving_app.services.charts import percentile_chart
from reserving_app.services.charts import reserve_by_origin_chart
from reserving_app.services.data_ingestion import detect_excel_sheets, load_file
from reserving_app.services.diagnostics import (
    detect_outlier_link_ratios,
    negative_value_warning,
    non_monotonic_cumulative_warning,
    sparse_data_warnings,
)
from reserving_app.services.mapping_validation import ALL_FIELDS, suggest_mapping, validate_mapping
from reserving_app.services.input_parsing import parse_exclusion_cells
from reserving_app.services.reporting import build_pdf_report, export_tables_to_excel
from reserving_app.services.reserving_models import run_bootstrap_chain_ladder, run_chain_ladder
from reserving_app.services.reserving_models import run_bootstrap_odp_distribution, run_bootstrap_odp_variability_comparison
from reserving_app.services.triangle_builder import (
    build_triangle,
    build_triangle_from_development_matrix,
    convert_origin_calendar_to_development_triangle,
    parse_development_period_label,
    parse_period_label,
)

setup_logging()
st.set_page_config(page_title="ActuaryGPT Reserving Studio", layout="wide")


if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []

st.title("ActuaryGPT Reserving Studio")
st.caption("Professional reserving workflow for paid/incurred claims, diagnostics, and AI commentary.")

section = st.sidebar.radio(
    "Navigate",
    [
        "Upload Data",
        "Map Fields",
        "Build Triangle",
        "Methods",
        "Diagnostics",
        "Results",
        "AI Assistant",
        "Reports",
    ],
)

with st.sidebar:
    st.markdown("---")
    demo_mode = st.button("One-click demo mode")

if demo_mode:
    try:
        demo_df = pd.read_csv("data/demo_claims.csv")
        st.session_state.df = demo_df
        st.session_state.file_name = "demo_claims.csv"
        st.session_state.mapping = suggest_mapping(list(demo_df.columns))
        st.success("Demo dataset loaded.")
    except Exception as exc:
        st.error(f"Unable to load demo dataset: {exc}")

if section == "Upload Data":
    st.subheader("Upload claims data")
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

    if uploaded:
        file_bytes = uploaded.read()
        sheet = None
        if uploaded.name.lower().endswith("xlsx"):
            sheets = detect_excel_sheets(file_bytes)
            sheet = st.selectbox("Select sheet", sheets)

        if st.button("Ingest file"):
            try:
                result = load_file(uploaded.name, file_bytes, sheet)
                st.session_state.df = result.df
                st.session_state.file_name = uploaded.name
                st.session_state.mapping = suggest_mapping(list(result.df.columns))
                st.session_state.audit_trail.append(f"[{datetime.utcnow().isoformat()}] Uploaded {uploaded.name}")

                st.success("File ingested and cleaned.")
                st.write("Data cleaning notes:")
                for note in result.cleaning_notes:
                    st.write(f"- {note}")
            except Exception as exc:
                st.error(f"Could not ingest file: {exc}")

    if "df" in st.session_state:
        st.write("Preview")
        st.dataframe(st.session_state.df.head(50), use_container_width=True)

if section == "Map Fields":
    st.subheader("Map source fields to reserving fields")

    if "df" not in st.session_state:
        st.info("Upload data first.")
    else:
        cols = list(st.session_state.df.columns)
        mapping = st.session_state.get("mapping", {})

        selected_basis = st.selectbox(
            "Triangle basis",
            ["paid_amount", "incurred_amount", "reported_count", "paid_count"],
            index=0,
        )
        st.session_state.triangle_basis = selected_basis

        period_grain = st.selectbox("Period grain", ["Yearly", "Half-Yearly", "Quarterly", "Monthly"], index=0)
        st.session_state.period_grain = period_grain

        new_mapping = {}
        for field in ALL_FIELDS:
            default = mapping.get(field)
            idx = cols.index(default) + 1 if default in cols else 0
            choice = st.selectbox(field, options=["<None>"] + cols, index=idx, key=f"map_{field}")
            new_mapping[field] = None if choice == "<None>" else choice

        validation = validate_mapping(new_mapping, st.session_state.df, selected_basis)
        if validation.valid:
            st.success("Mapping valid.")
            st.session_state.mapping = new_mapping
            st.session_state.audit_trail.append(f"[{datetime.utcnow().isoformat()}] Updated field mapping")
        else:
            for err in validation.errors:
                st.error(err)

if section == "Build Triangle":
    st.subheader("Triangle construction")

    if "df" not in st.session_state:
        st.info("Upload data first.")
    else:
        raw_df = st.session_state.df.copy()
        st.write("Uploaded raw data preview")
        st.dataframe(raw_df.head(50), use_container_width=True)

        input_format = st.selectbox(
            "Input Format",
            ["Mapped Transactional Data", "Development Triangle", "Origin × Calendar Movement Matrix"],
        )
        cols = raw_df.columns.tolist()
        if input_format == "Mapped Transactional Data":
            origin_col = None
            triangle_type = "Incremental"
            dev_cols = []
            calendar_cols = []
            if "mapping" not in st.session_state:
                st.warning("Map fields first before building a triangle from transactional data.")
            else:
                mapping = st.session_state.mapping
                basis = st.session_state.get("triangle_basis", "paid_amount")
                basis_col = mapping.get(basis)
                st.caption(f"Using mapped basis: **{basis}** → `{basis_col}`")

                segment_filter = None
                segment_col = mapping.get("segment")
                if segment_col and segment_col in raw_df.columns:
                    segment_options = ["All"] + sorted(raw_df[segment_col].dropna().astype(str).unique().tolist())
                    selected_segment = st.selectbox("Segment filter", segment_options, index=0)
                    segment_filter = None if selected_segment == "All" else selected_segment
                st.session_state.segment_filter = segment_filter
        else:
            origin_col = st.selectbox("Origin period column", cols, index=0)
            segment_filter = None

        if input_format == "Development Triangle":
            triangle_type = st.radio("Development triangle data type", ["Incremental", "Cumulative"], horizontal=True)
            candidate_cols = [c for c in cols if c != origin_col]
            default_dev_cols = []
            for c in candidate_cols:
                try:
                    parse_development_period_label(c)
                    default_dev_cols.append(c)
                except Exception:
                    continue
            dev_cols = st.multiselect(
                "Development period columns",
                options=candidate_cols,
                default=default_dev_cols or candidate_cols,
            )
        else:
            triangle_type = "Incremental"
            candidate_cols = [c for c in cols if c != origin_col]
            calendar_cols = st.multiselect("Calendar/valuation period columns", options=candidate_cols, default=candidate_cols)

        if st.button("Build Triangle"):
            try:
                if input_format == "Development Triangle":
                    if not dev_cols:
                        raise ValueError("Please select at least one development period column.")
                    tri = build_triangle_from_development_matrix(raw_df, origin_col, dev_cols, triangle_type)
                    st.session_state.audit_trail.append(
                        f"[{datetime.utcnow().isoformat()}] Built {triangle_type.lower()} development triangle"
                    )
                elif input_format == "Mapped Transactional Data":
                    mapping = st.session_state.get("mapping", {})
                    basis = st.session_state.get("triangle_basis", "paid_amount")
                    validation = validate_mapping(mapping, raw_df, basis)
                    if not validation.valid:
                        raise ValueError("Cannot build triangle from mapped data: " + "; ".join(validation.errors))
                    tri = build_triangle(
                        raw_df,
                        mapping,
                        basis,
                        st.session_state.get("period_grain", "Yearly"),
                        segment_filter=st.session_state.get("segment_filter"),
                    )
                    st.session_state.audit_trail.append(
                        f"[{datetime.utcnow().isoformat()}] Built mapped transactional triangle ({basis})"
                    )
                else:
                    if not calendar_cols:
                        raise ValueError("Please select at least one calendar/valuation period column.")
                    # validate period labels prior to conversion
                    _ = [parse_period_label(str(x)) for x in raw_df[origin_col].dropna().astype(str).tolist()]
                    _ = [parse_period_label(str(x)) for x in calendar_cols]
                    tri = convert_origin_calendar_to_development_triangle(raw_df, origin_col, calendar_cols)
                    st.session_state.audit_trail.append(
                        f"[{datetime.utcnow().isoformat()}] Built development triangle from origin×calendar matrix"
                    )

                st.session_state.triangle = tri
                st.session_state.input_format = input_format
                st.success("Triangle built successfully.")
            except Exception as exc:
                st.error(f"Could not build triangle: {exc}")

        if "triangle" in st.session_state:
            st.write("Transformed incremental development triangle preview")
            st.dataframe(st.session_state.triangle.incremental, use_container_width=True)
            st.write("Transformed cumulative development triangle preview")
            st.dataframe(st.session_state.triangle.cumulative, use_container_width=True)

            view = st.radio("Download view", ["Incremental", "Cumulative"], horizontal=True)
            table = st.session_state.triangle.incremental if view == "Incremental" else st.session_state.triangle.cumulative
            csv_bytes = table.to_csv(index=True).encode("utf-8")
            st.download_button("Download triangle CSV", data=csv_bytes, file_name="triangle.csv")

if section == "Methods":
    st.subheader("Reserving methods")

    if "triangle" not in st.session_state:
        st.info("Build a triangle first.")
    else:
        apply_tail = st.checkbox("Apply tail factor (1.02)", value=False)
        n_sims = st.number_input("Bootstrap simulation count", min_value=200, max_value=10000, step=100, value=1000)
        exclusions_text = st.text_input("Exclude link ratio cells (row,col pairs, e.g. 0,1;2,3)")
        exclusion_set = set()
        try:
            exclusion_set = parse_exclusion_cells(exclusions_text)
        except ValueError as exc:
            st.error(str(exc))

        if st.button("Run methods"):
            if exclusions_text.strip() and not exclusion_set:
                st.warning("Fix exclusion format before running methods.")
            else:
                try:
                    with st.spinner("Running deterministic and bootstrap models..."):
                        det = run_chain_ladder(st.session_state.triangle.cumulative, apply_tail, exclusions=exclusion_set)
                        boot = run_bootstrap_chain_ladder(st.session_state.triangle.cumulative, n_sims=int(n_sims))
                        st.session_state.det_result = det
                        st.session_state.boot_result = boot
                        st.session_state.assumptions = {
                            "tail_factor": apply_tail,
                            "bootstrap_sims": int(n_sims),
                            "excluded_cells": sorted(list(exclusion_set)),
                        }
                        st.session_state.audit_trail.append(f"[{datetime.utcnow().isoformat()}] Ran models with {n_sims} sims")
                    st.success("Models completed.")
                except Exception as exc:
                    st.error(f"Model run failed: {exc}")

if section == "Diagnostics":
    st.subheader("Diagnostics")

    if "det_result" not in st.session_state:
        st.info("Run methods first.")
    else:
        tri_inc = st.session_state.triangle.incremental
        tri = st.session_state.triangle.cumulative
        det = st.session_state.det_result

        for warning in sparse_data_warnings(tri):
            st.warning(warning)
        for warning in negative_value_warning(tri_inc):
            st.warning(warning)
        for warning in non_monotonic_cumulative_warning(tri):
            st.warning(warning)

        link_ratios = det.diagnostics["link_ratios"]
        outliers = detect_outlier_link_ratios(link_ratios)

        st.plotly_chart(heatmap_from_triangle(tri_inc, "Incremental Development Triangle Heatmap"), use_container_width=True)
        st.plotly_chart(heatmap_from_triangle(tri, "Cumulative Triangle Heatmap"), use_container_width=True)
        st.plotly_chart(heatmap_from_triangle(link_ratios.fillna(0), "Link Ratio Heatmap"), use_container_width=True)

        st.write("Latest diagonal")
        st.dataframe(st.session_state.triangle.latest_diagonal, use_container_width=True)

        st.write("Unusual link ratios (z-score threshold = 2)")
        st.dataframe(outliers, use_container_width=True)

if section == "Results":
    st.subheader("Results")

    if "det_result" not in st.session_state:
        st.info("Run methods first.")
    else:
        det = st.session_state.det_result
        boot = st.session_state.boot_result
        latest = st.session_state.triangle.cumulative.replace(0, pd.NA).ffill(axis=1).iloc[:, -1].fillna(0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total IBNR", f"{det.ibnr.sum():,.2f}")
            st.metric("Total Ultimate", f"{det.ultimates.sum():,.2f}")
        with col2:
            st.metric("Bootstrap mean reserve", f"{boot.summary['mean']:,.2f}")
            st.metric("Bootstrap 95th percentile", f"{boot.summary['95%']:,.2f}")

        st.plotly_chart(development_factor_chart(det.selected_ldf), use_container_width=True)
        st.plotly_chart(reserve_by_origin_chart(det.ibnr), use_container_width=True)
        st.plotly_chart(cumulative_vs_ultimate(latest, det.ultimates), use_container_width=True)
        st.plotly_chart(bootstrap_histogram(boot.reserve_distribution), use_container_width=True)
        st.plotly_chart(percentile_chart(boot.summary), use_container_width=True)

        st.write("Selected development factors")
        st.dataframe(det.selected_ldf.rename("factor"))
        st.write("IBNR by origin")
        st.dataframe(det.ibnr.rename("ibnr"))

        with st.expander("Bootstrap ODP plots (Chainladder examples adapted)", expanded=False):
            st.caption("Uses current selected development triangle from this app.")
            bs_default = int(st.session_state.get("assumptions", {}).get("bootstrap_sims", 1000))
            bs_sims = st.number_input("Bootstrap simulations (ODP plot)", min_value=200, max_value=20000, value=bs_default, step=100)
            bs_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
            run_variability = st.checkbox("Run variability comparison (drop outlier link ratios)", value=True)
            drop_high_count = st.number_input("drop_high count", min_value=0, max_value=10, value=1, step=1)
            drop_low_count = st.number_input("drop_low count", min_value=0, max_value=10, value=1, step=1)

            if st.button("Run Bootstrap ODP plots"):
                try:
                    basic_dist = run_bootstrap_odp_distribution(
                        st.session_state.triangle.cumulative,
                        n_sims=int(bs_sims),
                        random_state=int(bs_seed),
                    )
                    st.plotly_chart(bootstrap_histogram(basic_dist), use_container_width=True)
                    basic_summary = basic_dist.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).rename("value")
                    st.write("Basic Bootstrap ODP summary")
                    st.dataframe(basic_summary.to_frame())

                    if run_variability:
                        comp = run_bootstrap_odp_variability_comparison(
                            st.session_state.triangle.cumulative,
                            n_sims=int(bs_sims),
                            random_state=int(bs_seed),
                            drop_high_count=int(drop_high_count),
                            drop_low_count=int(drop_low_count),
                        )
                        st.plotly_chart(bootstrap_comparison_histogram(comp), use_container_width=True)
                        st.write("Variability comparison summary")
                        st.dataframe(comp.describe(percentiles=[0.5, 0.75, 0.9, 0.95]))
                except Exception as exc:
                    st.error(f"Bootstrap ODP plots could not be generated for this triangle: {exc}")

if section == "AI Assistant":
    st.subheader("ChatGPT reserve assistant")

    if "det_result" not in st.session_state:
        st.info("Run methods first.")
    else:
        prompt_buttons = {
            "Summarise the results": "Summarise the results in plain language for a reserving review.",
            "Explain unusual link ratios": "Explain unusual link ratios and potential data or claim drivers.",
            "Draft a reserve committee report": "Draft a reserve committee report with key decisions and caveats.",
            "Explain reserve uncertainty": "Explain reserve uncertainty based on bootstrap outputs.",
            "Highlight data quality concerns": "Highlight data quality concerns and likely reserving impacts.",
        }

        cols = st.columns(len(prompt_buttons))
        question = st.text_area("Ask a question about the results")
        for i, (label, text) in enumerate(prompt_buttons.items()):
            if cols[i].button(label):
                question = text

        if st.button("Ask Assistant") and question:
            try:
                det = st.session_state.det_result
                boot = st.session_state.boot_result
                context = AIContext(
                    mapping=st.session_state.mapping,
                    assumptions=st.session_state.get("assumptions", {}),
                    reserve_summary={
                        "total_ibnr": float(det.ibnr.sum()),
                        "ibnr_by_origin": det.ibnr.to_dict(),
                        "ultimates_by_origin": det.ultimates.to_dict(),
                    },
                    diagnostics_summary={
                        "selected_ldf": det.selected_ldf.to_dict(),
                        "cdf": det.cdf.to_dict(),
                    },
                    chart_summary={
                        "bootstrap_percentiles": {k: float(v) for k, v in boot.summary.to_dict().items()},
                    },
                )
                answer = ask_assistant(question, context)
                st.session_state.last_ai_answer = answer
                st.write(answer)
            except Exception as exc:
                st.error(f"Assistant request failed: {exc}")

if section == "Reports":
    st.subheader("Reporting")

    if "det_result" not in st.session_state:
        st.info("Run methods first.")
    else:
        comments = st.text_area("Actuary comments / notes")
        if st.button("Generate PDF report"):
            try:
                pdf = build_pdf_report(
                    portfolio=st.session_state.get("segment_filter") or "All",
                    selected_factors=st.session_state.det_result.selected_ldf,
                    ibnr=st.session_state.det_result.ibnr,
                    bootstrap_summary=st.session_state.boot_result.summary,
                    comments=comments,
                )
                st.download_button("Download PDF", data=pdf, file_name="reserving_report.pdf", mime="application/pdf")
            except Exception as exc:
                st.error(f"Could not generate PDF report: {exc}")

        tables = {
            "triangle_incremental": st.session_state.triangle.incremental,
            "triangle_cumulative": st.session_state.triangle.cumulative,
            "link_ratios": st.session_state.det_result.diagnostics["link_ratios"],
            "selected_ldf": st.session_state.det_result.selected_ldf.to_frame("factor"),
            "ibnr": st.session_state.det_result.ibnr.to_frame("ibnr"),
            "bootstrap_summary": st.session_state.boot_result.summary.to_frame("value"),
        }
        try:
            xlsx_bytes = export_tables_to_excel(tables)
            st.download_button("Download all result tables (Excel)", data=xlsx_bytes, file_name="reserving_outputs.xlsx")
        except Exception as exc:
            st.error(f"Could not export Excel file: {exc}")

        st.write("Audit trail")
        st.dataframe(pd.DataFrame({"event": st.session_state.audit_trail}))

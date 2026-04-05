from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from app.core.logging_config import setup_logging
from app.services.ai_assistant import AIContext, ask_assistant
from app.services.charts import (
    bootstrap_histogram,
    cumulative_vs_ultimate,
    development_factor_chart,
    heatmap_from_triangle,
    percentile_chart,
    reserve_by_origin_chart,
)
from app.services.data_ingestion import detect_excel_sheets, load_file
from app.services.diagnostics import detect_outlier_link_ratios, sparse_data_warnings
from app.services.mapping_validation import ALL_FIELDS, suggest_mapping, validate_mapping
from app.services.reporting import build_pdf_report, export_tables_to_excel
from app.services.reserving_models import run_bootstrap_chain_ladder, run_chain_ladder
from app.services.triangle_builder import build_triangle

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
    demo_df = pd.read_csv("data/demo_claims.csv")
    st.session_state.df = demo_df
    st.session_state.file_name = "demo_claims.csv"
    st.session_state.mapping = suggest_mapping(list(demo_df.columns))
    st.success("Demo dataset loaded.")

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
            result = load_file(uploaded.name, file_bytes, sheet)
            st.session_state.df = result.df
            st.session_state.file_name = uploaded.name
            st.session_state.mapping = suggest_mapping(list(result.df.columns))
            st.session_state.audit_trail.append(f"[{datetime.utcnow().isoformat()}] Uploaded {uploaded.name}")

            st.success("File ingested and cleaned.")
            st.write("Data cleaning notes:")
            for note in result.cleaning_notes:
                st.write(f"- {note}")

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

    if "df" not in st.session_state or "mapping" not in st.session_state:
        st.info("Upload data and map fields first.")
    else:
        seg_col = st.session_state.mapping.get("segment")
        segment_filter = None
        if seg_col:
            options = ["<All>"] + sorted([str(x) for x in st.session_state.df[seg_col].dropna().unique()])
            pick = st.selectbox("Segment filter", options)
            segment_filter = None if pick == "<All>" else pick

        if st.button("Build Triangle"):
            tri = build_triangle(
                st.session_state.df,
                st.session_state.mapping,
                st.session_state.triangle_basis,
                st.session_state.period_grain,
                segment_filter=segment_filter,
            )
            st.session_state.triangle = tri
            st.session_state.segment_filter = segment_filter
            st.session_state.audit_trail.append(f"[{datetime.utcnow().isoformat()}] Built triangle ({st.session_state.triangle_basis})")

        if "triangle" in st.session_state:
            view = st.radio("View", ["Incremental", "Cumulative"], horizontal=True)
            table = st.session_state.triangle.incremental if view == "Incremental" else st.session_state.triangle.cumulative
            st.dataframe(table, use_container_width=True)

            csv_bytes = table.to_csv().encode("utf-8")
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
        if exclusions_text.strip():
            for token in exclusions_text.split(";"):
                row, col = token.split(",")
                exclusion_set.add((int(row.strip()), int(col.strip())))

        if st.button("Run methods"):
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

if section == "Diagnostics":
    st.subheader("Diagnostics")

    if "det_result" not in st.session_state:
        st.info("Run methods first.")
    else:
        tri = st.session_state.triangle.cumulative
        det = st.session_state.det_result

        for warning in sparse_data_warnings(tri):
            st.warning(warning)

        link_ratios = det.diagnostics["link_ratios"]
        outliers = detect_outlier_link_ratios(link_ratios)

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

if section == "Reports":
    st.subheader("Reporting")

    if "det_result" not in st.session_state:
        st.info("Run methods first.")
    else:
        comments = st.text_area("Actuary comments / notes")
        if st.button("Generate PDF report"):
            pdf = build_pdf_report(
                portfolio=st.session_state.get("segment_filter") or "All",
                selected_factors=st.session_state.det_result.selected_ldf,
                ibnr=st.session_state.det_result.ibnr,
                bootstrap_summary=st.session_state.boot_result.summary,
                comments=comments,
            )
            st.download_button("Download PDF", data=pdf, file_name="reserving_report.pdf", mime="application/pdf")

        tables = {
            "triangle_incremental": st.session_state.triangle.incremental,
            "triangle_cumulative": st.session_state.triangle.cumulative,
            "link_ratios": st.session_state.det_result.diagnostics["link_ratios"],
            "selected_ldf": st.session_state.det_result.selected_ldf.to_frame("factor"),
            "ibnr": st.session_state.det_result.ibnr.to_frame("ibnr"),
            "bootstrap_summary": st.session_state.boot_result.summary.to_frame("value"),
        }
        xlsx_bytes = export_tables_to_excel(tables)
        st.download_button("Download all result tables (Excel)", data=xlsx_bytes, file_name="reserving_outputs.xlsx")

        st.write("Audit trail")
        st.dataframe(pd.DataFrame({"event": st.session_state.audit_trail}))

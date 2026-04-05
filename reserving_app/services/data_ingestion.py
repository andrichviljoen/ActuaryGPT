from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    df: pd.DataFrame
    sheet_names: list[str]
    inferred_types: dict[str, str]
    cleaning_notes: list[str]


ALLOWED_EXTENSIONS = {"csv", "xlsx"}


def detect_excel_sheets(file_bytes: bytes) -> list[str]:
    with pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl") as xl:
        return xl.sheet_names


def load_file(file_name: str, file_bytes: bytes, sheet_name: str | None = None) -> IngestionResult:
    extension = file_name.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}. Use CSV or XLSX.")

    if extension == "csv":
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
        sheet_names: list[str] = []
    else:
        sheet_names = detect_excel_sheets(file_bytes)
        selected_sheet = sheet_name or sheet_names[0]
        raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected_sheet, engine="openpyxl")

    cleaned_df, notes = clean_dataset(raw_df)
    inferred_types = {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()}

    return IngestionResult(
        df=cleaned_df,
        sheet_names=sheet_names,
        inferred_types=inferred_types,
        cleaning_notes=notes,
    )


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    before_rows = len(df)

    df = df.dropna(how="all")
    if len(df) != before_rows:
        notes.append(f"Removed {before_rows - len(df)} blank rows.")

    before_rows = len(df)
    df = df.drop_duplicates()
    if len(df) != before_rows:
        notes.append(f"Removed {before_rows - len(df)} duplicate rows.")

    df = df.copy()
    for col in df.columns:
        if "date" in col.lower() or "period" in col.lower() or "accident" in col.lower() or "valuation" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed
                notes.append(f"Parsed date-like column '{col}' to datetime where possible.")

        if df[col].dtype == "object":
            numeric_attempt = pd.to_numeric(df[col], errors="coerce")
            if numeric_attempt.notna().sum() > 0 and numeric_attempt.notna().sum() >= (0.6 * len(df)):
                df[col] = numeric_attempt
                notes.append(f"Converted mostly numeric text column '{col}' to numeric.")

    missing_summary = df.isna().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if not missing_cols.empty:
        notes.append(
            "Missing values detected in columns: "
            + ", ".join([f"{c} ({int(v)})" for c, v in missing_cols.items()])
        )

    logger.info("Data cleaned. Final rows=%s, cols=%s", df.shape[0], df.shape[1])
    return df, notes

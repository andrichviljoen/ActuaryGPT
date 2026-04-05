from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_FIELDS = ["origin_period", "development_period"]
MEASURE_FIELDS = ["paid_amount", "incurred_amount", "reported_count", "paid_count"]
OPTIONAL_FIELDS = ["claim_id", "segment"]
ALL_FIELDS = REQUIRED_FIELDS + MEASURE_FIELDS + OPTIONAL_FIELDS

SUGGESTION_KEYWORDS = {
    "origin_period": ["origin", "accident", "ay", "occurrence"],
    "development_period": ["dev", "valuation", "age", "dy", "lag"],
    "paid_amount": ["paid", "payment"],
    "incurred_amount": ["incurred", "case", "reported_loss", "total_incurred"],
    "reported_count": ["reported_count", "reported claims", "reported"],
    "paid_count": ["paid_count", "paid claims"],
    "claim_id": ["claim", "id", "claim_number"],
    "segment": ["lob", "segment", "line", "product", "class"],
}


@dataclass
class MappingValidationResult:
    valid: bool
    errors: list[str]


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    suggestions: dict[str, str | None] = {field: None for field in ALL_FIELDS}
    lower_cols = {col.lower(): col for col in columns}

    for field, keywords in SUGGESTION_KEYWORDS.items():
        for col_lower, original_col in lower_cols.items():
            if any(keyword in col_lower for keyword in keywords):
                suggestions[field] = original_col
                break

    return suggestions


def validate_mapping(mapping: dict[str, str | None], df: pd.DataFrame, triangle_basis: str) -> MappingValidationResult:
    errors: list[str] = []

    for req in REQUIRED_FIELDS:
        if not mapping.get(req):
            errors.append(f"Missing required mapping for '{req}'.")

    if triangle_basis not in MEASURE_FIELDS:
        errors.append("Triangle basis must be one of paid_amount, incurred_amount, reported_count, paid_count.")

    basis_col = mapping.get(triangle_basis)
    if not basis_col:
        errors.append(f"You selected basis '{triangle_basis}' but did not map a source column for it.")
    elif basis_col not in df.columns:
        errors.append(f"Mapped basis column '{basis_col}' not found in dataset.")

    for target, source in mapping.items():
        if source and source not in df.columns:
            errors.append(f"Mapped source column '{source}' for '{target}' is not in dataset.")

    if mapping.get("origin_period") and mapping.get("development_period"):
        if mapping["origin_period"] == mapping["development_period"]:
            errors.append("Origin period and development period cannot be the same column.")

    return MappingValidationResult(valid=(len(errors) == 0), errors=errors)

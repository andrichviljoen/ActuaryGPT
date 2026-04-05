from __future__ import annotations


def parse_exclusion_cells(raw_text: str) -> set[tuple[int, int]]:
    """
    Parse exclusion cells from UI text input in the format `row,col;row,col`.
    """
    exclusions: set[tuple[int, int]] = set()
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return exclusions

    for token in cleaned.split(";"):
        token = token.strip()
        if not token:
            continue
        if "," not in token:
            raise ValueError(f"Invalid exclusion token '{token}'. Expected row,col.")
        row_text, col_text = token.split(",", maxsplit=1)
        try:
            row = int(row_text.strip())
            col = int(col_text.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid exclusion token '{token}'. Row and col must be integers.") from exc
        if row < 0 or col < 0:
            raise ValueError(f"Invalid exclusion token '{token}'. Row and col must be non-negative.")
        exclusions.add((row, col))

    return exclusions

import io

import pandas as pd
import pytest

from reserving_app.services.data_ingestion import load_file


def test_load_file_excel_invalid_sheet_name_raises_clear_error():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Claims", index=False)
    xlsx_bytes = buff.getvalue()

    with pytest.raises(ValueError, match="Sheet 'Missing' not found"):
        load_file("claims.xlsx", xlsx_bytes, sheet_name="Missing")


def test_load_file_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file("demo_claims.txt", b"abc")

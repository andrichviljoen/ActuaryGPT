import pandas as pd

from reserving_app.services.mapping_validation import validate_mapping
from app.services.mapping_validation import validate_mapping


def test_mapping_validation_ok():
    df = pd.DataFrame({"origin": [2020], "dev": [12], "paid": [100.0]})
    mapping = {
        "origin_period": "origin",
        "development_period": "dev",
        "paid_amount": "paid",
        "incurred_amount": None,
        "reported_count": None,
        "paid_count": None,
        "claim_id": None,
        "segment": None,
    }
    result = validate_mapping(mapping, df, "paid_amount")
    assert result.valid

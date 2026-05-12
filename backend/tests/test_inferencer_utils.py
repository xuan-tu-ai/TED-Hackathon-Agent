import pandas as pd
import pytest

from app.anomaly_detection_v6.inference import utils


def test_row_to_dict_from_dict():
    row = {"a": 1}
    assert utils._row_to_dict(row) == {"a": 1}


def test_row_to_dict_from_series():
    series = pd.Series({"a": 1})
    assert utils._row_to_dict(series) == {"a": 1}


def test_row_to_dict_invalid():
    with pytest.raises(TypeError):
        utils._row_to_dict(123)


def test_sanitize_start_time():
    value = "2026/05/11 10:20:30"
    assert utils._sanitize_start_time(value) == "2026_05_11_10_20_30"

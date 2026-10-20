import pandas as pd

from src.data_processing.validator import validate_gold_prices


def test_valid_gold_prices():
    """正常行情应当通过校验。"""

    df = pd.DataFrame({
        "symbol": ["XAU/USD", "XAU/USD"],
        "timestamp": [
            "2026-10-18 09:00:00",
            "2026-10-18 10:00:00",
        ],
        "open": [4300.0, 4310.0],
        "high": [4320.0, 4330.0],
        "low": [4290.0, 4300.0],
        "close": [4310.0, 4320.0],
        "volume": [100, 120],
        "source": ["test", "test"],
    })

    result = validate_gold_prices(df)

    assert result["valid"] is True
    assert result["output_rows"] == 2
    assert result["removed_rows"] == 0


def test_invalid_price_range():
    """最高价低于最低价的数据应被排除。"""

    df = pd.DataFrame({
        "symbol": ["XAU/USD"],
        "timestamp": ["2026-10-18 09:00:00"],
        "open": [4300.0],
        "high": [4280.0],
        "low": [4290.0],
        "close": [4310.0],
        "volume": [100],
        "source": ["test"],
    })

    result = validate_gold_prices(df)

    assert result["valid"] is False
    assert result["output_rows"] == 0
    assert result["removed_rows"] == 1


def test_duplicate_prices():
    """重复行情应当只保留一条。"""

    df = pd.DataFrame({
        "symbol": ["XAU/USD", "XAU/USD"],
        "timestamp": [
            "2026-10-18 09:00:00",
            "2026-10-18 09:00:00",
        ],
        "open": [4300.0, 4300.0],
        "high": [4320.0, 4320.0],
        "low": [4290.0, 4290.0],
        "close": [4310.0, 4310.0],
        "volume": [100, 100],
        "source": ["test", "test"],
    })

    result = validate_gold_prices(df)

    assert result["valid"] is False
    assert result["output_rows"] == 1
    assert result["removed_rows"] == 1
# 本文件是基于pandas下用于数据清洗的文件
import pandas as pd


def validate_gold_prices(df: pd.DataFrame) -> dict:
    """
    校验黄金行情数据。

    参数：
        df: 包含黄金行情的 DataFrame。

    返回：
        dict: 包含校验结果、错误信息和清洗后数据。
    """

    required_columns = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    ]

    # 1. 检查必要字段是否存在
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return {
            "valid": False,
            "errors": [f"缺少必要字段：{missing_columns}"],
            "data": None,
        }

    # 2. 复制数据，避免修改调用者原始数据
    cleaned_df = df.copy()

    errors = []

    # 3. 检查并转换时间格式
    cleaned_df["timestamp"] = pd.to_datetime(
        cleaned_df["timestamp"],
        errors="coerce",
        utc=True,
    )

    invalid_timestamps = cleaned_df["timestamp"].isna()

    if invalid_timestamps.any():
        errors.append(
            f"发现 {invalid_timestamps.sum()} 条无效时间记录"
        )

    # 4. 检查价格字段是否为数值
    price_columns = ["open", "high", "low", "close"]

    for column in price_columns:
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        )

    # 5. 检查价格是否缺失
    missing_prices = cleaned_df[price_columns].isna().any(axis=1)

    if missing_prices.any():
        errors.append(
            f"发现 {missing_prices.sum()} 条价格缺失或格式错误的记录"
        )

    # 6. 检查价格是否为正数
    invalid_prices = (
        cleaned_df[price_columns].le(0).any(axis=1)
        & ~missing_prices
    )

    if invalid_prices.any():
        errors.append(
            f"发现 {invalid_prices.sum()} 条价格小于或等于零的记录"
        )

    # 7. 检查最高价、最低价之间的逻辑关系
    valid_price_rows = ~missing_prices & ~invalid_prices

    invalid_price_range = valid_price_rows & (
        (cleaned_df["high"] < cleaned_df["low"])
        | (cleaned_df["high"] < cleaned_df["open"])
        | (cleaned_df["high"] < cleaned_df["close"])
        | (cleaned_df["low"] > cleaned_df["open"])
        | (cleaned_df["low"] > cleaned_df["close"])
    )

    if invalid_price_range.any():
        errors.append(
            f"发现 {invalid_price_range.sum()} 条价格关系异常的记录"
        )

    # 8. 检查重复行情
    duplicate_rows = cleaned_df.duplicated(
        subset=["symbol", "timestamp", "source"],
        keep="first",
    )

    if duplicate_rows.any():
        errors.append(
            f"发现 {duplicate_rows.sum()} 条重复行情记录"
        )

    # 9. 只有通过基础检查的记录才进入清洗结果
    invalid_rows = (
        invalid_timestamps
        | missing_prices
        | invalid_prices
        | invalid_price_range
        | duplicate_rows
    )

    cleaned_df = cleaned_df.loc[~invalid_rows].copy()

    # 10. 按时间升序排列
    cleaned_df = cleaned_df.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "data": cleaned_df,
        "input_rows": len(df),
        "output_rows": len(cleaned_df),
        "removed_rows": int(invalid_rows.sum()),
    }
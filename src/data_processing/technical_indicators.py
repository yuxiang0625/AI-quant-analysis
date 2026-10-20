# 本文件用于各指标下黄金的技术分析
import pandas as pd

from database.db import get_connection


def load_gold_prices() -> pd.DataFrame:
    """从数据库读取黄金历史行情。"""

    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                source
            FROM gold_prices
            WHERE symbol = 'XAU/USD'
              AND source = 'XAUS'
            ORDER BY timestamp
            """,
            conn,
        )

    if df.empty:
        raise RuntimeError(
            "数据库中没有黄金行情，请先运行 gold_pipeline.py。"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MA、RSI 和 MACD 指标。"""

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 移动平均线
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()

    # RSI：使用14个交易日的平均上涨和平均下跌幅度
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss

    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD：12日指数移动平均线减去26日指数移动平均线
    ema12 = df["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["MACD_hist"] = (
        df["MACD"] - df["MACD_signal"]
    )

    return df


if __name__ == "__main__":
    gold_data = load_gold_prices()
    result = calculate_indicators(gold_data)

    print("黄金技术指标计算完成！")
    print(f"行情记录数：{len(result)}")

    print("\n最近十条行情与技术指标：")

    columns = [
        "timestamp",
        "close",
        "MA20",
        "MA50",
        "RSI14",
        "MACD",
        "MACD_signal",
        "MACD_hist",
    ]

    print(
        result[columns]
        .tail(10)
        .to_string(index=False)
    )
import requests
import pandas as pd


def fetch_gold_history() -> pd.DataFrame:
    """获取黄金现货 XAU/USD 的日线历史数据。"""

    url = "https://xaus.com/api/v1/chart"

    response = requests.get(
        url,
        params={
            "symbol": "xau",
            "range": "1y",
            "interval": "1d",
        },
        timeout=20,
    )
    response.raise_for_status()

    result = response.json()

    points = result.get("points", [])

    if not points:
        raise RuntimeError(
            f"接口没有返回历史行情数据，实际响应：{result}"
        )

    data = pd.DataFrame(points)

    # 将数据源字段转换为项目统一的字段名称
    data = data.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    # 如果接口没有提供完整的 OHLC 数据，就不伪造缺失字段
    required_price_columns = [
        "timestamp", "high", "low", "close"
    ]

    missing = [
        column
        for column in required_price_columns
        if column not in data.columns
    ]

    if missing:
        raise RuntimeError(
            f"接口返回的数据缺少必要字段：{missing}。"
            f"实际字段：{list(data.columns)}"
        )

    # 现货黄金没有统一的交易所成交量，因此不把成交量伪装成真实数据
    if "volume" not in data.columns:
        data["volume"] = float("nan")

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        unit="s",
        errors="coerce",
        utc=True,
    )

    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["symbol"] = "XAU/USD"
    data["source"] = "XAUS"

    return data[
        [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
        ]
    ].copy()


if __name__ == "__main__":
    gold_data = fetch_gold_history()

    print("黄金历史行情获取成功！")
    print("数据来源：XAUS")
    print("交易品种：XAU/USD 黄金现货")
    print(f"获取记录数：{len(gold_data)}")
    print("\n最近五条行情：")
    print(gold_data.tail().to_string(index=False))

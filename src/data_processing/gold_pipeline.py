# 本文件用于黄金行情验证处理
import pandas as pd

from database.db import initialize_database, get_connection
from src.data_sources.yahoo_finance import fetch_gold_history
from src.data_processing.validator import validate_gold_prices


def save_gold_prices(df: pd.DataFrame) -> int:
    """
    将通过验证的黄金行情保存到数据库。
    返回本次新增的记录数量。
    """

    if df.empty:
        return 0

    records = []

    for _, row in df.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])

        # 统一时间格式，保证同一时刻的数据能够正确识别
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        records.append((
            row["symbol"],
            timestamp.isoformat(),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            None if pd.isna(row["volume"]) else float(row["volume"]),
            row["source"],
        ))

    with get_connection() as conn:
        before = conn.total_changes

        conn.executemany("""
            INSERT INTO gold_prices (
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timestamp, source) DO NOTHING
        """, records)

        inserted_rows = conn.total_changes - before

    return inserted_rows


def run_gold_pipeline():
    """执行黄金行情获取、验证和入库流程。"""

    print("=" * 45)
    print("Aurum Intelligence | 黄金数据处理")
    print("=" * 45)

    # 1. 初始化数据库
    initialize_database()
    print("\n[1/4] 数据库初始化完成")

    # 2. 获取行情
    gold_data = fetch_gold_history()
    print(f"[2/4] 获取行情：{len(gold_data)} 条")

    # 3. 验证行情
    result = validate_gold_prices(gold_data)

    print(f"[3/4] 输入记录：{result['input_rows']}")
    print(f"      有效记录：{result['output_rows']}")
    print(f"      移除记录：{result['removed_rows']}")

    if result["errors"]:
        print("\n数据检查提示：")
        for error in result["errors"]:
            print(f"- {error}")

    if not result["valid"]:
        raise RuntimeError("行情数据未通过验证，停止入库。")

    valid_data = result["data"]

    # 4. 保存数据库
    inserted_rows = save_gold_prices(valid_data)

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT COUNT(*)
            FROM gold_prices
            WHERE symbol = ? AND source = ?
        """, ("XAU/USD", "XAUS"))

        total_rows = cursor.fetchone()[0]

    print(f"\n[4/4] 本次新增：{inserted_rows} 条")
    print(f"数据库中该品种累计记录：{total_rows} 条")
    print("\n最近五条已验证行情：")
    print(
        valid_data.tail(5).to_string(index=False)
    )

    print("\n黄金行情处理流程完成！")


if __name__ == "__main__":
    run_gold_pipeline()
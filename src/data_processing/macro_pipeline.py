# 该文件用于创建宏观数据入库程序
from src.data_sources.macro_data import fetch_all_macro_data
from database.db import initialize_database, save_macro_data, get_connection


def run_macro_pipeline():
    """获取、检查并保存宏观经济数据。"""

    print("=" * 50)
    print("Aurum Intelligence | 宏观数据入库")
    print("=" * 50)

    initialize_database()

    # 获取公开宏观经济数据
    macro_data = fetch_all_macro_data()

    if macro_data.empty:
        raise RuntimeError("没有获取到宏观经济数据。")

    # 检查关键字段
    required_columns = [
        "observation_date",
        "indicator",
        "value",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in macro_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"宏观数据缺少必要字段：{missing_columns}"
        )

    # 保存到 SQLite
    changed_rows = save_macro_data(macro_data)

    # 检查数据库实际记录数
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT indicator, COUNT(*), MAX(observation_date)
            FROM macro_data
            WHERE source = 'FRED'
            GROUP BY indicator
            ORDER BY indicator
            """
        )

        results = cursor.fetchall()

    print(f"\n本次新增或更新记录数：{changed_rows}")
    print("\n数据库中的宏观数据：")

    for indicator, count, latest_date in results:
        print(
            f"{indicator}: {count} 条记录，"
            f"最新观测日期：{latest_date}"
        )

    print("\n宏观数据入库完成！")


if __name__ == "__main__":
    run_macro_pipeline()
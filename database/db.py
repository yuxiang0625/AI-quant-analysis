import sqlite3
from pathlib import Path
from contextlib import contextmanager


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据库文件路径
DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATABASE_DIR / "aurum.db"


def _ensure_columns(conn, table, columns):
    """
    为已存在的表补齐缺失的列。

    SQLite 不支持 ADD COLUMN IF NOT EXISTS，因此先查 PRAGMA 再决定是否补。
    只做增量补列，不修改或删除已有列与数据。
    """

    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }

    for name, definition in columns.items():
        if name not in existing:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {name} {definition}"
            )


def initialize_database():
    """
    初始化数据库，创建项目所需的数据表。
    """

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as conn:

        # 黄金行情表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                volume REAL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, source)
            )
        """)

        # COMEX 黄金库存表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comex_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                registered REAL,
                eligible REAL,
                total REAL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(report_date, source)
            )
        """)

        # COMEX 黄金库存核验状态表
        # 记录每条库存数据的来源网址、核验结论与理由。
        # 状态取值：
        #   VERIFIED  与第二来源在同一 CME 业务日数值一致
        #   ACCEPTED  单源数据，通过全部自动校验且无告警
        #   REVIEW    存在告警或校验未通过，需人工复核
        #   REJECTED  已证伪，保留审计但不参与分析
        # 量化打分只采用 VERIFIED 与 ACCEPTED。
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comex_inventory_quality (
                report_date TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT,
                status TEXT NOT NULL DEFAULT 'REVIEW',
                reason TEXT,
                flags_json TEXT,
                reviewed_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (report_date, source)
            )
        """)

        # 为已存在的旧库补齐新增列（幂等，不重建表、不丢数据）
        # 注意：SQLite 的 ALTER TABLE ADD COLUMN 不允许非常量默认值，
        # 因此 updated_at 只能建成可空列，不能带 CURRENT_TIMESTAMP 默认值。
        _ensure_columns(conn, "comex_inventory", {
            # 来源接口给出的原始日期，使日期校正可审计、可回溯
            "source_date": "TEXT",
            "updated_at": "TEXT",
        })

        # 宏观经济数据表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                value REAL,
                release_date TEXT,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(indicator, observation_date, source)
            )
        """)

        # 机构观点表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS institution_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution TEXT NOT NULL,
                title TEXT NOT NULL,
                published_at TEXT,
                fetched_at TEXT NOT NULL,
                url TEXT NOT NULL,
                content TEXT,
                source_type TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url)
            )
        """)

        conn.commit()


@contextmanager
def get_connection():
    """
    提供数据库连接，并在使用结束后自动关闭。
    """

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH, timeout=30)

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def get_table_names():
    """
    获取当前数据库中所有的数据表名称。
    """

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        return [row[0] for row in cursor.fetchall()]

def save_macro_data(df):
    """将宏观经济数据保存到 SQLite 数据库。"""

    if df.empty:
        return 0

    records = []

    for _, row in df.iterrows():
        records.append(
            (
                row["indicator"],
                row["observation_date"].strftime("%Y-%m-%d"),
                float(row["value"]),
                None,
                row["source"],
            )
        )

    with get_connection() as conn:
        before = conn.total_changes

        conn.executemany(
            """
            INSERT INTO macro_data (
                indicator,
                observation_date,
                value,
                release_date,
                source
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(indicator, observation_date, source)
            DO UPDATE SET
                value = excluded.value
            """,
            records,
        )

        changed_rows = conn.total_changes - before

    return changed_rows

if __name__ == "__main__":

    initialize_database()

    print("数据库初始化完成！")
    print(f"数据库位置：{DATABASE_PATH}")
    print("已创建的数据表：")

    for table in get_table_names():
        print(f"- {table}")
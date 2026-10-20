# 本文件是每日数据更新的统一入口，供 Windows 计划任务调用
"""
Aurum 每日数据更新。

一条命令跑完三条数据链，任一步失败不影响其他步骤：
    1. 黄金行情        src.data_processing.gold_pipeline
    2. 宏观经济数据    src.data_processing.macro_pipeline
    3. COMEX 黄金库存  src.data_sources.comex_inventory_api

为什么必须这样组织：
    - 幂等：三个步骤都按唯一键去重，重复运行不会产生重复行，
      因此计划任务重跑、手工补跑、机器重启后补跑都是安全的。
    - 隔离：单步失败只记录并继续，避免行情源临时故障导致
      库存数据整天不更新。
    - 留痕：结果写入 data/daily_update/ 下的 JSON 与日志，
      便于事后排查「哪天断了、断在哪一步」。

用法：
    python -m scripts.daily_update              # 预览，不写库
    python -m scripts.daily_update --commit     # 正式更新（计划任务用这个）
    python -m scripts.daily_update --only comex # 只更新库存
"""

import argparse
import json
import logging
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import DATABASE_PATH, initialize_database  # noqa: E402

# 与 database.db 中的定义保持一致，避免各处硬编码路径
DB_PATH = DATABASE_PATH

RESULT_DIR = PROJECT_ROOT / "data" / "daily_update"
LOG_PATH = RESULT_DIR / "daily_update.log"
STATE_PATH = RESULT_DIR / "last_run.json"

STEP_GOLD = "gold"
STEP_MACRO = "macro"
STEP_COMEX = "comex"

ALL_STEPS = (STEP_GOLD, STEP_MACRO, STEP_COMEX)


def _setup_logging():
    """初始化日志；结果目录不可写时降级为纯控制台输出。"""

    handlers = [logging.StreamHandler(sys.stdout)]

    try:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_PATH, encoding="utf-8"))
    except OSError:
        # 目录只读或不可写：仅用控制台，不阻断更新
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _safe_write(path, payload):
    """落盘运行结果；写失败只告警，不影响更新本身。"""

    try:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        logging.warning("写入 %s 失败（不影响本次更新）：%s", path.name, exc)


def run_gold():
    """更新黄金行情。"""

    from src.data_processing.gold_pipeline import run_gold_pipeline

    result = run_gold_pipeline()

    return {"returned": result}


def run_macro():
    """更新宏观经济数据。"""

    from src.data_processing.macro_pipeline import run_macro_pipeline

    result = run_macro_pipeline()

    return {"returned": result}


def run_comex(commit):
    """更新 COMEX 黄金库存。"""

    from src.data_sources import comex_inventory_api as api

    summary = api.run(dry_run=not commit)

    return summary


STEP_RUNNERS = {
    STEP_GOLD: lambda commit: run_gold(),
    STEP_MACRO: lambda commit: run_macro(),
    STEP_COMEX: run_comex,
}


def table_counts():
    """读取各表行数，用于更新前后的对比。"""

    counts = {}

    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]

            for table in tables:
                try:
                    counts[table] = conn.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                except sqlite3.Error as exc:
                    counts[table] = f"error: {exc}"
    except sqlite3.Error as exc:
        logging.error("读取表行数失败：%s", exc)

    return counts


def inventory_status_counts():
    """读取库存核验状态分布，反映本次更新后数据的可信度构成。"""

    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='comex_inventory_quality'"
            ).fetchone()

            if not exists:
                return {}

            rows = conn.execute(
                "SELECT status, source, COUNT(*) "
                "FROM comex_inventory_quality "
                "GROUP BY status, source ORDER BY status, source"
            ).fetchall()

        return {
            f"{status}/{source}": count
            for status, source, count in rows
        }
    except sqlite3.Error as exc:
        logging.error("读取核验状态失败：%s", exc)
        return {}


def run(steps=ALL_STEPS, commit=False):
    """按顺序执行各步骤，单步失败不阻断后续步骤。"""

    _setup_logging()

    started = datetime.now()

    logging.info("=" * 62)
    logging.info("Aurum 每日数据更新")
    logging.info("模式：%s", "正式更新" if commit else "预览 DRY_RUN")
    logging.info("步骤：%s", ", ".join(steps))
    logging.info("数据库：%s", DB_PATH)
    logging.info("=" * 62)

    # 初始化是幂等的：新建缺失的表、为旧库补齐新增列，
    # 不重建表、不删除任何数据。
    if not DB_PATH.exists():
        logging.info("数据库不存在，执行初始化。")
    else:
        logging.info("数据库已存在，按需补齐表结构。")

    initialize_database()

    counts_before = table_counts()

    results = {}

    for step in steps:
        runner = STEP_RUNNERS.get(step)

        if runner is None:
            logging.warning("未知步骤：%s，已跳过。", step)
            continue

        logging.info("--- 步骤：%s ---", step)

        step_started = datetime.now()

        try:
            payload = runner(commit)

            elapsed = (datetime.now() - step_started).total_seconds()

            results[step] = {
                "status": "ok",
                "seconds": round(elapsed, 2),
                "result": payload,
            }

            logging.info("步骤 %s 完成，用时 %.2f 秒。", step, elapsed)

        except Exception as exc:
            elapsed = (datetime.now() - step_started).total_seconds()

            results[step] = {
                "status": "failed",
                "seconds": round(elapsed, 2),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

            # 单步失败只记录，继续后续步骤
            logging.error("步骤 %s 失败：%s", step, exc)

    counts_after = table_counts()

    diff = {
        table: {
            "before": counts_before.get(table),
            "after": counts_after.get(table),
        }
        for table in sorted(set(counts_before) | set(counts_after))
        if counts_before.get(table) != counts_after.get(table)
    }

    succeeded = [k for k, v in results.items() if v["status"] == "ok"]
    failed = [k for k, v in results.items() if v["status"] != "ok"]

    summary = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "steps_requested": list(steps),
        "steps_succeeded": succeeded,
        "steps_failed": failed,
        "ok": not failed,
        "table_row_changes": diff,
        "inventory_quality_status": inventory_status_counts(),
        "results": results,
    }

    _safe_write(STATE_PATH, summary)

    if diff:
        logging.info("表行数变化：%s", json.dumps(diff, ensure_ascii=False))
    else:
        logging.info("表行数无变化。")

    logging.info(
        "更新结束：成功 %s，失败 %s。", len(succeeded), len(failed)
    )

    return summary


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(
        description="Aurum 每日数据更新（行情 / 宏观 / COMEX 库存）"
    )

    parser.add_argument(
        "--commit",
        action="store_true",
        help="正式写入数据库；不加此参数则为预览模式，不写库",
    )

    parser.add_argument(
        "--only",
        choices=list(ALL_STEPS),
        action="append",
        help="只执行指定步骤，可重复传入；默认执行全部三步",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只在出错时输出日志",
    )

    args = parser.parse_args(argv)

    steps = tuple(args.only) if args.only else ALL_STEPS

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    summary = run(steps=steps, commit=args.commit)

    # 有任一步骤失败时返回非零退出码，便于计划任务告警
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

# 本文件负责从结构化公开接口采集 COMEX 黄金库存，替代原有搜索文章的方式
"""
COMEX 黄金库存采集（结构化接口版）。

为什么不再用 comex_auto_ingest.py：
    旧方式依赖 DeepSeek 搜索零散文章再正则解析，存在两个无法根治的问题：
    1. 日期口径错误 —— cngold 文章表格里的日期标签比 CME 真实业务日早 1 天；
    2. 字段错位 —— 同一行内 registered / eligible / total 可能来自不同日期，
       虽然三者加总自洽，但与权威源逐字段比对时对不上。
    本模块改为直接读取结构化接口，并在入库前做跨源交叉验证。

数据源与口径（已三方交叉验证）：
    东方财富 RPT_FUTUOPT_GOLDSIL
        - 日频，字段 STORAGE_OUNCE = COMEX 黄金总库存（盎司）
        - REPORT_DATE 比 CME 业务日早 1 天，需 +1 校正
        - 不提供 registered / eligible 拆分
    metalcharts 免费周频 CSV
        - 周频，含 registered_oz / eligible_oz / total_oz
        - report_date 即 CME 业务日，无需校正

日期偏移不写死：detect_date_offset() 会用重叠区间的数值匹配度反推偏移量，
一旦上游改了口径，程序会自行发现并告警，而不是静默产出错一天的数据。
"""

import csv
import io
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import requests


# =========================================================
# 1. 路径与常量
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "aurum.db"
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "comex_inventory_api.log"
EVIDENCE_DIR = DATA_DIR / "comex_inventory"

SOURCE_EASTMONEY = "eastmoney"
SOURCE_METALCHARTS = "metalcharts"
SOURCE_LEGACY = "cngold"

# 核验状态分级。量化打分默认采用 VERIFIED + ACCEPTED。
STATUS_VERIFIED = "VERIFIED"    # 与第二来源在同一 CME 业务日数值一致
STATUS_ACCEPTED = "ACCEPTED"    # 单源，但通过全部自动校验且无告警
STATUS_REVIEW = "REVIEW"        # 存在告警，需人工复核
STATUS_REJECTED = "REJECTED"    # 已证伪，保留审计但不参与分析

# 参与量化打分与正式分析的状态
TRUSTED_STATUSES = (STATUS_VERIFIED, STATUS_ACCEPTED)

EASTMONEY_API = "https://datacenter-web.eastmoney.com/api/data/v1/get"

METALCHARTS_CSV = (
    "https://metalcharts.org/data/"
    "comex-gold-silver-registered-eligible-weekly.csv"
)

# 东方财富黄金库存指标 ID（由其页面 JS 中的 reportName/filter 确认）
EASTMONEY_GOLD_ID = "EMI00069026"

# 各源相对 CME 业务日的日期偏移（天）。
#
# 东方财富 = 0：其 REPORT_DATE 本身就是 CME 业务日。用排除法证明——
#   1236 条日频记录呈完美的周一至周五分布、零周末；
#   若它是「业务日 -1」，则应出现周日且不含周五（实际有 247 个周五、零周日）；
#   若它是「业务日 +1」，则应出现周六且不含周一（实际有 231 个周一、零周六）。
#   两种假设均被排除，故偏移为 0。
#   注意：早期实现曾误设为 +1，会凭空制造 247 个周六业务日。
# metalcharts = 0：周频，report_date 多为周五，同样是合法业务日。
DATE_OFFSET_DAYS = {
    SOURCE_EASTMONEY: 0,
    SOURCE_METALCHARTS: 0,
}

# 跨源比对允许的日期窗口（天）。
# 日频与周频来源对「同一个值挂在哪一天」的约定可能相差一天
# （周频快照常以周末日标注、却包含前一业务日的收盘库存），
# 因此匹配用 ±1 天窗口，而冲突判定只在完全同一日期上进行。
CROSS_SOURCE_DATE_WINDOW_DAYS = 1

# 跨源比对的相对容差：0.01% 足以吸收四舍五入差异
CROSS_SOURCE_TOLERANCE = 0.0001

# 相邻记录变化超过该比例时标记复核，不自动判错
LARGE_CHANGE_THRESHOLD = 0.10

PAGE_SIZE = 500
TIMEOUT = 35
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://data.eastmoney.com/pmetal/comex/by.html",
}

METALCHARTS_HEADERS = {
    "User-Agent": HEADERS["User-Agent"],
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://metalcharts.org/data",
}

def _setup_logging():
    """
    初始化日志。

    日志文件不可写时降级为纯控制台输出，不能让整个模块导入失败
    （例如在只读环境或被测试导入时）。
    """

    handlers = [logging.StreamHandler(sys.stdout)]

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_PATH, encoding="utf-8"))
    except OSError:
        # 目录只读或不可写：仅用控制台，不阻断采集
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


_setup_logging()


# =========================================================
# 2. 网络请求
# =========================================================

def request_with_retry(url, params=None, headers=None, timeout=TIMEOUT):
    """带退避重试的只读 GET 请求。"""

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers or HEADERS,
                timeout=timeout,
            )

            response.raise_for_status()

            return response

        except requests.RequestException as exc:
            last_error = exc

            logging.warning(
                "请求失败（第 %s/%s 次）：%s", attempt, MAX_RETRIES, exc
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(f"请求多次失败：{url} -> {last_error}")


# =========================================================
# 3. 东方财富日频库存
# =========================================================

def fetch_eastmoney_gold_inventory(max_pages=40):
    """
    抓取东方财富 COMEX 黄金库存日频序列。

    返回 list[dict]，字段：
        source_date    接口原始 REPORT_DATE
        report_date    校正后的 CME 业务日
        total          总库存（盎司）
        ton            总库存（吨）
        source_url
    """

    rows = []
    page = 1

    while page <= max_pages:
        params = {
            "reportName": "RPT_FUTUOPT_GOLDSIL",
            "columns": "ALL",
            "quoteColumns": "",
            "source": "WEB",
            "client": "WEB",
            "pageSize": str(PAGE_SIZE),
            "pageNumber": str(page),
            "sortColumns": "REPORT_DATE",
            "sortTypes": "-1",
            "filter": (
                f'(INDICATOR_ID1="{EASTMONEY_GOLD_ID}")'
                '(@STORAGE_TON<>"NULL")'
            ),
        }

        response = request_with_retry(EASTMONEY_API, params=params)

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"东方财富返回非 JSON：{response.text[:300]}"
            ) from exc

        if not payload.get("success"):
            raise RuntimeError(
                f"东方财富接口返回失败：{payload.get('message')}"
            )

        result = payload.get("result") or {}
        batch = result.get("data") or []

        if not batch:
            break

        for item in batch:
            raw_date = (item.get("REPORT_DATE") or "")[:10]

            ounce = item.get("STORAGE_OUNCE")
            ton = item.get("STORAGE_TON")

            if raw_date is None or ounce in (None, ""):
                continue

            try:
                parsed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
                total_value = float(ounce)
            except (ValueError, TypeError):
                logging.warning("跳过无法解析的东方财富记录：%s", item)
                continue

            rows.append(
                {
                    "source": SOURCE_EASTMONEY,
                    "source_date": parsed_date.isoformat(),
                    "report_date": (
                        parsed_date
                        + timedelta(days=DATE_OFFSET_DAYS[SOURCE_EASTMONEY])
                    ).isoformat(),
                    "registered": None,
                    "eligible": None,
                    "total": total_value,
                    "ton": (
                        float(ton) if ton not in (None, "") else None
                    ),
                    "source_url": (
                        "https://data.eastmoney.com/pmetal/comex/by.html"
                    ),
                }
            )

        logging.info(
            "东方财富第 %s 页取得 %s 条（累计 %s）",
            page, len(batch), len(rows),
        )

        total_count = result.get("count") or 0

        if len(batch) < PAGE_SIZE or len(rows) >= total_count:
            break

        page += 1

    return rows


# =========================================================
# 4. metalcharts 周频库存（含 registered / eligible）
# =========================================================

def fetch_metalcharts_weekly(metal="gold"):
    """
    抓取 metalcharts 免费周频 CSV。

    该源 report_date 即 CME 业务日，且提供 registered/eligible 拆分。
    """

    response = request_with_retry(
        METALCHARTS_CSV, headers=METALCHARTS_HEADERS
    )

    response.encoding = "utf-8"

    reader = csv.DictReader(io.StringIO(response.text))

    rows = []

    for item in reader:
        if (item.get("metal") or "").strip().lower() != metal:
            continue

        raw_date = (item.get("report_date") or "").strip()

        try:
            parsed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            registered = float(item["registered_oz"])
            eligible = float(item["eligible_oz"])
            total = float(item["total_oz"])
        except (ValueError, TypeError, KeyError) as exc:
            logging.warning("跳过无法解析的 metalcharts 记录：%s（%s）", item, exc)
            continue

        offset = DATE_OFFSET_DAYS[SOURCE_METALCHARTS]

        rows.append(
            {
                "source": SOURCE_METALCHARTS,
                "source_date": parsed_date.isoformat(),
                "report_date": (
                    parsed_date + timedelta(days=offset)
                ).isoformat(),
                "registered": registered,
                "eligible": eligible,
                "total": total,
                "ton": None,
                "source_url": "https://metalcharts.org/data",
            }
        )

    logging.info("metalcharts 取得 %s 条 %s 记录", len(rows), metal)

    return rows


# =========================================================
# 5. 日期偏移自检
# =========================================================

def detect_date_offset(
    primary_rows,
    reference_rows,
    candidates=(-2, -1, 0, 1, 2),
):
    """
    用「值变化日」反推 primary 相对 reference 的日期偏移。

    为什么不能直接用全部记录匹配：
        COMEX 库存经常连续多日不变（实测约 16% 的相邻记录数值完全相同）。
        一段持续不变的值，在 -1 / 0 / +1 / +2 任意偏移下都能"碰巧"对上，
        导致检测结果失真——早期实现正是因此误判为 +1，
        凭空制造出 247 个周六业务日。
        因此只取「与前一条相比数值发生变化」的日期参与匹配，
        变化点才是有判别力的信号。

    两条硬约束：
        1. 只有值变化日参与匹配计数；
        2. 校正后若产生周末日期，该偏移直接判为无效。
           CME 业务日只可能是周一至周五，这是定义级约束，
           优先级高于匹配数量。

    reference 视为基准（metalcharts 使用 CME 业务日）。
    返回 (最优偏移天数, 详细诊断 dict)；无法判定时最优偏移为 None。
    """

    reference_by_date = {}

    for row in reference_rows:
        if row.get("total") is None:
            continue

        try:
            key = datetime.strptime(row["report_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError, KeyError):
            continue

        reference_by_date[key] = float(row["total"])

    # 只保留 primary 中「数值发生变化」的记录，按来源原始日期排序
    ordered = []

    for row in primary_rows:
        try:
            parsed = datetime.strptime(row["source_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError, KeyError):
            continue

        if row.get("total") is None:
            continue

        ordered.append((parsed, float(row["total"])))

    ordered.sort(key=lambda item: item[0])

    changed = []
    previous_value = None

    for parsed, value in ordered:
        if previous_value is None or abs(value - previous_value) >= 0.01:
            changed.append((parsed, value))

        previous_value = value

    scores = {}
    weekend_counts = {}

    for offset in candidates:
        matched = 0

        for parsed, value in changed:
            target = parsed + timedelta(days=offset)

            expected = reference_by_date.get(target)

            if expected is None:
                continue

            if abs(value - expected) <= expected * CROSS_SOURCE_TOLERANCE:
                matched += 1

        scores[offset] = matched

        # 硬约束：统计该偏移下会产生多少周末日期
        weekend = 0

        for row in primary_rows:
            try:
                parsed = datetime.strptime(
                    row["source_date"], "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError, KeyError):
                continue

            if (parsed + timedelta(days=offset)).weekday() >= 5:
                weekend += 1

        weekend_counts[offset] = weekend

    # 先按业务日有效性筛除会产生周末的偏移
    valid_offsets = [
        offset for offset in candidates if weekend_counts[offset] == 0
    ]

    detail = {
        "match_counts_by_offset": scores,
        "weekend_counts_by_offset": weekend_counts,
        "changed_days_used": len(changed),
        "valid_offsets": valid_offsets,
    }

    if not valid_offsets:
        # 所有偏移都会产生周末日期：数据本身异常，交由调用方中止
        detail["reason"] = (
            "所有候选偏移都会产生周末日期，违反 CME 业务日定义，"
            "无法判定偏移量。"
        )
        return None, detail

    if not any(scores[offset] for offset in valid_offsets):
        detail["reason"] = (
            "两源没有可判别的重叠值变化日，无法用交叉比对确定偏移；"
            "仅依据业务日有效性筛选。"
        )

    # 在合法偏移中选匹配数最高者；并列时取绝对值最小（改动最少）
    best = max(
        valid_offsets, key=lambda o: (scores[o], -abs(o))
    )

    detail["reason"] = (
        f"在 {len(changed)} 个值变化日上比对，"
        f"偏移 {best:+d} 天匹配 {scores[best]} 处且不产生周末日期。"
    )

    return best, detail


# =========================================================
# 6. 校验
# =========================================================

def _is_weekend_date(value):
    """
    判断日期字符串是否落在周末。

    CME 业务日只可能是周一至周五，因此周末日期是口径错误的硬指标。
    无法解析时返回 False，交由 validate_row 报告格式错误。
    """

    if not value:
        return False

    try:
        parsed = datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False

    return parsed.weekday() >= 5


def validate_row(row):
    """单条记录的完整性与自洽性检查，返回错误列表。"""

    errors = []
    flags = []

    try:
        parsed = datetime.strptime(row["report_date"], "%Y-%m-%d").date()
    except (ValueError, TypeError, KeyError):
        errors.append("report_date 格式错误")
        parsed = None

    if parsed:
        if parsed > datetime.now().date():
            errors.append("报告日期晚于当前日期")

        if parsed.weekday() >= 5:
            flags.append("报告日期落在周末，CME 业务日通常为工作日")

    values = {}

    for key in ("registered", "eligible", "total"):
        raw = row.get(key)

        if raw is None or raw == "":
            continue

        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{key} 不是有效数值")
            continue

        if not (0 < number < float("inf")):
            errors.append(f"{key} 不是正的有限数值")
            continue

        values[key] = number

    if "total" not in values:
        errors.append("缺少 total（总库存）")

    if len(values) == 3:
        drift = abs(
            values["registered"] + values["eligible"] - values["total"]
        )

        if drift > max(0.01, values["total"] * CROSS_SOURCE_TOLERANCE):
            errors.append(
                f"registered + eligible 与 total 不一致（差 {drift:.3f}）"
            )

    if not row.get("source_url"):
        errors.append("缺少来源网址")

    return errors, flags


def cross_validate(rows):
    """
    跨源交叉验证：在 ±CROSS_SOURCE_DATE_WINDOW_DAYS 天窗口内比对总库存。

    为什么必须允许窗口：
        各源对「同一个观测挂在哪一天」的约定不同。实测金属图表周频源
        用发布日（周五）标注，其数值等于日频源周四的业务日库存，
        两者相差一天。若只按完全相同的日期比对，这 24 个重叠观测
        会被全部误判为冲突，反而淹没了真正的异常。

    冲突判定因此收紧为：窗口内能找到另一来源的数据，但没有任何一天
    的数值对得上，才算冲突——说明确实存在无法解释的分歧。

    按来源两两配对，并从记录较少的一侧发起匹配，
    避免同一分歧被正反两个方向重复报告。

    返回 (验证通过的 (report_date, source) 集合, 冲突列表)。
    """

    by_source = {}

    for row in rows:
        if row.get("total") is None:
            continue

        try:
            parsed = datetime.strptime(
                str(row["report_date"])[:10], "%Y-%m-%d"
            ).date()
        except (ValueError, TypeError):
            continue

        by_source.setdefault(row["source"], {})[parsed] = float(row["total"])

    sources = sorted(by_source)

    passed = set()
    conflicts = []

    if len(sources) < 2:
        return passed, conflicts

    window = range(
        -CROSS_SOURCE_DATE_WINDOW_DAYS,
        CROSS_SOURCE_DATE_WINDOW_DAYS + 1,
    )

    for index, first in enumerate(sources):
        for second in sources[index + 1:]:
            # 从记录较少的一侧发起，保证每个分歧只报告一次
            if len(by_source[first]) <= len(by_source[second]):
                sparse, dense = first, second
            else:
                sparse, dense = second, first

            for target_date, value in sorted(by_source[sparse].items()):
                matched = []
                nearby = []

                for delta in window:
                    probe = target_date + timedelta(days=delta)

                    other_value = by_source[dense].get(probe)

                    if other_value is None:
                        continue

                    nearby.append((probe, other_value))

                    if abs(value - other_value) <= value * CROSS_SOURCE_TOLERANCE:
                        matched.append((probe, other_value))

                if matched:
                    passed.add((target_date.isoformat(), sparse))

                    for probe, _ in matched:
                        passed.add((probe.isoformat(), dense))

                elif nearby:
                    values = [value] + [item[1] for item in nearby]

                    spread = max(values) - min(values)
                    base = max(values)

                    totals = {sparse: value}
                    affected = {(target_date.isoformat(), sparse)}

                    for probe, other_value in nearby:
                        totals[f"{dense}@{probe.isoformat()}"] = other_value
                        affected.add((probe.isoformat(), dense))

                    conflicts.append(
                        {
                            "report_date": target_date.isoformat(),
                            "source": sparse,
                            "compared_source": dense,
                            "window_days": CROSS_SOURCE_DATE_WINDOW_DAYS,
                            "totals": totals,
                            # 精确到 (日期, 来源) 的受影响记录，
                            # 供核验分级只标记真正有分歧的行
                            "affected_keys": sorted(affected),
                            "spread_oz": round(spread, 3),
                            "spread_pct": (
                                round(spread / base * 100, 4) if base else None
                            ),
                        }
                    )

    return passed, conflicts


def flag_large_changes(rows):
    """按来源标记相邻记录的极端变化，仅提示不判错。"""

    warnings = []

    by_source = {}

    for row in rows:
        by_source.setdefault(row["source"], []).append(row)

    for source, group in by_source.items():
        ordered = sorted(
            [item for item in group if item.get("total")],
            key=lambda x: x["report_date"],
        )

        for previous, current in zip(ordered, ordered[1:]):
            old = float(previous["total"])

            if old <= 0:
                continue

            change = abs(float(current["total"]) - old) / old

            if change >= LARGE_CHANGE_THRESHOLD:
                warnings.append(
                    {
                        "source": source,
                        "report_date": current["report_date"],
                        "related_date": previous["report_date"],
                        "change_percent": round(change * 100, 4),
                    }
                )

    return warnings


# =========================================================
# 7. 数据库写入
# =========================================================

def ensure_schema(conn):
    """创建库存表与质量表，并为旧库补齐新增列。"""

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

    existing = {
        column[1]
        for column in conn.execute(
            "PRAGMA table_info(comex_inventory)"
        ).fetchall()
    }

    # 保留来源原始日期，使日期校正可审计、可回溯
    if "source_date" not in existing:
        conn.execute(
            "ALTER TABLE comex_inventory ADD COLUMN source_date TEXT"
        )
        logging.info("已为 comex_inventory 增加 source_date 列。")

    # SQLite 的 ALTER TABLE ADD COLUMN 不允许非常量默认值，
    # 因此 updated_at 只能建成可空列，不能带 CURRENT_TIMESTAMP 默认值。
    if "updated_at" not in existing:
        conn.execute(
            "ALTER TABLE comex_inventory ADD COLUMN updated_at TEXT"
        )
        logging.info("已为 comex_inventory 增加 updated_at 列。")

    conn.commit()


def save_rows(rows, dry_run=True):
    """
    幂等入库：按 (report_date, source) 去重。

    数值有实质变化时更新记录并把核验状态退回 REVIEW，
    避免「已核验」标签覆盖在被修改过的数据上。
    """

    if not rows:
        logging.warning("没有可写入的记录。")
        return {"inserted": 0, "updated": 0, "unchanged": 0}

    if dry_run:
        logging.info("预览模式：本次不写入数据库（候选 %s 条）。", len(rows))
        return {"inserted": 0, "updated": 0, "unchanged": 0}

    stats = Counter()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        ensure_schema(conn)

        for row in rows:
            key = (row["report_date"], row["source"])

            existing = conn.execute(
                "SELECT registered, eligible, total FROM comex_inventory "
                "WHERE report_date = ? AND source = ?",
                key,
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO comex_inventory (
                        report_date, registered, eligible, total,
                        source, source_date, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["report_date"],
                        row.get("registered"),
                        row.get("eligible"),
                        row.get("total"),
                        row["source"],
                        row.get("source_date"),
                        now,
                    ),
                )
                stats["inserted"] += 1
                continue

            changed = False

            for index, field in enumerate(
                ("registered", "eligible", "total")
            ):
                old = existing[index]
                new = row.get(field)

                if old is None and new is None:
                    continue

                if old is None or new is None:
                    changed = True
                    break

                if abs(float(old) - float(new)) > 0.01:
                    changed = True
                    break

            if not changed:
                stats["unchanged"] += 1
                continue

            conn.execute(
                """
                UPDATE comex_inventory
                SET registered = ?, eligible = ?, total = ?,
                    source_date = ?, updated_at = ?
                WHERE report_date = ? AND source = ?
                """,
                (
                    row.get("registered"),
                    row.get("eligible"),
                    row.get("total"),
                    row.get("source_date"),
                    now,
                    row["report_date"],
                    row["source"],
                ),
            )

            # 数值变了，旧核验结论不再适用，退回待复核
            conn.execute(
                """
                UPDATE comex_inventory_quality
                SET status = ?,
                    reason = '来源数值已更新，需重新核验',
                    reviewed_at = NULL,
                    updated_at = ?
                WHERE report_date = ? AND source = ?
                """,
                (
                    STATUS_REVIEW,
                    now,
                    row["report_date"],
                    row["source"],
                ),
            )

            stats["updated"] += 1

        conn.commit()

    logging.info(
        "入库完成：新增 %s，更新 %s，无变化 %s。",
        stats["inserted"], stats["updated"], stats["unchanged"],
    )

    # dict(Counter) 会丢掉计数为 0 的键，导致调用方取值 KeyError，
    # 因此显式构造完整的返回结构。
    return {
        "inserted": stats["inserted"],
        "updated": stats["updated"],
        "unchanged": stats["unchanged"],
    }


# =========================================================
# 8. 核验状态维护
# =========================================================

def apply_quality_status(
    rows,
    passed_pairs,
    conflicts,
    large_changes=None,
    dry_run=True,
):
    """
    写入四级核验状态。

    VERIFIED  与另一独立来源在同一 CME 业务日的总库存一致（相对偏差 <= 0.01%）
    ACCEPTED  单源数据，通过全部自动校验且无任何告警
    REVIEW    存在告警：跨源冲突、极端变化、日期落在周末或校验未通过
    REJECTED  旧来源数据，日期口径与字段均已证伪（由 reject_legacy_source 处理）

    为什么需要 ACCEPTED：
        免费可得且带 registered/eligible 拆分的源只有 metalcharts 周频 CSV，
        与日频源的重叠区间只有 31 个业务日。若要求「必须有第二来源交叉验证」，
        日频源的上千条记录将永远无法使用，库存因子会长期缺数据。
        因此对单源数据采用「通过全部自动校验即接受」的次级信任，
        并把可信度差异如实记录在 status 与 reason 中，由使用方决定是否收紧。
    """

    if dry_run:
        logging.info("预览模式：不写入核验状态。")
        return Counter()

    # 精确到 (日期, 来源) 的冲突键；无 affected_keys 时退回按日期匹配，
    # 保证人工传入的简单冲突结构仍然有效
    conflict_keys = set()
    conflict_dates = set()

    for item in conflicts:
        affected = item.get("affected_keys")

        if affected:
            conflict_keys.update(tuple(key) for key in affected)
        else:
            conflict_dates.add(item["report_date"])

    large_change_dates = {
        item["report_date"] for item in (large_changes or [])
    }

    stats = Counter()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    records = []

    # 先在内存中判定每行状态，再单连接批量写入，避免逐行开关连接
    for row in rows:
        errors, flags = validate_row(row)

        key = (row["report_date"], row["source"])

        alerts = list(flags)

        # 冲突告警按精确键匹配，与下方状态判定保持一致
        if key in conflict_keys or row["report_date"] in conflict_dates:
            alerts.append("与另一来源在同一 CME 业务日附近的总库存不一致")

        if row["report_date"] in large_change_dates:
            alerts.append("相邻记录变化幅度达到或超过 10%")

        if errors:
            status = STATUS_REVIEW
            reason = "校验未通过：" + "；".join(errors)
        elif key in conflict_keys or row["report_date"] in conflict_dates:
            status = STATUS_REVIEW
            reason = "与另一来源在同一 CME 业务日附近的总库存不一致"
        elif alerts:
            status = STATUS_REVIEW
            reason = "存在待复核告警：" + "；".join(alerts)
        elif key in passed_pairs:
            status = STATUS_VERIFIED
            reason = (
                "已通过基础校验，且与另一独立来源在同一 CME 业务日"
                "的总库存数值一致（相对偏差 <= 0.01%）"
            )
        else:
            status = STATUS_ACCEPTED
            reason = (
                "单源数据，已通过全部自动校验（日期合法、数值为正、"
                "字段自洽、无极端变化告警）；暂无第二来源可交叉比对"
            )

        records.append(
            (
                row["report_date"],
                row["source"],
                row.get("source_url"),
                status,
                reason,
                json.dumps(alerts, ensure_ascii=False) if alerts else None,
                now if status in (STATUS_VERIFIED, STATUS_ACCEPTED) else None,
                now,
            )
        )

        stats[status] += 1

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        ensure_schema(conn)

        conn.executemany(
            """
            INSERT INTO comex_inventory_quality (
                report_date, source, source_url, status,
                reason, flags_json, reviewed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_date, source) DO UPDATE SET
                source_url = excluded.source_url,
                status = excluded.status,
                reason = excluded.reason,
                flags_json = excluded.flags_json,
                reviewed_at = excluded.reviewed_at,
                updated_at = excluded.updated_at
            """,
            records,
        )

        conn.commit()

    logging.info(
        "核验状态写入完成：VERIFIED=%s，ACCEPTED=%s，REVIEW=%s。",
        stats[STATUS_VERIFIED], stats[STATUS_ACCEPTED], stats[STATUS_REVIEW],
    )

    return stats


def reject_legacy_source(source=SOURCE_LEGACY, dry_run=True, reason=None):
    """
    把旧来源标记为 REJECTED，保留数据用于审计但不参与分析。

    不做物理删除：历史记录的错误本身就是重要证据。
    """

    default_reason = (
        "旧来源（搜索文章 + 正则解析）已被证伪：日期标签比 CME 业务日早 1 天，"
        "且同一行内 registered / eligible / total 存在跨日期拼接错误。"
        "数据保留以便审计，不参与任何分析与打分。"
    )

    if dry_run:
        # 预览模式不得改动数据库结构，因此不调用 ensure_schema
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='comex_inventory'"
            ).fetchone()

            count = 0

            if table_exists:
                count = conn.execute(
                    "SELECT COUNT(*) FROM comex_inventory WHERE source = ?",
                    (source,),
                ).fetchone()[0]

        logging.info(
            "预览模式：将把 %s 条 %s 记录标记为 REJECTED（不删除）。",
            count, source,
        )
        return count

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        ensure_schema(conn)

        existing = {
            (row[0], row[1])
            for row in conn.execute(
                "SELECT report_date, source FROM comex_inventory "
                "WHERE source = ?",
                (source,),
            ).fetchall()
        }

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for report_date, row_source in existing:
            conn.execute(
                """
                INSERT INTO comex_inventory_quality (
                    report_date, source, source_url, status,
                    reason, flags_json, reviewed_at, updated_at
                ) VALUES (?, ?, NULL, 'REJECTED', ?, NULL, NULL, ?)
                ON CONFLICT(report_date, source) DO UPDATE SET
                    status = 'REJECTED',
                    reason = excluded.reason,
                    reviewed_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (report_date, row_source, reason or default_reason, now),
            )

        conn.commit()

    logging.info("已将 %s 条 %s 记录标记为 REJECTED。", len(existing), source)

    return len(existing)


# =========================================================
# 9. 证据留档
# =========================================================

def save_evidence(payload, name):
    """
    把本轮采集与验证结果落盘，便于事后追溯。

    证据文件只是附属产物：写盘失败时记录警告并继续，
    绝不能让一次成功的采集中断。
    """

    try:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

        path = EVIDENCE_DIR / name

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        logging.warning(
            "证据保存失败（不影响本次采集）：%s -> %s", name, exc
        )
        return None

    logging.info("证据已保存：%s", path)

    return path


# =========================================================
# 10. 主流程
# =========================================================

def run(dry_run=True, sources=("eastmoney", "metalcharts")):
    """执行一次完整采集：抓取 -> 自检 -> 校验 -> 交叉验证 -> 入库。"""

    logging.info("=" * 62)
    logging.info("COMEX 黄金库存采集（结构化接口）")
    logging.info("模式：%s", "预览 DRY_RUN" if dry_run else "正式入库")
    logging.info("数据源：%s", ", ".join(sources))
    logging.info("=" * 62)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"数据库不存在：{DB_PATH}\n"
            "请先运行 python database/db.py 初始化数据库。"
        )

    all_rows = []

    eastmoney_rows = []
    metalcharts_rows = []

    if SOURCE_EASTMONEY in sources:
        eastmoney_rows = fetch_eastmoney_gold_inventory()
        all_rows.extend(eastmoney_rows)

    if SOURCE_METALCHARTS in sources:
        metalcharts_rows = fetch_metalcharts_weekly("gold")
        all_rows.extend(metalcharts_rows)

    if not all_rows:
        logging.error("没有取得任何数据，终止。")
        return {"rows": 0}

    # --- 日期偏移自检 ---
    offset_report = {}

    if eastmoney_rows and metalcharts_rows:
        detected, detail = detect_date_offset(
            eastmoney_rows, metalcharts_rows
        )

        expected = DATE_OFFSET_DAYS[SOURCE_EASTMONEY]

        offset_report = {
            "expected_offset_days": expected,
            "detected_offset_days": detected,
            "match_counts_by_offset": detail.get("match_counts_by_offset"),
            "weekend_counts_by_offset": detail.get("weekend_counts_by_offset"),
            "changed_days_used": detail.get("changed_days_used"),
            "valid_offsets": detail.get("valid_offsets"),
            "reason": detail.get("reason"),
            "consistent": detected == expected,
        }

        if detected is None:
            logging.error(
                "日期偏移无法判定：%s。已中止入库，请人工确认上游口径。",
                detail.get("reason"),
            )

            save_evidence(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "stage": "offset_check_failed",
                    "offset_report": offset_report,
                },
                "offset_check_failed.json",
            )

            return {
                "rows": len(all_rows),
                "offset_report": offset_report,
                "aborted": "date_offset_undetermined",
            }

        if detected != expected:
            logging.error(
                "日期偏移自检不一致！预设 %+d 天，实测最优 %+d 天。"
                "已中止入库，请人工确认上游口径。详情：%s",
                expected, detected, detail.get("reason"),
            )

            save_evidence(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "stage": "offset_check_failed",
                    "offset_report": offset_report,
                },
                "offset_check_failed.json",
            )

            return {
                "rows": len(all_rows),
                "offset_report": offset_report,
                "aborted": "date_offset_mismatch",
            }

        logging.info(
            "日期偏移自检通过：东方财富 %+d 天（%s）",
            detected, detail.get("reason"),
        )

    # --- 业务日硬校验：CME 业务日绝不可能是周末 ---
    # 这道闸门独立于偏移检测，任何原因导致的日期错位都会在此被拦下，
    # 避免把「周六库存」这种不可能的数据静默写进数据库。
    weekend_rows = [
        item
        for item in all_rows
        if _is_weekend_date(item.get("report_date"))
    ]

    if weekend_rows:
        sample = weekend_rows[:5]

        logging.error(
            "业务日校验失败：有 %s 条记录的校正后日期落在周末，"
            "CME 业务日只可能是周一至周五，说明日期口径有误。"
            "已中止入库。样本：%s",
            len(weekend_rows),
            [
                (item["source"], item["source_date"], item["report_date"])
                for item in sample
            ],
        )

        save_evidence(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "stage": "business_day_check_failed",
                "weekend_count": len(weekend_rows),
                "offset_report": offset_report,
                "samples": [
                    {
                        "source": item["source"],
                        "source_date": item["source_date"],
                        "report_date": item["report_date"],
                        "total": item.get("total"),
                    }
                    for item in weekend_rows[:50]
                ],
            },
            "business_day_check_failed.json",
        )

        return {
            "rows": len(all_rows),
            "offset_report": offset_report,
            "aborted": "weekend_business_day",
            "weekend_count": len(weekend_rows),
        }

    logging.info(
        "业务日校验通过：%s 条记录的校正后日期全部落在周一至周五。",
        len(all_rows),
    )

    # --- 单条校验 ---
    valid_rows = []
    invalid_rows = []

    for row in all_rows:
        errors, flags = validate_row(row)

        if errors:
            invalid_rows.append({**row, "errors": errors})
            continue

        valid_rows.append(row)

    logging.info(
        "基础校验：通过 %s 条，未通过 %s 条。",
        len(valid_rows), len(invalid_rows),
    )

    # --- 跨源交叉验证 ---
    passed_pairs, conflicts = cross_validate(valid_rows)

    logging.info(
        "跨源交叉验证：一致 %s 组，冲突 %s 组。",
        len(passed_pairs), len(conflicts),
    )

    for conflict in conflicts:
        logging.warning("跨源数值冲突：%s", conflict)

    # --- 极端变化提示 ---
    large_changes = flag_large_changes(valid_rows)

    for item in large_changes:
        logging.warning("相邻记录变化超过 10%%，需复核：%s", item)

    # --- 去重（同一 report_date+source 只保留一条）---
    unique_rows = {}

    for row in valid_rows:
        unique_rows[(row["report_date"], row["source"])] = row

    unique_rows = list(unique_rows.values())

    # --- 证据留档 ---
    save_evidence(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": dry_run,
            "sources": list(sources),
            "offset_report": offset_report,
            "counts": {
                "fetched": len(all_rows),
                "eastmoney": len(eastmoney_rows),
                "metalcharts": len(metalcharts_rows),
                "valid": len(valid_rows),
                "invalid": len(invalid_rows),
                "unique": len(unique_rows),
                "cross_validated_pairs": len(passed_pairs),
                "conflicts": len(conflicts),
                "large_change_warnings": len(large_changes),
            },
            "date_range": {
                "min": min(r["report_date"] for r in unique_rows),
                "max": max(r["report_date"] for r in unique_rows),
            } if unique_rows else None,
            "conflicts": conflicts,
            "large_changes": large_changes,
            "invalid_rows": invalid_rows[:50],
        },
        "ingest_evidence.json",
    )

    # --- 作废旧来源 ---
    rejected = reject_legacy_source(dry_run=dry_run)

    # --- 入库 ---
    stats = save_rows(unique_rows, dry_run=dry_run)

    quality_stats = apply_quality_status(
        unique_rows,
        passed_pairs,
        conflicts,
        large_changes=large_changes,
        dry_run=dry_run,
    )

    summary = {
        "rows": len(unique_rows),
        "stats": stats,
        "quality": dict(quality_stats),
        "rejected_legacy": rejected,
        "offset_report": offset_report,
        "conflicts": len(conflicts),
        "large_changes": len(large_changes),
    }

    logging.info("采集汇总：%s", json.dumps(summary, ensure_ascii=False, default=str))

    return summary


def main(argv=None):
    """命令行入口。"""

    argv = list(sys.argv[1:] if argv is None else argv)

    dry_run = "--commit" not in argv

    sources = ("eastmoney", "metalcharts")

    if "--only-eastmoney" in argv:
        sources = (SOURCE_EASTMONEY,)
    elif "--only-metalcharts" in argv:
        sources = (SOURCE_METALCHARTS,)

    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print(
            "\n用法：\n"
            "  python -m src.data_sources.comex_inventory_api           # 预览，不写库\n"
            "  python -m src.data_sources.comex_inventory_api --commit   # 正式入库\n"
            "  --only-eastmoney / --only-metalcharts                     # 只用单一来源\n"
        )
        return 0

    result = run(dry_run=dry_run, sources=sources)

    if result.get("aborted"):
        logging.error("本次采集已中止：%s", result["aborted"])
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

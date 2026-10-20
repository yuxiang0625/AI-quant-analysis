import os
import re
import json
import sqlite3
import logging
import shutil
import time

from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# =========================================================
# 1. 项目路径与基本配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "aurum.db"
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "comex_ingest.log"

SEARCH_RESULT_PATH = DATA_DIR / "comex_search" / "search_results.json"
PARSED_RESULT_PATH = DATA_DIR / "comex_search" / "parsed_preview.json"
AUDIT_REPORT_PATH = DATA_DIR / "comex_search" / "quality_audit.json"
RAW_ARCHIVE_DIR = DATA_DIR / "comex_search" / "raw_pages"

SOURCE = "cngold"

# 第一次运行必须保持 True：
# True  = 只搜索、解析、校验、预览，不写入数据库
# False = 允许将通过校验的数据写入数据库
DRY_RUN = True

# 每次搜索返回多少条网页
SEARCH_COUNT = 10

# 每次最多处理多少个不同的文章链接
MAX_ARTICLES = 10

# 每次运行最多调用多少次 Fetch API
MAX_FETCH_CALLS = 10

# DeepSeek API
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SEARCH_API_URL = f"{DEEPSEEK_BASE_URL}/v1/tools/search"
FETCH_API_URL = f"{DEEPSEEK_BASE_URL}/v1/tools/fetch"

# 旧文章作为备用来源。
# 保留这些 URL，不删除原有数据来源。
SEED_URLS = [
    "https://www.cngold.org/c/2026-10-16/c10819703.html",
    "https://m.cngold.org/home/xw10816848.html",
    "https://m.cngold.org/home/xw10819703.html",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RAW_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 日志设置
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


# =========================================================
# 3. 读取 DeepSeek API Key
# =========================================================

def load_api_key():
    """
    优先读取系统环境变量。
    如果没有，再读取项目根目录下的 .env 文件。
    不需要额外安装 python-dotenv。
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if api_key:
        return api_key.strip()

    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        return None

    try:
        for line in env_path.read_text(
            encoding="utf-8-sig"
        ).splitlines():

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[7:].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            if key.strip() == "DEEPSEEK_API_KEY":
                return value.strip().strip('"').strip("'")

    except OSError as exc:
        logging.warning("读取 .env 失败：%s", exc)

    return None


# =========================================================
# 4. DeepSeek 网页搜索
# =========================================================

def deepseek_search(query, api_key, count=10):
    """
    调用 DeepSeek Search API。
    返回结构化网页搜索结果。
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "count": count,
        "need_content": True,
    }

    try:
        response = requests.post(
            SEARCH_API_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )

        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        logging.info(
            "DeepSeek 搜索完成：查询=%s，结果=%s 条",
            query,
            len(results),
        )

        return results

    except requests.RequestException as exc:
        logging.error("DeepSeek 搜索失败：%s", exc)
        return []

    except ValueError as exc:
        logging.error("搜索接口返回的 JSON 无法解析：%s", exc)
        return []


# =========================================================
# 5. DeepSeek 网页内容获取
# =========================================================

def deepseek_fetch(url, api_key):
    """
    调用 DeepSeek Fetch API 获取网页正文。
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            FETCH_API_URL,
            headers=headers,
            json={"url": url},
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        content = data.get("content", "")

        if not content:
            logging.warning("Fetch 未返回正文：%s", url)
            return None

        return {
            "url": url,
            "title": data.get("title", ""),
            "content": content,
        }

    except requests.RequestException as exc:
        logging.warning("DeepSeek Fetch 失败：%s；原因：%s", url, exc)
        return None

    except ValueError as exc:
        logging.warning("Fetch 返回内容无法解析：%s", exc)
        return None


# =========================================================
# 6. 普通网页备用获取方式
# =========================================================

def fetch_page_directly(url):
    """
    当 DeepSeek Fetch 不可用时，尝试普通 HTTP 获取。
    如果目标网站拒绝访问，不绕过网站限制。
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )

        response.raise_for_status()
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "lxml")

        return {
            "url": url,
            "title": (
                soup.title.get_text(" ", strip=True)
                if soup.title
                else ""
            ),
            "content": soup.get_text("\n", strip=True),
        }

    except requests.RequestException as exc:
        logging.warning(
            "普通网页获取失败：%s；原因：%s",
            url,
            exc,
        )
        return None


# =========================================================
# 7. 搜索 COMEX 黄金库存文章
# =========================================================

def discover_articles(api_key):
    """
    通过 DeepSeek 搜索发现公开的 COMEX 黄金库存文章。

    同时保留原来的备用文章链接。
    """

    queries = [
        'COMEX黄金库存 注册仓单 非注册仓单 总量 盎司 site:cngold.org',
        'COMEX黄金库存量数据 注册仓单 非注册仓单 site:m.cngold.org',
    ]

    all_results = []
    seen_urls = set()

    # 先搜索互联网
    for query in queries:

        results = deepseek_search(
            query=query,
            api_key=api_key,
            count=SEARCH_COUNT,
        )

        for item in results:

            url = item.get("url", "").strip()

            if not url:
                continue

            parsed = urlparse(url)

            if parsed.scheme not in ("http", "https"):
                continue

            if url not in seen_urls:
                seen_urls.add(url)

                item["discovery_method"] = "deepseek_search"

                all_results.append(item)

        # 避免连续请求过快
        time.sleep(0.3)

    # 将原有链接作为备用文章来源
    for url in SEED_URLS:

        if url not in seen_urls:

            seen_urls.add(url)

            all_results.append({
                "title": "原有备用库存文章",
                "url": url,
                "snippet": "",
                "summary": "",
                "content": "",
                "discovery_method": "seed_url",
            })

    # 保存原始搜索结果，便于追踪来源
    try:
        SEARCH_RESULT_PATH.write_text(
            json.dumps(
                all_results,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logging.info(
            "搜索结果已保存：%s",
            SEARCH_RESULT_PATH,
        )

    except OSError as exc:
        logging.warning("保存搜索结果失败：%s", exc)

    # 优先处理搜索结果，再处理备用链接
    return all_results[:MAX_ARTICLES]


# =========================================================
# 7.5 原始网页留档
# =========================================================

def archive_raw_page(url, title, content):
    """保存网页正文快照，便于以后复查解析结果；文件名不包含敏感信息。"""
    import hashlib

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    payload = {
        "source_url": url,
        "title": title or "",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "content": content,
    }
    path = RAW_ARCHIVE_DIR / f"{digest}.json"
    try:
        # 每次保留最近一次快照；历史采集运行日志仍然保留。
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("原始网页快照已保存：%s", path)
    except OSError as exc:
        logging.warning("原始网页快照保存失败：%s；原因：%s", url, exc)


# =========================================================
# 8. 识别日期与库存数值
# =========================================================

DATE_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?(?!\d)"
)

NUMBER_PATTERN = re.compile(
    r"(?<![\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d.])"
)


def normalize_date(match):
    """
    将 2026-10-14、2026/10/14、2026年10月14日
    统一转换为 YYYY-MM-DD。
    """

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))

    try:
        value = datetime(year, month, day)
        return value.strftime("%Y-%m-%d")

    except ValueError:
        return None


def to_number(value):
    return float(value.replace(",", ""))


def extract_labeled_value(text, label_pattern):
    """
    在当前日期区块内按完整标签提取数值。
    特别避免“注册仓单”误匹配“非注册仓单”。
    """
    number = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    pattern = re.compile(
        r"(?<!非)" + label_pattern
        + r"(?:\s*仓单)?"
        + r"[^\d\n]{0,40}"
        + number,
        re.IGNORECASE,
    )

    match = pattern.search(text)
    if not match:
        return None

    try:
        return to_number(match.group(1))
    except (ValueError, IndexError):
        return None


def validate_inventory(row):
    """基础质量检查；不代表来源真实性已经核实。"""
    errors = []

    try:
        parsed = datetime.strptime(
            str(row.get("report_date", ""))[:10], "%Y-%m-%d"
        ).date()
        if parsed > datetime.now().date():
            errors.append("报告日期晚于当前日期")
    except (ValueError, TypeError):
        errors.append("报告日期格式错误")

    values = {}
    for key in ("registered", "eligible", "total"):
        try:
            value = float(row.get(key))
            if not (value > 0 and value < float("inf")):
                errors.append(f"{key} 必须是正的有限数值")
            values[key] = value
        except (TypeError, ValueError):
            errors.append(f"{key} 缺失或不是有效数值")

    if len(values) == 3 and not errors:
        if abs(values["registered"] + values["eligible"] - values["total"]) > 0.01:
            errors.append("注册库存 + 非注册库存与总库存不一致")

    if not row.get("source_url"):
        errors.append("缺少原始来源网址")

    return errors


# =========================================================
# 9. 解析库存记录
# =========================================================

def parse_inventory(content, source_url):
    """
    保守解析 COMEX 黄金库存：
    - 只使用明确日期区块；
    - 注册仓单标签不能匹配到“非注册仓单”；
    - 每个字段必须在同一日期区块内找到；
    - 库存合计不一致的记录直接跳过；
    - 不根据文章发布日期推断报告日期。
    """
    if not content:
        return []

    soup = BeautifulSoup(content, "lxml")
    text = soup.get_text("\n", strip=True).replace("\r", "\n")
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
        if line.strip()
    ]

    # 只把包含完整年月日的行视为报告日期候选行。
    # 日期可以位于表格行内，但该行或相邻上下文必须与库存数据有关，
    # 避免把网页页脚、文章发布时间等日期误当成报告日期。
    date_line_items = []
    for idx, line in enumerate(lines):
        matches = list(DATE_PATTERN.finditer(line))
        if not matches:
            continue

        report_date = normalize_date(matches[0])
        if not report_date:
            continue

        context = "\n".join(lines[idx:min(idx + 8, len(lines))])
        if not re.search(
            r"注册仓单|非注册仓单|总量|库存量|comex|COMEX",
            context,
            re.IGNORECASE,
        ):
            continue

        date_line_items.append((idx, report_date))

    if not date_line_items:
        logging.warning("未发现明确的库存日期区块：%s", source_url)
        return []

    results = []
    seen_dates = set()

    for i, (line_idx, report_date) in enumerate(date_line_items):
        next_idx = (
            date_line_items[i + 1][0]
            if i + 1 < len(date_line_items)
            else len(lines)
        )
        block = "\n".join(lines[line_idx:next_idx])

        # 防止一块中重复出现日期时误取下一条记录。
        registered = extract_labeled_value(block, r"注册仓单")
        eligible = extract_labeled_value(block, r"非注册仓单")
        total = extract_labeled_value(block, r"总量")

        if None in (registered, eligible, total):
            logging.info(
                "字段不完整，跳过：日期=%s，来源=%s",
                report_date, source_url
            )
            continue

        row = {
            "report_date": report_date,
            "registered": registered,
            "eligible": eligible,
            "total": total,
            "source": SOURCE,
            "source_url": source_url,
        }

        errors = validate_inventory(row)
        if errors:
            logging.warning(
                "数据校验失败，跳过：日期=%s，错误=%s，来源=%s",
                report_date, "; ".join(errors), source_url
            )
            continue

        if report_date in seen_dates:
            logging.warning(
                "同一篇文章出现重复日期，保留第一条：%s，来源=%s",
                report_date, source_url
            )
            continue

        seen_dates.add(report_date)
        results.append(row)

    return results


# =========================================================
# 10. 兼容已有数据库
# =========================================================

def prepare_database(conn):
    """
    保留原有表结构与历史记录。
    不删除表，不重建数据库。
    """

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

    conn.commit()

    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(comex_inventory)"
        ).fetchall()
    }

    required_columns = {
        "report_date",
        "registered",
        "eligible",
        "total",
        "source",
    }

    missing = required_columns - columns

    if missing:
        raise RuntimeError(
            f"现有 comex_inventory 表缺少字段：{missing}"
        )


# =========================================================
# 11. 检查已有记录
# =========================================================

def check_existing_rows(rows):
    """比较候选数据与数据库；返回存在冲突的 (日期, 来源) 键。"""
    conflicts = set()
    if not rows:
        return conflicts

    with sqlite3.connect(DB_PATH) as conn:
        prepare_database(conn)
        for row in rows:
            existing = conn.execute("""
                SELECT registered, eligible, total
                FROM comex_inventory
                WHERE report_date = ? AND source = ?
            """, (row["report_date"], row["source"])).fetchone()
            if not existing:
                continue

            new_values = (row["registered"], row["eligible"], row["total"])
            if any(value is None for value in existing):
                conflicts.add((row["report_date"], row["source"]))
                logging.error("已有记录存在空值，禁止自动覆盖/跳过候选：%s", row["report_date"])
                continue

            if any(abs(float(old) - float(new)) > 0.01 for old, new in zip(existing, new_values)):
                conflicts.add((row["report_date"], row["source"]))
                logging.error(
                    "数据库与新来源数值冲突，候选数据进入复核，不自动入库：日期=%s；旧值=%s；新值=%s；来源=%s",
                    row["report_date"], existing, new_values, row.get("source_url")
                )
    return conflicts


# =========================================================
# 11.5 候选数据质量报告
# =========================================================

def audit_candidates(rows):
    """为本次采集候选生成审核报告；异常仅标记，不自行删除数据。"""
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_count": len(rows),
        "accepted_for_review": [],
        "issues": [],
        "note": "本报告是自动筛查，不等于来源真实性认证；极端变化需对照原文。",
    }
    ordered = sorted(rows, key=lambda r: r["report_date"])
    seen_dates = {}
    previous = None

    for row in ordered:
        key = row["report_date"]
        vals = (row["registered"], row["eligible"], row["total"])
        errors = validate_inventory(row)
        if key in seen_dates:
            prior = seen_dates[key]
            prior_vals = (prior["registered"], prior["eligible"], prior["total"])
            if any(abs(float(a) - float(b)) > 0.01 for a, b in zip(vals, prior_vals)):
                report["issues"].append({
                    "type": "same_date_conflict", "report_date": key,
                    "sources": [prior.get("source_url"), row.get("source_url")],
                    "values": [prior_vals, vals],
                    "issues": ["同一报告日期存在不同库存数值，禁止自动选择其中一条"],
                })
            else:
                report["issues"].append({
                    "type": "same_date_duplicate", "report_date": key,
                    "sources": [prior.get("source_url"), row.get("source_url")],
                    "issues": ["同一日期重复出现；保留一条候选，但保留来源记录以供追溯"],
                })
        else:
            seen_dates[key] = row

        if errors:
            report["issues"].append({
                "type": "validation_error", "report_date": key,
                "source_url": row.get("source_url"), "issues": errors,
            })

        if previous:
            for field in ("registered", "eligible", "total"):
                old = float(previous[field])
                current = float(row[field])
                if old > 0 and abs(current - old) / old >= 0.10:
                    report["issues"].append({
                        "type": "large_change", "report_date": key,
                        "related_date": previous["report_date"],
                        "field": field, "previous_value": old,
                        "current_value": current,
                        "change_percent": round((current - old) / old * 100, 4),
                        "source_url": row.get("source_url"),
                        "issues": ["与上一条候选记录相比变化达到或超过10%；可能是真实变化，也可能是解析/来源问题，需人工复核"],
                    })
        previous = row
        report["accepted_for_review"].append({
            "report_date": row["report_date"],
            "registered": row["registered"],
            "eligible": row["eligible"],
            "total": row["total"],
            "source_url": row.get("source_url"),
            "basic_validation": "passed" if not errors else "failed",
        })

    path = DATA_DIR / "comex_search" / "candidate_quality_audit.json"
    try:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("本次候选数据质量报告已保存：%s", path)
    except OSError as exc:
        logging.error("候选质量报告保存失败：%s", exc)

    logging.info("候选质量检查完成：候选=%s，待核查项目=%s", len(rows), len(report["issues"]))
    for issue in report["issues"]:
        logging.warning("候选数据需要复核：%s", issue)
    return report


# =========================================================
# 12. 写入数据库
# =========================================================

def save_inventory(rows):
    """
    使用 report_date + source 唯一约束去重。
    不覆盖已经存在的记录。
    """

    if not rows:
        logging.warning("没有有效库存数据，不写入数据库。")
        return 0

    if DRY_RUN:
        logging.info(
            "当前为预览模式 DRY_RUN=True，本次不会写入数据库。"
        )
        return 0

    inserted = 0

    with sqlite3.connect(DB_PATH) as conn:

        prepare_database(conn)

        for row in rows:

            cursor = conn.execute("""
                INSERT OR IGNORE INTO comex_inventory (
                    report_date,
                    registered,
                    eligible,
                    total,
                    source
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                row["report_date"],
                row["registered"],
                row["eligible"],
                row["total"],
                row["source"],
            ))

            if cursor.rowcount > 0:

                inserted += 1

                logging.info(
                    "新增库存：%s，注册=%.3f，"
                    "非注册=%.3f，总量=%.3f",
                    row["report_date"],
                    row["registered"],
                    row["eligible"],
                    row["total"],
                )

            else:

                logging.info(
                    "记录已存在，跳过：%s（%s）",
                    row["report_date"],
                    row["source"],
                )

        conn.commit()

    return inserted


# =========================================================
# 13. 保存解析预览
# =========================================================

def save_preview(rows):
    """
    将解析结果保存为 JSON。
    每条数据保留来源 URL，便于人工核验。
    """

    try:
        PARSED_RESULT_PATH.write_text(
            json.dumps(
                rows,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logging.info(
            "解析预览已保存：%s",
            PARSED_RESULT_PATH,
        )

    except OSError as exc:
        logging.error("保存解析预览失败：%s", exc)


def audit_database():
    """审计历史记录并生成机器可读报告；只读现有记录，不修正或删除。"""
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "database": str(DB_PATH),
        "record_count": 0,
        "issues": [],
        "notes": [
            "异常变化只代表需要核查，不自动认定为错误。",
            "数据通过加总检查不代表日期和来源已被独立证实。",
        ],
    }
    if not DB_PATH.exists():
        logging.warning("数据库不存在，跳过历史审计：%s", DB_PATH)
        return report

    with sqlite3.connect(DB_PATH) as conn:
        # 这里只确认表存在及结构，不更新历史记录。
        prepare_database(conn)
        rows = conn.execute("""
            SELECT report_date, registered, eligible, total, source
            FROM comex_inventory ORDER BY report_date ASC
        """).fetchall()

    report["record_count"] = len(rows)
    logging.info("开始审计数据库现有记录：%s 条", len(rows))
    valid_rows = []

    for report_date, registered, eligible, total, source in rows:
        issues = []
        try:
            parsed_date = datetime.strptime(str(report_date)[:10], "%Y-%m-%d").date()
            if parsed_date > datetime.now().date():
                issues.append("报告日期晚于当前日期")
        except (ValueError, TypeError):
            parsed_date = None
            issues.append("日期格式错误")

        try:
            vals = [float(registered), float(eligible), float(total)]
            if any(not (v > 0 and v < float("inf")) for v in vals):
                issues.append("库存值不是正的有限数值")
            if abs(vals[0] + vals[1] - vals[2]) > 0.01:
                issues.append("注册库存 + 非注册库存与总库存不一致")
            if parsed_date:
                valid_rows.append({
                    "report_date": parsed_date.isoformat(),
                    "registered": vals[0], "eligible": vals[1], "total": vals[2],
                    "source": source,
                })
        except (ValueError, TypeError):
            issues.append("库存值缺失或不是有效数值")

        if issues:
            report["issues"].append({
                "type": "invalid_record", "report_date": str(report_date),
                "source": source, "issues": issues,
            })

    # 检查同一来源相邻记录的重复数值，以及极端变动。
    valid_rows.sort(key=lambda r: r["report_date"])
    previous_by_source = {}
    for row in valid_rows:
        prior = previous_by_source.get(row["source"])
        if prior:
            if all(abs(row[k] - prior[k]) <= 0.001 for k in ("registered", "eligible", "total")):
                report["issues"].append({
                    "type": "repeated_values", "report_date": row["report_date"],
                    "source": row["source"], "related_date": prior["report_date"],
                    "issues": ["相邻有效记录库存值完全相同，需要对照原始来源核查"],
                })
            for key in ("registered", "eligible", "total"):
                old = prior[key]
                if old > 0 and abs(row[key] - old) / old >= 0.10:
                    report["issues"].append({
                        "type": "large_change", "report_date": row["report_date"],
                        "source": row["source"], "related_date": prior["report_date"],
                        "field": key, "previous_value": old, "current_value": row[key],
                        "change_percent": round((row[key] - old) / old * 100, 4),
                        "issues": ["相邻记录变化幅度达到或超过10%，需要人工核对；不自动判错"],
                    })
        previous_by_source[row["source"]] = row

    try:
        AUDIT_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("历史数据质量报告已保存：%s", AUDIT_REPORT_PATH)
    except OSError as exc:
        logging.error("保存历史审计报告失败：%s", exc)

    logging.info("历史审计完成：发现 %s 个待核查项目；没有自动修改或删除旧记录。", len(report["issues"]))
    for issue in report["issues"]:
        logging.warning("历史数据质量提示：%s", issue)
    return report


# =========================================================
# 14. 主程序
# =========================================================

def main():

    logging.info("=" * 60)
    logging.info("COMEX 黄金库存采集启动")
    logging.info("项目目录：%s", PROJECT_ROOT)
    logging.info("数据库：%s", DB_PATH)
    logging.info("预览模式：%s", DRY_RUN)

    if not DB_PATH.exists():

        raise FileNotFoundError(
            f"数据库不存在：{DB_PATH}\n"
            "为避免误建空数据库，程序已停止。"
        )

    api_key = load_api_key()

    if not api_key:

        raise RuntimeError(
            "没有读取到 DEEPSEEK_API_KEY。\n"
            "请检查项目根目录下的 .env 文件。"
        )

    # 运行前确认数据库结构正常
    with sqlite3.connect(DB_PATH) as conn:
        prepare_database(conn)

    audit_database()

    # -----------------------------------------------------
    # A. 搜索文章
    # -----------------------------------------------------

    articles = discover_articles(api_key)

    logging.info(
        "本次准备处理 %s 个文章链接。",
        len(articles),
    )

    # -----------------------------------------------------
    # B. 获取正文并解析
    # -----------------------------------------------------

    all_rows = []
    seen_records = set()
    conflicted_records = set()
    fetch_calls = 0

    for article in articles:

        url = article.get("url", "").strip()

        if not url:
            continue

        logging.info("正在处理文章：%s", url)

        content = article.get("content", "")

        # Search API 已返回正文时优先使用
        if not content:

            if fetch_calls < MAX_FETCH_CALLS:

                page = deepseek_fetch(
                    url=url,
                    api_key=api_key,
                )

                fetch_calls += 1

                if page:
                    content = page.get("content", "")

                time.sleep(0.3)

        # DeepSeek Fetch 没有获取到正文时再尝试普通 HTTP
        if not content:

            page = fetch_page_directly(url)

            if page:
                content = page.get("content", "")

        if not content:
            logging.warning("无法取得正文，跳过：%s", url)
            continue

        archive_raw_page(url, article.get("title", ""), content)

        rows = parse_inventory(
            content=content,
            source_url=url,
        )

        logging.info(
            "本篇文章解析到 %s 条有效记录。",
            len(rows),
        )

        for row in rows:
            row_errors = validate_inventory(row)
            if row_errors:
                logging.warning(
                    "最终入库前校验未通过，跳过：日期=%s，错误=%s",
                    row.get("report_date"),
                    "; ".join(row_errors),
                )
                continue

            key = (row["report_date"], row["source"])
            existing_in_run = next(
                (
                    item for item in all_rows
                    if (item["report_date"], item["source"]) == key
                ),
                None,
            )

            if existing_in_run:
                old_values = (
                    existing_in_run["registered"],
                    existing_in_run["eligible"],
                    existing_in_run["total"],
                )
                new_values = (
                    row["registered"],
                    row["eligible"],
                    row["total"],
                )
                if any(abs(float(a) - float(b)) > 0.01
                       for a, b in zip(old_values, new_values)):
                    logging.error(
                        "不同文章对同一日期给出冲突数值，全部标记复核："
                        "日期=%s，旧值=%s，候选值=%s，旧来源=%s，新来源=%s",
                        row["report_date"], old_values, new_values,
                        existing_in_run.get("source_url"), row.get("source_url"),
                    )
                    all_rows.remove(existing_in_run)
                    seen_records.discard(key)
                    conflicted_records.add(key)
                else:
                    logging.info(
                        "重复来源记录数值一致，跳过重复项：%s",
                        row["report_date"],
                    )
                continue

            # 如果该日期已经因为冲突被移除，则不再接受本轮该日期的其他候选项。
            if key in conflicted_records:
                continue

            if key in seen_records:
                continue

            seen_records.add(key)
            all_rows.append(row)

    # -----------------------------------------------------
    # C. 展示解析结果
    # -----------------------------------------------------

    all_rows.sort(
        key=lambda item: item["report_date"],
        reverse=True,
    )

    save_preview(all_rows)
    audit_candidates(all_rows)

    logging.info(
        "本次合计解析到 %s 条候选库存记录（尚未等同于来源已核实）。",
        len(all_rows),
    )

    print("\n========== COMEX 库存解析预览 ==========")

    for row in all_rows:

        print(
            f"日期：{row['report_date']} | "
            f"注册：{row['registered']:,.3f} 盎司 | "
            f"非注册：{row['eligible']:,.3f} 盎司 | "
            f"总量：{row['total']:,.3f} 盎司"
        )

        print(f"来源：{row['source_url']}")
        print("-" * 60)

    if not all_rows:

        logging.warning(
            "没有解析到有效数据。请检查搜索结果与网页正文。"
        )

        return

    # -----------------------------------------------------
    # D. 与数据库现有记录比较
    # -----------------------------------------------------

    existing_conflicts = check_existing_rows(all_rows)
    if existing_conflicts:
        before_count = len(all_rows)
        all_rows = [
            row for row in all_rows
            if (row["report_date"], row["source"]) not in existing_conflicts
        ]
        logging.warning(
            "因与数据库已有值冲突，已将 %s 条候选记录从可入库列表移出；详情请查日志。",
            before_count - len(all_rows),
        )
        save_preview(all_rows)

    # -----------------------------------------------------
    # E. 入库
    # -----------------------------------------------------

    inserted = save_inventory(all_rows)

    if DRY_RUN:

        logging.info(
            "预览结束：没有写入数据库。"
            "请检查 parsed_preview.json 后再决定是否入库。"
        )

    else:

        logging.info(
            "采集结束：解析 %s 条，新增 %s 条。",
            len(all_rows),
            inserted,
        )

    # -----------------------------------------------------
    # F. 查看数据库最新记录
    # -----------------------------------------------------

    with sqlite3.connect(DB_PATH) as conn:

        rows = conn.execute("""
            SELECT
                report_date,
                registered,
                eligible,
                total,
                source
            FROM comex_inventory
            ORDER BY report_date DESC
            LIMIT 10
        """).fetchall()

    print("\n========== 数据库现有最新记录 ==========")

    for row in rows:
        print(row)

    print("========================================")

    logging.info("COMEX 黄金库存采集结束。")


if __name__ == "__main__":
    main()
# 本文件用于测试 COMEX 库存结构化采集模块
"""
采集模块单元测试。

重点覆盖容易出错且后果严重的部分：
1. 日期偏移自检：上游改口径时必须能被发现，而不是静默产出错一天的数据。
2. 跨源交叉验证：同一 CME 业务日的总库存不一致时必须标记冲突。
3. 四级核验分级：VERIFIED / ACCEPTED / REVIEW / REJECTED 的判定条件。
4. 入库幂等：重复抓取不得产生重复行；数值变化必须让旧核验结论失效。
5. 旧数据作废：只标记 REJECTED，绝不物理删除，保留审计证据。
6. 预览模式：dry_run 下不得写入任何数据。

测试全程不联网：抓取函数不在测试范围内，只测纯逻辑与数据库操作。
"""

import json
import sqlite3
from datetime import date

import pytest

from src.data_sources import comex_inventory_api as api


# =========================================================
# 测试夹具
# =========================================================

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """把模块的数据库路径指向临时库，避免污染项目数据库。"""

    db_path = tmp_path / "test.db"

    monkeypatch.setattr(api, "DB_PATH", db_path)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "EVIDENCE_DIR", tmp_path / "evidence")

    with sqlite3.connect(db_path) as conn:
        api.ensure_schema(conn)

    return db_path


def row(report_date, total, registered=None, eligible=None,
        source="eastmoney", source_date=None, url="https://example.com/a"):
    """构造一条采集记录。"""

    return {
        "source": source,
        "source_date": source_date or report_date,
        "report_date": report_date,
        "registered": registered,
        "eligible": eligible,
        "total": total,
        "ton": None,
        "source_url": url,
    }


# =========================================================
# 日期偏移自检
# =========================================================

def test_detect_offset_finds_correct_shift():
    """偏移自检应识别出 primary 相对 reference 的正确天数差。"""

    reference = [
        row("2026-10-16", 23_426_323.294, source="metalcharts"),
        row("2026-10-09", 23_477_464.405, source="metalcharts"),
    ]

    # primary 的 source_date 比真实业务日早 1 天
    primary = [
        row("2026-10-16", 23_426_323.294, source_date="2026-10-15"),
        row("2026-10-09", 23_477_464.405, source_date="2026-10-08"),
    ]

    best, detail = api.detect_date_offset(primary, reference)

    assert best == 1
    assert detail["match_counts_by_offset"][1] == 2
    assert best in detail["valid_offsets"]


def test_detect_offset_returns_none_without_overlap():
    """两源无重叠区间时，不得凭空判出某个偏移有匹配。"""

    reference = [row("2026-10-16", 100.0, source="metalcharts")]
    primary = [row("2025-01-01", 100.0, source_date="2025-01-01")]

    best, detail = api.detect_date_offset(primary, reference)

    # 没有任何值变化日能对上，匹配数必须全为 0
    assert not any(detail["match_counts_by_offset"].values())
    assert detail["reason"]
    # 回退时只从「不产生周末」的合法偏移里选，绝不猜一个有匹配的偏移
    assert best in detail["valid_offsets"]


def test_detect_offset_rejects_wrong_values():
    """日期对得上但数值对不上时，不得判定为该偏移有效。"""

    reference = [row("2026-10-16", 23_426_323.294, source="metalcharts")]

    primary = [
        row("2026-10-16", 99_999_999.0, source_date="2026-10-15"),
    ]

    best, detail = api.detect_date_offset(primary, reference)

    assert detail["match_counts_by_offset"][1] == 0


# =========================================================
# 单条校验
# =========================================================

def test_validate_accepts_total_only_row():
    """只有 total 的日频记录应通过校验（拆分字段允许缺失）。"""

    errors, flags = api.validate_row(row("2026-10-16", 23_426_323.294))

    assert errors == []
    assert flags == []


def test_validate_rejects_inconsistent_sum():
    """registered + eligible 与 total 不一致时必须报错。"""

    errors, _ = api.validate_row(
        row(
            "2026-10-16",
            total=100.0,
            registered=60.0,
            eligible=50.0,
            source="metalcharts",
        )
    )

    assert any("不一致" in item for item in errors)


def test_validate_accepts_consistent_split():
    """拆分字段自洽时应通过校验。"""

    errors, _ = api.validate_row(
        row(
            "2026-10-16",
            total=100.0,
            registered=60.0,
            eligible=40.0,
            source="metalcharts",
        )
    )

    assert errors == []


def test_validate_rejects_non_positive_total():
    """总库存必须为正数。"""

    errors, _ = api.validate_row(row("2026-10-16", 0.0))

    assert errors
    assert any("total" in item for item in errors)


def test_validate_rejects_future_date():
    """报告日期不得晚于当前日期。"""

    errors, _ = api.validate_row(row("2999-01-01", 100.0))

    assert any("晚于当前日期" in item for item in errors)


def test_validate_flags_weekend_date():
    """日期落在周末应给出提示（CME 业务日通常为工作日）。"""

    # 2026-10-17 是周六
    errors, flags = api.validate_row(row("2026-10-17", 100.0))

    assert errors == []
    assert any("周末" in item for item in flags)


def test_validate_rejects_missing_url():
    """缺少来源网址无法溯源，应报错。"""

    errors, _ = api.validate_row(
        row("2026-10-16", 100.0, url="")
    )

    assert any("来源网址" in item for item in errors)


def test_validate_rejects_bad_date_format():
    """日期格式错误应报错。"""

    errors, _ = api.validate_row(row("16/10/2026", 100.0))

    assert any("report_date" in item for item in errors)


# =========================================================
# 跨源交叉验证
# =========================================================

def test_cross_validate_accepts_matching_sources():
    """两源同一业务日数值一致时应判定通过。"""

    rows = [
        row("2026-10-16", 23_426_323.294, source="eastmoney"),
        row("2026-10-16", 23_426_323.294, source="metalcharts"),
    ]

    passed, conflicts = api.cross_validate(rows)

    assert passed == {("2026-10-16", "eastmoney"), ("2026-10-16", "metalcharts")}
    assert conflicts == []


def test_cross_validate_detects_conflict():
    """两源同一业务日数值不一致时必须标记冲突。"""

    rows = [
        row("2026-02-20", 34_094_940.93, source="eastmoney"),
        row("2026-02-20", 33_774_506.555, source="metalcharts"),
    ]

    passed, conflicts = api.cross_validate(rows)

    assert passed == set()
    assert len(conflicts) == 1
    assert conflicts[0]["report_date"] == "2026-02-20"
    assert conflicts[0]["spread_pct"] > 0


def test_cross_validate_tolerates_rounding():
    """四舍五入级别的微小差异不应误判为冲突。"""

    rows = [
        row("2026-10-16", 23_426_323.294, source="eastmoney"),
        row("2026-10-16", 23_426_323.000, source="metalcharts"),
    ]

    passed, conflicts = api.cross_validate(rows)

    assert conflicts == []
    assert len(passed) == 2


def test_cross_validate_skips_single_source():
    """只有单一来源时不做交叉验证（既不通过也不冲突）。"""

    rows = [row("2026-10-16", 100.0, source="eastmoney")]

    passed, conflicts = api.cross_validate(rows)

    assert passed == set()
    assert conflicts == []


# =========================================================
# 极端变化检测
# =========================================================

def test_flag_large_changes_detects_jump():
    """相邻记录变化超过阈值应被标记。"""

    rows = [
        row("2026-09-19", 27_383_786.190),
        row("2026-09-22", 23_356_326.071),
    ]

    warnings = api.flag_large_changes(rows)

    assert len(warnings) == 1
    assert warnings[0]["report_date"] == "2026-09-22"
    assert warnings[0]["change_percent"] > 10


def test_flag_large_changes_ignores_small_moves():
    """正常幅度的变化不应被标记。"""

    rows = [
        row("2026-10-15", 23_459_459.851),
        row("2026-10-16", 23_426_323.294),
    ]

    assert api.flag_large_changes(rows) == []


def test_flag_large_changes_isolated_per_source():
    """不同来源之间不得互相比较相邻变化。"""

    rows = [
        row("2026-10-16", 100.0, source="eastmoney"),
        row("2026-10-16", 999_999.0, source="metalcharts"),
    ]

    assert api.flag_large_changes(rows) == []


# =========================================================
# 数据库结构
# =========================================================

def test_ensure_schema_creates_both_tables(temp_db):
    """初始化应同时创建库存表与质量表。"""

    with sqlite3.connect(temp_db) as conn:
        api.ensure_schema(conn)

        tables = {
            item[0]
            for item in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert "comex_inventory" in tables
    assert "comex_inventory_quality" in tables


def test_ensure_schema_is_idempotent(temp_db):
    """重复初始化不得报错，也不得丢数据。"""

    api.save_rows([row("2026-10-16", 100.0)], dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        api.ensure_schema(conn)
        api.ensure_schema(conn)

        count = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory"
        ).fetchone()[0]

    assert count == 1


def test_ensure_schema_adds_columns_to_legacy_table(tmp_path, monkeypatch):
    """旧库缺少新增列时应自动补齐，且保留原有数据。"""

    db_path = tmp_path / "legacy.db"

    monkeypatch.setattr(api, "DB_PATH", db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE comex_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                registered REAL, eligible REAL, total REAL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(report_date, source)
            )
        """)
        conn.execute(
            "INSERT INTO comex_inventory "
            "(report_date, registered, eligible, total, source) "
            "VALUES ('2026-10-16', 1, 2, 3, 'cngold')"
        )
        conn.commit()

    with sqlite3.connect(db_path) as conn:
        api.ensure_schema(conn)

        columns = {
            item[1]
            for item in conn.execute("PRAGMA table_info(comex_inventory)")
        }

        preserved = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory WHERE source='cngold'"
        ).fetchone()[0]

    assert "source_date" in columns
    assert "updated_at" in columns
    assert preserved == 1


# =========================================================
# 入库幂等性
# =========================================================

def test_save_rows_is_idempotent(temp_db):
    """重复入库同一记录不得产生重复行。"""

    rows = [row("2026-10-16", 100.0)]

    first = api.save_rows(rows, dry_run=False)
    second = api.save_rows(rows, dry_run=False)

    assert first["inserted"] == 1
    assert second["unchanged"] == 1
    assert second["inserted"] == 0

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory"
        ).fetchone()[0]

    assert count == 1


def test_save_rows_updates_changed_value(temp_db):
    """数值变化时应更新记录而不是新增。"""

    api.save_rows([row("2026-10-16", 100.0)], dry_run=False)
    result = api.save_rows([row("2026-10-16", 200.0)], dry_run=False)

    assert result["updated"] == 1
    assert result["inserted"] == 0

    with sqlite3.connect(temp_db) as conn:
        total = conn.execute(
            "SELECT total FROM comex_inventory WHERE report_date='2026-10-16'"
        ).fetchone()[0]

    assert total == 200.0


def test_save_rows_resets_quality_on_change(temp_db):
    """数值被修改后，旧的核验结论必须失效。"""

    rows = [row("2026-10-16", 100.0)]

    api.save_rows(rows, dry_run=False)
    api.apply_quality_status(rows, {("2026-10-16", "eastmoney")}, [], dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        before = conn.execute(
            "SELECT status FROM comex_inventory_quality "
            "WHERE report_date='2026-10-16'"
        ).fetchone()[0]

    assert before == api.STATUS_VERIFIED

    changed = [row("2026-10-16", 999.0)]
    api.save_rows(changed, dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        after = conn.execute(
            "SELECT status, reviewed_at FROM comex_inventory_quality "
            "WHERE report_date='2026-10-16'"
        ).fetchone()

    assert after[0] == api.STATUS_REVIEW
    assert after[1] is None


def test_save_rows_records_source_date(temp_db):
    """应保留来源原始日期，使日期校正可审计。"""

    rows = [
        row("2026-10-16", 100.0, source_date="2026-10-15"),
    ]

    api.save_rows(rows, dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        saved = conn.execute(
            "SELECT report_date, source_date FROM comex_inventory"
        ).fetchone()

    assert saved == ("2026-10-16", "2026-10-15")


def test_dry_run_writes_nothing(temp_db):
    """预览模式不得写入任何数据。"""

    rows = [row("2026-10-16", 100.0)]

    stats = api.save_rows(rows, dry_run=True)

    assert stats == {"inserted": 0, "updated": 0, "unchanged": 0}

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory"
        ).fetchone()[0]

    assert count == 0


# =========================================================
# 四级核验分级
# =========================================================

def test_quality_verified_on_cross_source_match(temp_db):
    """跨源数值一致应判为 VERIFIED。"""

    rows = [
        row("2026-10-16", 100.0, source="eastmoney"),
        row("2026-10-16", 100.0, source="metalcharts"),
    ]

    api.save_rows(rows, dry_run=False)

    stats = api.apply_quality_status(
        rows,
        {("2026-10-16", "eastmoney"), ("2026-10-16", "metalcharts")},
        [],
        dry_run=False,
    )

    assert stats[api.STATUS_VERIFIED] == 2

    with sqlite3.connect(temp_db) as conn:
        record = conn.execute(
            "SELECT status, source_url, reviewed_at "
            "FROM comex_inventory_quality WHERE source='eastmoney'"
        ).fetchone()

    assert record[0] == api.STATUS_VERIFIED
    assert record[1] == "https://example.com/a"
    assert record[2] is not None


def test_quality_accepted_for_single_source(temp_db):
    """单源数据通过全部自动校验时应判为 ACCEPTED，而不是永远不可用。"""

    rows = [row("2026-10-16", 100.0)]

    api.save_rows(rows, dry_run=False)

    stats = api.apply_quality_status(rows, set(), [], dry_run=False)

    assert stats[api.STATUS_ACCEPTED] == 1

    with sqlite3.connect(temp_db) as conn:
        status = conn.execute(
            "SELECT status FROM comex_inventory_quality"
        ).fetchone()[0]

    assert status == api.STATUS_ACCEPTED


def test_quality_review_on_conflict(temp_db):
    """跨源冲突应判为 REVIEW，不得进入可信集合。"""

    rows = [
        row("2026-10-16", 100.0, source="eastmoney"),
        row("2026-10-16", 999.0, source="metalcharts"),
    ]

    conflicts = [{"report_date": "2026-10-16", "totals": {}, "spread_pct": 9.0}]

    api.save_rows(rows, dry_run=False)

    stats = api.apply_quality_status(
        rows, set(), conflicts, dry_run=False
    )

    assert stats[api.STATUS_REVIEW] == 2
    assert api.STATUS_ACCEPTED not in stats
    assert api.STATUS_VERIFIED not in stats


def test_quality_review_on_validation_error(temp_db):
    """校验未通过应判为 REVIEW。"""

    rows = [
        row("2026-10-16", 100.0, url=""),  # 缺来源网址
    ]

    api.save_rows(rows, dry_run=False)

    stats = api.apply_quality_status(rows, set(), [], dry_run=False)

    assert stats[api.STATUS_REVIEW] == 1


def test_quality_review_on_large_change(temp_db):
    """极端变化应判为 REVIEW 并等待人工复核。"""

    rows = [row("2026-10-16", 100.0)]

    large = [
        {
            "source": "eastmoney",
            "report_date": "2026-10-16",
            "related_date": "2026-10-15",
            "change_percent": 15.0,
        }
    ]

    api.save_rows(rows, dry_run=False)

    stats = api.apply_quality_status(
        rows, set(), [], large_changes=large, dry_run=False
    )

    assert stats[api.STATUS_REVIEW] == 1

    with sqlite3.connect(temp_db) as conn:
        reason = conn.execute(
            "SELECT reason FROM comex_inventory_quality"
        ).fetchone()[0]

    assert "10%" in reason


def test_quality_dry_run_writes_nothing(temp_db):
    """预览模式不得写入核验状态。"""

    rows = [row("2026-10-16", 100.0)]

    stats = api.apply_quality_status(rows, set(), [], dry_run=True)

    assert stats == {} or not stats

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory_quality"
        ).fetchone()[0]

    assert count == 0


def test_quality_stores_source_url_for_traceability(temp_db):
    """核验记录必须带来源网址，否则等于闭眼盖章。"""

    rows = [row("2026-10-16", 100.0, url="https://data.eastmoney.com/x")]

    api.save_rows(rows, dry_run=False)
    api.apply_quality_status(rows, set(), [], dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        url = conn.execute(
            "SELECT source_url FROM comex_inventory_quality"
        ).fetchone()[0]

    assert url == "https://data.eastmoney.com/x"


def test_quality_upsert_does_not_duplicate(temp_db):
    """重复写入核验状态应更新而非新增。"""

    rows = [row("2026-10-16", 100.0)]

    api.save_rows(rows, dry_run=False)
    api.apply_quality_status(rows, set(), [], dry_run=False)
    api.apply_quality_status(rows, set(), [], dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory_quality"
        ).fetchone()[0]

    assert count == 1


# =========================================================
# 旧数据作废
# =========================================================

def test_reject_legacy_marks_without_deleting(temp_db):
    """作废旧来源时只改状态，绝不物理删除数据。"""

    legacy = [row("2026-10-14", 100.0, source="cngold")]
    fresh = [row("2026-10-16", 200.0, source="eastmoney")]

    api.save_rows(legacy + fresh, dry_run=False)

    rejected = api.reject_legacy_source(dry_run=False)

    assert rejected == 1

    with sqlite3.connect(temp_db) as conn:
        inventory_count = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory"
        ).fetchone()[0]

        legacy_rows = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory WHERE source='cngold'"
        ).fetchone()[0]

        status = conn.execute(
            "SELECT status FROM comex_inventory_quality "
            "WHERE source='cngold'"
        ).fetchone()[0]

    # 数据仍在，只是被标记为已证伪
    assert inventory_count == 2
    assert legacy_rows == 1
    assert status == api.STATUS_REJECTED


def test_reject_legacy_records_reason(temp_db):
    """作废必须留下可追溯的理由。"""

    api.save_rows([row("2026-10-14", 100.0, source="cngold")], dry_run=False)

    api.reject_legacy_source(dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        reason = conn.execute(
            "SELECT reason FROM comex_inventory_quality WHERE source='cngold'"
        ).fetchone()[0]

    assert "日期" in reason
    assert "审计" in reason


def test_reject_legacy_overrides_previous_verified(temp_db):
    """先前被误判为已核验的旧数据，作废后必须失去可信状态。"""

    legacy = [row("2026-10-15", 100.0, source="cngold")]

    api.save_rows(legacy, dry_run=False)
    api.apply_quality_status(
        legacy, {("2026-10-15", "cngold")}, [], dry_run=False
    )

    with sqlite3.connect(temp_db) as conn:
        before = conn.execute(
            "SELECT status FROM comex_inventory_quality"
        ).fetchone()[0]

    assert before == api.STATUS_VERIFIED

    api.reject_legacy_source(dry_run=False)

    with sqlite3.connect(temp_db) as conn:
        after = conn.execute(
            "SELECT status, reviewed_at FROM comex_inventory_quality"
        ).fetchone()

    assert after[0] == api.STATUS_REJECTED
    assert after[1] is None


def test_reject_legacy_dry_run_does_not_write(temp_db):
    """预览模式不得改动核验状态。"""

    api.save_rows([row("2026-10-14", 100.0, source="cngold")], dry_run=False)

    count = api.reject_legacy_source(dry_run=True)

    assert count == 1

    with sqlite3.connect(temp_db) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory_quality"
        ).fetchone()[0]

    assert rows == 0


# =========================================================
# 证据留档
# =========================================================

def test_save_evidence_writes_json(temp_db, tmp_path):
    """证据应落盘为可读 JSON。"""

    path = api.save_evidence({"a": 1, "b": [1, 2]}, "test.json")

    assert path is not None
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {"a": 1, "b": [1, 2]}


def test_save_evidence_failure_does_not_raise(temp_db, monkeypatch, tmp_path):
    """证据目录不可写时不得中断采集，只记录警告。"""

    # 把 EVIDENCE_DIR 指向一个「已存在的普通文件」：
    # mkdir 对文件必然失败，且在 Windows 与 POSIX 上行为一致。
    # （不能用 /proc/nonexistent 这类 Linux 路径：在 Windows 上会被
    #   解析成 C:\\proc\\... 这个可创建的路径，测试就失去意义了。）
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    monkeypatch.setattr(api, "EVIDENCE_DIR", blocker / "nested")

    assert api.save_evidence({"a": 1}, "test.json") is None


# =========================================================
# 可信状态常量
# =========================================================

def test_trusted_statuses_exclude_review_and_rejected():
    """可信状态集合必须排除待复核与已证伪。"""

    assert api.STATUS_VERIFIED in api.TRUSTED_STATUSES
    assert api.STATUS_ACCEPTED in api.TRUSTED_STATUSES
    assert api.STATUS_REVIEW not in api.TRUSTED_STATUSES
    assert api.STATUS_REJECTED not in api.TRUSTED_STATUSES


def test_date_offsets_documented():
    """各源日期偏移必须有明确记录。

    东方财富的 REPORT_DATE 本身就是 CME 业务日（偏移 0）。
    早期实现曾误设为 +1，会凭空制造周六业务日 —— 该回归由下方
    test_detect_offset_rejects_weekend_shift 锁定。
    """

    assert api.DATE_OFFSET_DAYS[api.SOURCE_METALCHARTS] == 0
    assert api.DATE_OFFSET_DAYS[api.SOURCE_EASTMONEY] == 0


def test_detect_offset_rejects_weekend_shift():
    """核心回归：会制造周末业务日的偏移必须被判为无效，即使匹配数更高。

    构造一段「库存长期不变」的数据——这正是早期误判的成因：
    持续不变的值在多种偏移下都能碰巧匹配，若只看匹配数会选错偏移，
    把周五的库存错标成周六。业务日硬约束必须优先于匹配数量。
    """

    # reference：以真实业务日（含周五）为基准
    reference = [
        row("2026-10-16", 100.0, source="metalcharts"),  # 周五
        row("2026-10-09", 100.0, source="metalcharts"),  # 周五
    ]

    # primary：库存恒为 100，因此 +1 偏移也能匹配到 reference 的值，
    # 但 +1 会把周五推成周六。
    primary = [
        row("2026-10-16", 100.0, source_date="2026-10-16"),  # 周五
        row("2026-10-09", 100.0, source_date="2026-10-09"),  # 周五
    ]

    best, detail = api.detect_date_offset(primary, reference)

    weekend_counts = detail["weekend_counts_by_offset"]

    # +1 会产生周六业务日，必须被排除
    assert weekend_counts[1] > 0
    assert weekend_counts[0] == 0

    # 即便 +1 的匹配数不低于 0，也必须选 0
    assert best == 0
    assert 1 not in detail["valid_offsets"]
    assert 0 in detail["valid_offsets"]


def test_is_weekend_date():
    """周末判定辅助函数应正确识别周六周日。"""

    assert api._is_weekend_date("2026-10-17") is True   # 周六
    assert api._is_weekend_date("2026-10-18") is True   # 周日
    assert api._is_weekend_date("2026-10-16") is False  # 周五
    assert api._is_weekend_date("2026-10-19") is False  # 周一
    assert api._is_weekend_date(None) is False
    assert api._is_weekend_date("bad-date") is False

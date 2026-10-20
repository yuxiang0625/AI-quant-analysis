# 本文件用于测试 COMEX 库存页面的来源判定与人工核验写回逻辑
"""
库存页面逻辑测试。

重点覆盖两个会造成误导的问题：
1. 已作废（REJECTED）来源不得被选为默认分析对象，也不得参与相关性计算。
2. 人工核验写回必须落库、可追溯，且不得产生重复行。

测试使用临时数据库，不触碰项目数据库，也不联网。
"""

import sqlite3

import pandas as pd
import pytest

from src.ui import comex_page


# =========================================================
# 测试数据构造
# =========================================================

def inventory_frame(rows):
    """构造库存页面所需的 DataFrame。"""

    frame = pd.DataFrame(rows)

    frame["report_date"] = pd.to_datetime(frame["report_date"])

    for column in ("registered", "eligible", "total"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def make_multi_source():
    """构造含三个来源、三种核验状态的库存数据。"""

    dates = pd.bdate_range("2026-09-01", periods=6)

    rows = []

    # 日频来源：全部可信
    for index, item in enumerate(dates):
        rows.append(
            {
                "report_date": item,
                "registered": None,
                "eligible": None,
                "total": 23_400_000.0 + index * 1000,
                "source": "eastmoney",
                "source_url": "https://data.eastmoney.com/x",
                "status": "ACCEPTED",
            }
        )

    # 周频来源：跨源已验证
    for index, item in enumerate(dates[:3]):
        rows.append(
            {
                "report_date": item,
                "registered": 14_800_000.0,
                "eligible": 8_600_000.0,
                "total": 23_400_000.0 + index * 1000,
                "source": "metalcharts",
                "source_url": "https://metalcharts.org/data",
                "status": "VERIFIED",
            }
        )

    # 旧来源：全部作废
    for index, item in enumerate(dates[:4]):
        rows.append(
            {
                "report_date": item,
                "registered": 14_800_000.0,
                "eligible": 8_600_000.0,
                "total": 23_400_000.0,
                "source": "cngold",
                "source_url": None,
                "status": "REJECTED",
            }
        )

    return inventory_frame(rows)


# =========================================================
# 状态归一化
# =========================================================

def test_normalize_status_maps_known_values():
    """四级状态应原样保留。"""

    for status in ("VERIFIED", "ACCEPTED", "REVIEW", "REJECTED"):
        assert comex_page.normalize_status(status) == status


def test_normalize_status_maps_legacy_value():
    """历史取值 UNVERIFIED 应归一到待复核，不得当成可信。"""

    assert comex_page.normalize_status("UNVERIFIED") == comex_page.STATUS_REVIEW


def test_normalize_status_handles_none():
    """空状态应返回 None 而不是抛异常。"""

    assert comex_page.normalize_status(None) is None
    assert comex_page.normalize_status(float("nan")) is None


def test_normalize_status_is_case_insensitive():
    """状态判定应忽略大小写。"""

    assert comex_page.normalize_status("verified") == "VERIFIED"
    assert comex_page.normalize_status("Rejected") == "REJECTED"


# =========================================================
# 已作废来源判定
# =========================================================

def test_rejected_source_detected():
    """整体作废的来源必须被识别出来。"""

    frame = make_multi_source()

    cngold = frame[frame["source"] == "cngold"]

    assert comex_page.is_rejected_source(cngold) is True


def test_trusted_source_not_rejected():
    """可信来源不得被误判为作废。"""

    frame = make_multi_source()

    for source in ("eastmoney", "metalcharts"):
        subset = frame[frame["source"] == source]

        assert comex_page.is_rejected_source(subset) is False


def test_partially_rejected_source_not_rejected():
    """只有部分记录作废时，不视为整体作废，以便查看其可信部分。"""

    frame = make_multi_source()

    cngold = frame[frame["source"] == "cngold"].copy()

    cngold.loc[cngold.index[0], "status"] = "ACCEPTED"

    assert comex_page.is_rejected_source(cngold) is False


def test_rejected_check_handles_empty_and_missing_status():
    """空数据或缺少状态列时不得抛异常。"""

    assert comex_page.is_rejected_source(pd.DataFrame()) is False

    frame = make_multi_source().drop(columns=["status"])

    assert comex_page.is_rejected_source(frame) is False


# =========================================================
# 默认来源选择
# =========================================================

def test_primary_source_excludes_rejected():
    """默认来源绝不能是已作废的来源。"""

    frame = make_multi_source()

    selected = comex_page.pick_primary_source(frame)

    assert selected != "cngold"
    assert selected in ("eastmoney", "metalcharts")


def test_primary_source_prefers_trusted_records():
    """默认来源应优先选择可信记录更多的来源。"""

    frame = make_multi_source()

    selected = comex_page.pick_primary_source(frame)

    trusted_by_source = (
        frame.assign(
            trusted=frame["status"].map(comex_page.normalize_status).isin(
                comex_page.TRUSTED_STATUSES
            )
        )
        .groupby("source")["trusted"]
        .sum()
    )

    assert trusted_by_source[selected] == trusted_by_source.max()


def test_primary_source_falls_back_when_all_rejected():
    """全部来源都作废时仍需返回一个，让页面能展示审计信息。"""

    frame = make_multi_source()

    frame["status"] = "REJECTED"

    selected = comex_page.pick_primary_source(frame)

    assert selected is not None


def test_primary_source_handles_empty():
    """空数据应返回 None。"""

    assert comex_page.pick_primary_source(pd.DataFrame()) is None
    assert comex_page.pick_primary_source(None) is None


def test_primary_source_handles_missing_status_column():
    """缺少状态列时不得抛异常。"""

    frame = make_multi_source().drop(columns=["status"])

    assert comex_page.pick_primary_source(frame) is not None


# =========================================================
# 人工核验写回
# =========================================================

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """把页面模块的数据库连接指向临时库。"""

    db_path = tmp_path / "test.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE comex_inventory_quality (
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
        conn.commit()

    class FakeConnection:
        def __enter__(self):
            self.conn = sqlite3.connect(db_path)
            return self.conn

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
            return False

    monkeypatch.setattr(
        comex_page, "get_connection", lambda: FakeConnection()
    )

    return db_path


def read_status(db_path, report_date, source):
    """读取指定记录的核验状态与理由。"""

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, reason, source_url, reviewed_at "
            "FROM comex_inventory_quality "
            "WHERE report_date = ? AND source = ?",
            (report_date, source),
        ).fetchone()

    return row


def test_manual_review_writes_verified(temp_db):
    """人工确认应写入 VERIFIED，并留下说明与核验时间。"""

    comex_page.record_manual_review(
        report_date="2026-10-16",
        source="eastmoney",
        status="VERIFIED",
        note="已对照东方财富原始页面，数值与业务日均一致",
        source_url="https://data.eastmoney.com/x",
    )

    status, reason, url, reviewed_at = read_status(
        temp_db, "2026-10-16", "eastmoney"
    )

    assert status == "VERIFIED"
    assert "东方财富原始页面" in reason
    assert url == "https://data.eastmoney.com/x"
    assert reviewed_at is not None


def test_manual_review_writes_rejected(temp_db):
    """人工作废应写入 REJECTED，且不设核验时间。"""

    comex_page.record_manual_review(
        report_date="2026-10-15",
        source="cngold",
        status="REJECTED",
        note="日期比 CME 业务日早一天，已证伪",
    )

    status, reason, _, reviewed_at = read_status(
        temp_db, "2026-10-15", "cngold"
    )

    assert status == "REJECTED"
    assert "早一天" in reason
    assert reviewed_at is None


def test_manual_review_requires_allowed_status(temp_db):
    """不得写入 ACCEPTED 等只应由自动核验产生的状态。"""

    with pytest.raises(ValueError):
        comex_page.record_manual_review(
            report_date="2026-10-16",
            source="eastmoney",
            status="ACCEPTED",
            note="测试",
        )


def test_manual_review_rejects_unknown_status(temp_db):
    """未知状态应被拒绝，避免污染状态集合。"""

    with pytest.raises(ValueError):
        comex_page.record_manual_review(
            report_date="2026-10-16",
            source="eastmoney",
            status="MAYBE",
            note="测试",
        )


def test_manual_review_is_upsert(temp_db):
    """重复核验同一记录应更新而非新增。"""

    comex_page.record_manual_review(
        "2026-10-16", "eastmoney", "REVIEW", "第一次：待核对原文"
    )
    comex_page.record_manual_review(
        "2026-10-16", "eastmoney", "VERIFIED", "第二次：已核对一致"
    )

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM comex_inventory_quality"
        ).fetchone()[0]

    assert count == 1

    status, reason, _, _ = read_status(temp_db, "2026-10-16", "eastmoney")

    assert status == "VERIFIED"
    assert "第二次" in reason


def test_manual_review_preserves_existing_url(temp_db):
    """未提供网址时应保留原有网址，不得覆盖为空。"""

    comex_page.record_manual_review(
        "2026-10-16",
        "eastmoney",
        "REVIEW",
        "初次登记",
        source_url="https://data.eastmoney.com/x",
    )

    comex_page.record_manual_review(
        "2026-10-16", "eastmoney", "VERIFIED", "已核对"
    )

    _, _, url, _ = read_status(temp_db, "2026-10-16", "eastmoney")

    assert url == "https://data.eastmoney.com/x"


def test_manual_review_can_reverse_rejection(temp_db):
    """误作废后应能改回已验证，保留纠错能力。"""

    comex_page.record_manual_review(
        "2026-10-16", "eastmoney", "REJECTED", "疑似错误"
    )

    comex_page.record_manual_review(
        "2026-10-16", "eastmoney", "VERIFIED", "复核后确认数据正确"
    )

    status, _, _, reviewed_at = read_status(temp_db, "2026-10-16", "eastmoney")

    assert status == "VERIFIED"
    assert reviewed_at is not None


# =========================================================
# 展示辅助
# =========================================================

def test_source_label_translates_known_sources():
    """已知来源代号应翻译成中文说明。"""

    assert "东方财富" in comex_page._source_label("eastmoney")
    assert "已作废" in comex_page._source_label("cngold")


def test_source_label_passes_through_unknown():
    """未知来源应原样返回，不得报错。"""

    assert comex_page._source_label("some_new_source") == "some_new_source"


def test_fmt_handles_missing_values():
    """缺失值应显示破折号而不是 nan。"""

    assert comex_page._fmt(None) == "—"
    assert comex_page._fmt(float("nan")) == "—"
    assert comex_page._fmt(1234567.0) == "1,234,567"
    assert comex_page._fmt(1234.5678, 2, " oz") == "1,234.57 oz"


def test_status_labels_cover_all_statuses():
    """四级状态都必须有中文标签，避免页面显示原始英文枚举。"""

    for status in comex_page.TRUSTED_STATUSES + (
        comex_page.STATUS_REVIEW,
        comex_page.STATUS_REJECTED,
    ):
        assert status in comex_page.STATUS_LABELS
        assert status in comex_page.STATUS_COLORS


def test_trusted_statuses_match_api_module():
    """页面与采集模块的可信状态定义必须一致。"""

    from src.data_sources import comex_inventory_api as api

    assert set(comex_page.TRUSTED_STATUSES) == set(api.TRUSTED_STATUSES)


def test_lag_days_match_factor_module():
    """页面与因子模块的库存公布滞后必须一致，否则口径会错开一天。"""

    from src.quant.factors import INVENTORY_LAG_DAYS

    assert comex_page.INVENTORY_LAG_DAYS == INVENTORY_LAG_DAYS

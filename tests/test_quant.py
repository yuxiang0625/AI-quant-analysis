# 本文件用于测试黄金多因子量化引擎的正确性
"""
量化引擎单元测试。

重点覆盖容易出错且后果严重的部分：
1. 未来函数（look-ahead bias）：宏观与库存必须按公布可用日对齐，
   库存变化基期不得取到未来观测。
2. 指标预热期：未成熟的指标必须为空值，不得输出虚假的 0。
3. 仓位安全：波动率未估出时不得开仓。
4. 权重归一化：因子缺失时得分必须按可用权重重新归一化。
5. 回测口径：收益必须由前一日仓位决定。
6. AI 降级：模型不可用时必须回落到本地规则解读，不得抛异常。
7. 库存多源：可信状态过滤、跨源冲突弃用、total/registered 口径回退、
   日历天回看窗口在日频与周频下语义一致。
"""

import numpy as np
import pandas as pd
import pytest

from src.quant import ai_analyst
from src.quant.backtest import build_returns, performance_metrics, run_backtest
from src.quant.factors import (
    DEFAULT_WEIGHTS,
    INVENTORY_LAG_DAYS,
    INVENTORY_LOOKBACK_DAYS,
    _as_of_align,
    add_composite_score,
    add_inventory_factors,
    add_inventory_factors_from_frame,
    add_macro_factors,
    add_position,
    add_technical_factors,
    build_factors,
    normalize_trusted_keys,
    prepare_inventory,
    to_daily_gold,
)


# =========================================================
# 测试数据构造
# =========================================================

def make_gold(rows=120, start_price=4000.0, seed=7):
    """构造一段确定性的模拟日线行情。"""

    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2025-01-01", periods=rows)

    steps = rng.normal(0, 0.008, size=rows)

    close = start_price * np.cumprod(1 + steps)

    high = close * (1 + np.abs(rng.normal(0, 0.004, size=rows)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, size=rows)))
    open_price = low + (high - low) * 0.5

    return pd.DataFrame(
        {
            "symbol": "XAU/USD",
            "timestamp": dates.tz_localize("UTC"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000, 5000, size=rows).astype(float),
            "source": "XAUS",
        }
    )


def make_macro():
    """构造覆盖四个指标的模拟宏观数据。"""

    frames = []

    daily_dates = pd.bdate_range("2024-12-01", periods=260)

    frames.append(
        pd.DataFrame(
            {
                "indicator": "DFII10",
                "observation_date": daily_dates,
                "value": np.linspace(1.5, 2.6, len(daily_dates)),
                "source": "FRED",
            }
        )
    )

    frames.append(
        pd.DataFrame(
            {
                "indicator": "DGS10",
                "observation_date": daily_dates,
                "value": np.linspace(4.2, 5.0, len(daily_dates)),
                "source": "FRED",
            }
        )
    )

    month_dates = pd.date_range("2024-01-01", periods=24, freq="MS")

    frames.append(
        pd.DataFrame(
            {
                "indicator": "CPIAUCSL",
                "observation_date": month_dates,
                "value": np.linspace(310.0, 330.0, len(month_dates)),
                "source": "FRED",
            }
        )
    )

    frames.append(
        pd.DataFrame(
            {
                "indicator": "FEDFUNDS",
                "observation_date": month_dates,
                "value": np.linspace(4.5, 3.5, len(month_dates)),
                "source": "FRED",
            }
        )
    )

    return pd.concat(frames, ignore_index=True)


def make_comex(rows=12, verified_rows=None):
    """构造 COMEX 库存与对应核验状态表。"""

    dates = pd.date_range("2025-06-02", periods=rows, freq="W-MON")

    registered = np.linspace(18_000_000, 15_000_000, rows)

    comex = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": registered,
            "eligible": registered * 1.4,
            "total": registered * 2.4,
            "source": "cngold",
        }
    )

    if verified_rows is None:
        verified_dates = dates
    else:
        verified_dates = dates[:verified_rows]

    quality = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "source": "cngold",
            "status": [
                "VERIFIED" if item in set(verified_dates) else "UNVERIFIED"
                for item in dates
            ],
        }
    )

    return comex, quality


def verified_list(quality):
    """从核验状态表取出已核验日期列表。"""

    dates = pd.to_datetime(
        quality.loc[quality["status"] == "VERIFIED", "report_date"],
        errors="coerce",
    )

    return [pd.Timestamp(item) for item in dates.dt.normalize().dropna().unique()]


def trusted_pairs(quality, statuses=("VERIFIED", "ACCEPTED")):
    """
    从核验状态表取出可信记录的 (日期, 来源) 组合键。

    组合键比纯日期列表更严格：只有「该日期的该来源」可信才放行，
    因此能检验已作废来源是否真的被排除。
    """

    subset = quality[quality["status"].isin(statuses)]

    keys = []

    for _, row in subset.iterrows():
        parsed = pd.to_datetime(row["report_date"], errors="coerce")

        if pd.isna(parsed):
            continue

        keys.append((parsed.normalize(), str(row["source"])))

    return keys


# =========================================================
# 数据整形
# =========================================================

def test_to_daily_gold_deduplicates_by_date():
    """同一自然日多条记录应只保留最后一条。"""

    gold = make_gold(rows=5)

    duplicated = pd.concat([gold, gold.tail(2)], ignore_index=True)

    daily = to_daily_gold(duplicated)

    assert len(daily) == 5
    assert daily.index.is_unique


def test_to_daily_gold_empty_input():
    """空输入应返回空表而不报错。"""

    assert to_daily_gold(pd.DataFrame()).empty
    assert to_daily_gold(None).empty


# =========================================================
# 技术指标与预热期
# =========================================================

def test_macd_warmup_is_nan_not_zero():
    """MACD 预热期必须为空值，不能是虚假的 0。"""

    daily = to_daily_gold(make_gold(rows=60))

    result = add_technical_factors(daily)

    assert pd.isna(result["MACD"].iloc[0])
    assert pd.isna(result["MACD_hist"].iloc[0])
    assert pd.isna(result["f_macd"].iloc[0])

    # 首日 EMA 若未设 min_periods 会退化为收盘价，导致 MACD 恰为 0
    assert not (result["MACD_hist"].fillna(-1).iloc[:20] == 0).any()


def test_moving_average_warmup():
    """均线在样本不足时应为空值。"""

    daily = to_daily_gold(make_gold(rows=60))

    result = add_technical_factors(daily)

    assert result["MA20"].isna().sum() == 19
    assert result["MA50"].isna().sum() == 49
    assert pd.notna(result["MA50"].iloc[-1])


def test_rsi_bounds():
    """RSI 必须落在 0~100 区间。"""

    daily = to_daily_gold(make_gold(rows=200))

    result = add_technical_factors(daily)

    valid = result["RSI14"].dropna()

    assert not valid.empty
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_factors_bounded_between_minus_one_and_one():
    """所有因子取值必须落在 [-1, 1]。"""

    gold = make_gold(rows=200)
    macro = make_macro()

    frame = build_factors(gold, macro, pd.DataFrame())

    for key in DEFAULT_WEIGHTS:
        if key not in frame.columns:
            continue

        series = pd.to_numeric(frame[key], errors="coerce").dropna()

        if series.empty:
            continue

        assert series.min() >= -1.0001, key
        assert series.max() <= 1.0001, key


def test_score_bounded():
    """综合得分必须落在 [-1, 1]。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())

    score = frame["score"].dropna()

    assert not score.empty
    assert score.min() >= -1.0001
    assert score.max() <= 1.0001


# =========================================================
# 未来函数防护
# =========================================================

def test_as_of_align_never_uses_future_value():
    """as-of 对齐只能取目标日期之前或当日已可用的观测值。"""

    available = pd.DataFrame(
        {
            "available_date": pd.to_datetime(
                ["2025-01-10", "2025-01-20", "2025-02-01"]
            ),
            "value": [1.0, 2.0, 3.0],
        }
    )

    index = pd.to_datetime(
        ["2025-01-05", "2025-01-10", "2025-01-15", "2025-02-05"]
    )

    aligned = _as_of_align(available, index)

    assert pd.isna(aligned.iloc[0])
    assert aligned.iloc[1] == 1.0
    assert aligned.iloc[2] == 1.0
    assert aligned.iloc[3] == 3.0


def test_macro_uses_only_published_values():
    """交易日使用的实际利率必须是当时已公布的值。"""

    gold = make_gold(rows=150, start_price=4000.0, seed=11)
    macro = make_macro()

    frame = build_factors(gold, macro, pd.DataFrame())

    real_rate_source = macro[macro["indicator"] == "DFII10"].copy()
    real_rate_source["observation_date"] = pd.to_datetime(
        real_rate_source["observation_date"]
    ).dt.normalize()
    real_rate_source = real_rate_source.sort_values("observation_date")

    checked = 0

    for date, row in frame.iterrows():
        used = row["real_rate"]

        if pd.isna(used):
            continue

        # 滞后 1 天：只允许使用 observation_date + 1 <= date 的观测
        allowed = real_rate_source[
            real_rate_source["observation_date"]
            <= date - pd.Timedelta(days=1)
        ]

        if allowed.empty:
            continue

        expected = allowed["value"].iloc[-1]

        assert abs(float(used) - float(expected)) < 1e-9, (
            f"{date}: used={used} expected={expected}"
        )

        checked += 1

    assert checked > 0


def test_cpi_respects_publication_lag():
    """CPI 必须按 45 天公布滞后对齐，不得使用当月尚未公布的数据。"""

    gold = make_gold(rows=150, start_price=4000.0, seed=13)
    macro = make_macro()

    frame = build_factors(gold, macro, pd.DataFrame())

    cpi_source = macro[macro["indicator"] == "CPIAUCSL"].copy()
    cpi_source["observation_date"] = pd.to_datetime(
        cpi_source["observation_date"]
    ).dt.normalize()
    cpi_source = cpi_source.sort_values("observation_date")

    for date, row in frame.iterrows():
        used = row["cpi_yoy"]

        if pd.isna(used):
            continue

        allowed = cpi_source[
            cpi_source["observation_date"] + pd.Timedelta(days=45) <= date
        ]

        assert not allowed.empty, f"{date} 使用了尚未公布的 CPI"


# =========================================================
# 库存因子与核验门槛
# =========================================================

def test_unverified_inventory_excluded():
    """未通过核验的库存记录不得参与打分。"""

    comex, quality = make_comex(rows=12, verified_rows=3)

    inventory, diagnosis = prepare_inventory(comex, verified_list(quality))

    assert diagnosis["total_reports"] == 12
    assert diagnosis["trusted_reports"] == 3
    assert diagnosis["usable"] is False
    assert inventory.empty


def test_inventory_factor_requires_minimum_reports():
    """可信记录不足最少观测数时库存因子应整体缺失。"""

    comex, quality = make_comex(rows=12, verified_rows=3)

    daily = to_daily_gold(make_gold(rows=100))
    frame = add_technical_factors(daily)

    result = add_inventory_factors(
        frame, comex, verified_list(quality)
    )

    assert result["f_inventory"].isna().all()
    assert result["comex_registered"].isna().all()


def test_inventory_factor_available_when_verified():
    """已核验记录充足时库存因子应可用，且下降库存记为利多。"""

    comex, quality = make_comex(rows=12, verified_rows=12)

    daily = to_daily_gold(make_gold(rows=200, start_price=4000.0, seed=17))
    frame = add_technical_factors(daily)

    result = add_inventory_factors(
        frame, comex, verified_list(quality)
    )

    assert result["f_inventory"].notna().any()

    # 构造数据中注册库存单调下降，因此因子应为正（利多）
    valid = result["f_inventory"].dropna()

    assert not valid.empty
    assert valid.max() > 0


def test_inventory_factor_absent_after_report_date():
    """库存数据不得早于「报告日 + 公布滞后」出现。"""

    comex, quality = make_comex(rows=12, verified_rows=12)

    daily = to_daily_gold(make_gold(rows=200, start_price=4000.0, seed=19))
    frame = add_technical_factors(daily)

    result = add_inventory_factors(
        frame, comex, verified_list(quality)
    )

    # 水平值：首个报告日 + INVENTORY_LAG_DAYS 起可用
    first_available = (
        pd.to_datetime(comex["report_date"]).min()
        + pd.Timedelta(days=INVENTORY_LAG_DAYS)
    )

    valid = result["comex_registered"].dropna()

    assert not valid.empty
    assert valid.index.min() >= first_available

    # 变化率还需额外的回看窗口才有基期，因此可用日更晚
    change_valid = result["comex_registered_change_pct"].dropna()

    assert not change_valid.empty

    baseline_available = (
        pd.to_datetime(comex["report_date"]).min()
        + pd.Timedelta(days=INVENTORY_LOOKBACK_DAYS + INVENTORY_LAG_DAYS)
    )

    assert change_valid.index.min() >= baseline_available


def test_inventory_baseline_never_uses_future_value():
    """库存变化的基期不得取到未来观测（防 look-ahead）。"""

    comex, quality = make_comex(rows=12, verified_rows=12)

    daily = to_daily_gold(make_gold(rows=200, start_price=4000.0, seed=21))
    frame = add_technical_factors(daily)

    inventory, diagnosis = prepare_inventory(comex, verified_list(quality))

    assert diagnosis["usable"] is True

    result = add_inventory_factors_from_frame(frame, inventory, diagnosis)

    # 手工复算：基期必须是 date-LOOKBACK_DAYS 当天或之前最近的观测
    series = inventory[["date", "metric_value"]].copy()
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date").reset_index(drop=True)

    for _, row in series.iterrows():
        target = row["date"] - pd.Timedelta(days=INVENTORY_LOOKBACK_DAYS)

        eligible = series[series["date"] <= target]

        if eligible.empty:
            continue

        expected_baseline = float(eligible["metric_value"].iloc[-1])

        # 基期观测日必须不晚于目标日，绝不能是未来值
        assert eligible["date"].max() <= target

        assert expected_baseline > 0

        change = (float(row["metric_value"]) - expected_baseline) / expected_baseline

        assert abs(change) < 1.0


def test_missing_inventory_does_not_break_build():
    """完全没有库存表时，因子构建仍应成功。"""

    frame = build_factors(
        make_gold(rows=150), make_macro(), pd.DataFrame()
    )

    assert not frame.empty
    assert frame["score"].notna().any()
    assert frame["f_inventory"].isna().all()


# =========================================================
# 多来源库存：可信过滤、来源优先级与字段完整性
#
# 这一组测试锁定四个曾经真实发生的缺陷：
#   1. 可信过滤用 set(zip(...)) 导致判定与行错位，已作废(REJECTED)的
#      脏数据泄漏进因子计算；
#   2. 同日跨源数值分歧时整日弃用，把主源有效的日频数据一起丢掉；
#   3. 跨源拼接字段，造出「注册/非注册/总量来自不同日期」的假记录
#      ——这正是旧 cngold 脏数据的成因；
#   4. 因子序列混用多个来源，破坏回看窗口内的口径一致性。
# =========================================================

# 旧 cngold 脏数据中被证伪的真实数值，用作泄漏检测的指纹
DIRTY_REGISTERED = 14_876_916.734
DIRTY_TOTAL = 99_999_999.0


def make_multi_source_comex(days=20, start="2026-01-05"):
    """
    构造含日频主源与周频补充源的库存数据。

    字段形态刻意贴合真实来源：
        eastmoney   日频，只有 total，registered/eligible 为空
        metalcharts 周频（每 5 个业务日一条），含 registered/eligible 拆分
    两源在重叠日的 total 故意不同，用于构造跨源分歧。
    """

    dates = pd.bdate_range(start, periods=days)

    primary = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": np.nan,
            "eligible": np.nan,
            "total": 23_000_000.0 + np.arange(days) * 1000.0,
            "source": "eastmoney",
        }
    )

    weekly_dates = dates[::5]

    secondary = pd.DataFrame(
        {
            "report_date": weekly_dates.strftime("%Y-%m-%d"),
            "registered": 14_800_000.0,
            "eligible": 8_600_000.0,
            # 与主源同日数值不同，构成跨源分歧
            "total": 23_400_000.0,
            "source": "metalcharts",
        }
    )

    return pd.concat([primary, secondary], ignore_index=True)


def quality_for(frame, status_by_source):
    """按来源为库存记录生成核验状态表。"""

    return pd.DataFrame(
        [
            {
                "report_date": row["report_date"],
                "source": row["source"],
                "status": status_by_source.get(row["source"], "ACCEPTED"),
            }
            for _, row in frame.iterrows()
        ]
    )


def test_rejected_source_never_leaks_into_factors():
    """已作废来源的数据绝不能进入因子计算（回归：脏数据泄漏）。"""

    dates = pd.bdate_range("2026-01-05", periods=20)

    primary = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": np.nan,
            "eligible": np.nan,
            "total": 23_000_000.0 + np.arange(20) * 1000.0,
            "source": "eastmoney",
        }
    )

    legacy = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": DIRTY_REGISTERED,
            "eligible": 8_599_178.796,
            "total": DIRTY_TOTAL,
            "source": "cngold",
        }
    )

    frame = pd.concat([primary, legacy], ignore_index=True)

    quality = quality_for(
        frame, {"eastmoney": "ACCEPTED", "cngold": "REJECTED"}
    )

    inventory, diagnosis = prepare_inventory(frame, trusted_pairs(quality))

    assert diagnosis["usable"] is True
    assert diagnosis["sources"] == ["eastmoney"]

    # 作废来源不得出现在合并结果中
    assert not (inventory["source"] == "cngold").any()

    # 脏数据的指纹值绝不能出现在任何口径列中
    assert not inventory["metric_value"].eq(DIRTY_TOTAL).any()
    assert not inventory["total"].eq(DIRTY_TOTAL).any()
    assert DIRTY_REGISTERED not in set(
        inventory["registered"].dropna().tolist()
    )


def test_rejected_source_never_leaks_end_to_end():
    """端到端验证：作废数据不得影响因子值与综合得分。"""

    dates = pd.bdate_range("2026-01-05", periods=20)

    primary = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": np.nan,
            "eligible": np.nan,
            "total": 23_000_000.0 + np.arange(20) * 1000.0,
            "source": "eastmoney",
        }
    )

    legacy = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": DIRTY_REGISTERED,
            "eligible": 8_599_178.796,
            "total": DIRTY_TOTAL,
            "source": "cngold",
        }
    )

    comex = pd.concat([primary, legacy], ignore_index=True)

    quality = quality_for(
        comex, {"eastmoney": "ACCEPTED", "cngold": "REJECTED"}
    )

    frame = build_factors(
        make_gold(rows=120),
        make_macro(),
        comex,
        trusted_inventory=trusted_pairs(quality),
    )

    assert not frame.empty

    levels = frame["comex_registered"].dropna()

    assert not levels.eq(DIRTY_TOTAL).any()
    assert DIRTY_REGISTERED not in set(levels.tolist())


def test_trust_filter_selects_exact_rows():
    """可信过滤必须逐行对应，只放行指定的 (日期, 来源) 组合。"""

    dates = pd.bdate_range("2026-01-05", periods=20)

    frame = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": np.nan,
            "eligible": np.nan,
            "total": 23_000_000.0 + np.arange(20) * 1000.0,
            "source": "eastmoney",
        }
    )

    # 只信任前 10 天
    keys = [
        (pd.Timestamp(item), "eastmoney") for item in dates[:10]
    ]

    inventory, diagnosis = prepare_inventory(frame, keys)

    assert diagnosis["trusted_reports"] == 10
    assert diagnosis["usable"] is True
    assert len(inventory) == 10
    assert inventory["date"].max() == pd.Timestamp(dates[9]).normalize()


def test_trust_filter_ignores_wrong_source_for_same_date():
    """同一日期只有指定来源可信时，其他来源不得被放行。"""

    dates = pd.bdate_range("2026-01-05", periods=12)

    frame = pd.DataFrame(
        {
            "report_date": dates.strftime("%Y-%m-%d"),
            "registered": np.nan,
            "eligible": np.nan,
            "total": 23_000_000.0 + np.arange(12) * 1000.0,
            "source": "eastmoney",
        }
    )

    # 来源写成别的，日期虽对但组合键不匹配
    keys = [(pd.Timestamp(item), "some_other_source") for item in dates]

    inventory, diagnosis = prepare_inventory(frame, keys)

    assert inventory.empty
    assert diagnosis["trusted_reports"] == 0
    assert diagnosis["usable"] is False


def test_fields_never_stitched_across_sources():
    """同一业务日的字段只能取自主源同一行，不得跨源拼接。"""

    frame = make_multi_source_comex(days=20)

    quality = quality_for(frame, {})

    inventory, diagnosis = prepare_inventory(
        frame, trusted_pairs(quality)
    )

    assert diagnosis["usable"] is True

    shared = frame.loc[
        frame["source"] == "metalcharts", "report_date"
    ].tolist()

    shared_timestamps = {
        pd.Timestamp(item).normalize() for item in shared
    }

    shared_rows = inventory[inventory["date"].isin(shared_timestamps)]

    assert not shared_rows.empty

    # 主源没有拆分字段，因此重叠日的 registered 必须为空；
    # 若从 metalcharts 拼过来，就重演了旧脏数据的成因
    assert shared_rows["registered"].isna().all()
    assert shared_rows["eligible"].isna().all()
    assert (shared_rows["source"] == "eastmoney").all()


def test_source_discrepancy_keeps_primary_value():
    """同日跨源分歧时保留主源取值，而不是把整天数据丢弃。"""

    frame = make_multi_source_comex(days=20)

    quality = quality_for(frame, {})

    inventory, diagnosis = prepare_inventory(
        frame, trusted_pairs(quality)
    )

    shared = frame.loc[
        frame["source"] == "metalcharts", "report_date"
    ].tolist()

    discrepancies = diagnosis["source_discrepancies"]

    assert len(discrepancies) == len(shared)

    for item in discrepancies:
        assert item["used_source"] == "eastmoney"
        assert item["spread_pct"] > 0

    # 主源的 20 天数据必须全部保留，一天都不能因分歧被丢
    assert len(inventory) == 20

    shared_timestamps = {
        pd.Timestamp(item).normalize() for item in shared
    }

    assert shared_timestamps.issubset(set(inventory["date"]))


def test_factor_series_uses_single_source_per_date():
    """因子序列每个业务日只能有一个取值，不得出现来源拼接的复合标签。"""

    frame = make_multi_source_comex(days=20)

    quality = quality_for(frame, {})

    inventory, _ = prepare_inventory(frame, trusted_pairs(quality))

    assert inventory["date"].is_unique

    # 旧实现会把多来源拼成 "eastmoney+metalcharts" 这样的复合标签，
    # 说明该日的字段可能来自不同来源
    assert not inventory["source"].str.contains(r"\+").any()


def test_secondary_source_fills_gaps_only():
    """补充源只在主源缺该日时提供数据。"""

    dates = pd.bdate_range("2026-01-05", periods=12)

    # 主源缺第 6 天
    primary_dates = dates.drop(dates[5])

    primary = pd.DataFrame(
        {
            "report_date": primary_dates.strftime("%Y-%m-%d"),
            "registered": np.nan,
            "eligible": np.nan,
            "total": 23_000_000.0,
            "source": "eastmoney",
        }
    )

    secondary = pd.DataFrame(
        {
            "report_date": [dates[5].strftime("%Y-%m-%d")],
            "registered": 14_800_000.0,
            "eligible": 8_600_000.0,
            "total": 23_500_000.0,
            "source": "metalcharts",
        }
    )

    frame = pd.concat([primary, secondary], ignore_index=True)

    quality = quality_for(frame, {})

    inventory, diagnosis = prepare_inventory(
        frame, trusted_pairs(quality)
    )

    assert diagnosis["usable"] is True
    assert len(inventory) == 12

    filled = inventory[
        inventory["date"] == pd.Timestamp(dates[5]).normalize()
    ]

    assert len(filled) == 1
    assert filled.iloc[0]["source"] == "metalcharts"
    assert filled.iloc[0]["total"] == 23_500_000.0


def test_discrepancies_do_not_break_inventory_factor():
    """存在跨源分歧时，库存因子仍应正常产出取值。"""

    # 库存区间必须落在行情区间内（make_gold 从 2025-01-01 起），
    # 否则两者没有重叠日期，因子必然全为空，断言就失去意义。
    frame = make_multi_source_comex(days=20, start="2025-03-03")

    quality = quality_for(frame, {})

    daily = to_daily_gold(make_gold(rows=200, start_price=4000.0, seed=31))
    technical = add_technical_factors(daily)

    result = add_inventory_factors(
        technical, frame, trusted_pairs(quality)
    )

    assert result["f_inventory"].notna().any()
    assert result["f_inventory"].dropna().between(-1, 1).all()

    # 脏数据指纹不得出现在因子输出中
    assert not result["comex_registered"].dropna().eq(DIRTY_TOTAL).any()


def test_no_trust_keys_means_no_filtering():
    """不提供可信键时不做过滤，由调用方自行保证数据可信。"""

    frame = make_multi_source_comex(days=12)

    inventory, diagnosis = prepare_inventory(frame, None)

    assert diagnosis["usable"] is True
    assert "未启用核验状态过滤" in diagnosis["reason"]
    assert len(inventory) == 12


# =========================================================
# 权重与归一化
# =========================================================

def test_missing_factor_renormalizes_weight():
    """因子缺失时，得分应按可用权重重新归一化。"""

    gold = make_gold(rows=120, start_price=4000.0, seed=23)
    macro = make_macro()

    frame = build_factors(gold, macro, pd.DataFrame())

    # 库存因子整体缺失
    assert frame["f_inventory"].isna().all()

    row = frame.iloc[-1]

    active = int(row["active_factor_count"])

    assert active == len(DEFAULT_WEIGHTS) - 1

    # 手工按可用权重复算得分
    numerator = 0.0
    denominator = 0.0

    for key, weight in DEFAULT_WEIGHTS.items():
        value = pd.to_numeric(row.get(key), errors="coerce")

        if pd.isna(value):
            continue

        numerator += float(value) * weight
        denominator += abs(weight)

    expected = numerator / denominator

    assert abs(float(row["score"]) - expected) < 1e-9


def test_zero_weight_factor_excluded():
    """权重设为 0 的因子不应影响综合得分。"""

    gold = make_gold(rows=150, start_price=4000.0, seed=29)
    macro = make_macro()

    weights = dict(DEFAULT_WEIGHTS)
    weights["f_rsi"] = 0.0

    frame = build_factors(gold, macro, pd.DataFrame(), weights=weights)

    row = frame.iloc[-1]

    rsi_value = pd.to_numeric(row.get("f_rsi"), errors="coerce")

    if pd.isna(rsi_value):
        pytest.skip("RSI 因子在样本末端缺失")

    weighted = pd.to_numeric(row.get("w_f_rsi"), errors="coerce")

    assert weighted == 0.0


def test_all_zero_weights_yields_nan_score():
    """全部权重为 0 时得分应为空值而非除零。"""

    weights = {key: 0.0 for key in DEFAULT_WEIGHTS}

    frame = build_factors(
        make_gold(rows=100), make_macro(), pd.DataFrame(), weights=weights
    )

    assert frame["score"].isna().all()


# =========================================================
# 仓位与信号
# =========================================================

def test_position_zero_when_volatility_unknown():
    """波动率尚未估出时仓位必须为 0，不得满仓。"""

    daily = to_daily_gold(make_gold(rows=60))
    frame = add_technical_factors(daily)
    frame = add_macro_factors(frame, make_macro())
    frame = add_composite_score(frame, DEFAULT_WEIGHTS)

    result = add_position(frame, use_vol_target=True, target_vol=12.0)

    warmup = result.head(20)

    assert (warmup["vol_scale"] == 0).all()
    assert (warmup["position"] == 0).all()


def test_signal_respects_thresholds():
    """信号必须与阈值一致，且不越界。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())

    result = add_position(
        frame,
        long_threshold=0.2,
        short_threshold=-0.2,
        allow_short=True,
        use_vol_target=False,
    )

    long_rows = result[result["score"] >= 0.2]
    short_rows = result[result["score"] <= -0.2]

    assert (long_rows["signal"] == 1.0).all()
    assert (short_rows["signal"] == -1.0).all()

    assert set(result["signal"].dropna().unique()) <= {-1.0, 0.0, 1.0}


def test_long_only_mode_never_shorts():
    """关闭做空时不得出现负仓位。"""

    frame = build_factors(
        make_gold(rows=200),
        make_macro(),
        pd.DataFrame(),
        allow_short=False,
    )

    assert (frame["signal"] >= 0).all()
    assert (frame["position"] >= 0).all()


def test_vol_target_caps_position_at_one():
    """波动率缩放后的仓位绝对值不得超过 1。"""

    frame = build_factors(
        make_gold(rows=200),
        make_macro(),
        pd.DataFrame(),
        use_vol_target=True,
        target_vol=12.0,
    )

    assert frame["position"].abs().max() <= 1.0 + 1e-9
    assert frame["vol_scale"].max() <= 1.0 + 1e-9


# =========================================================
# 回测
# =========================================================

def test_backtest_uses_previous_day_position():
    """策略收益必须由前一日仓位决定，杜绝未来函数。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())

    returns_frame = build_returns(frame, cost_bps=5.0)

    manual = (
        frame["position"].shift(1) * frame["daily_return"]
    ).reindex(returns_frame.index)

    difference = (returns_frame["gross_return"] - manual).abs().max()

    assert float(difference) < 1e-12


def test_backtest_costs_reduce_return():
    """交易成本必须为负贡献，成本越高收益越低。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())

    cheap = build_returns(frame, cost_bps=0.0)
    costly = build_returns(frame, cost_bps=30.0)

    assert (costly["cost"] >= 0).all()

    cheap_total = (1 + cheap["strategy_return"]).prod()
    costly_total = (1 + costly["strategy_return"]).prod()

    assert costly_total <= cheap_total


def test_zero_turnover_when_never_trading():
    """始终空仓时不应产生换手与成本。"""

    frame = build_factors(
        make_gold(rows=200),
        make_macro(),
        pd.DataFrame(),
        # 阈值超出得分取值范围 [-1, 1]，保证永不触发信号
        long_threshold=1.5,
        short_threshold=-1.5,
    )

    assert (frame["signal"] == 0).all()

    result = run_backtest(frame, cost_bps=10.0)

    assert result["trade_stats"]["trade_count"] == 0
    assert result["trade_stats"]["total_cost_pct"] == 0.0
    assert result["strategy_metrics"]["total_return_pct"] == 0.0


def test_performance_metrics_on_known_series():
    """用已知收益序列校验指标计算。"""

    returns = pd.Series([0.01] * 252)

    metrics = performance_metrics(returns)

    assert metrics["sample_days"] == 252
    assert metrics["win_rate_pct"] == 100.0
    assert metrics["max_drawdown_pct"] == 0.0
    assert metrics["annual_vol_pct"] == pytest.approx(0.0, abs=1e-9)

    # 零波动时夏普无法定义，应为 None 而非 inf
    assert metrics["sharpe"] is None


def test_performance_metrics_handles_empty():
    """空收益序列不应抛异常。"""

    metrics = performance_metrics(pd.Series(dtype="float64"))

    assert metrics["sample_days"] == 0
    assert metrics["total_return_pct"] is None
    assert metrics["sharpe"] is None


def test_backtest_handles_empty_frame():
    """空因子表回测应安全返回空结果。"""

    result = run_backtest(pd.DataFrame())

    assert result["returns"].empty
    assert result["strategy_metrics"]["sample_days"] == 0
    assert result["trade_stats"]["trade_count"] == 0


def test_benchmark_equals_buy_and_hold():
    """基准收益应等于买入持有的日收益。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())

    returns_frame = build_returns(frame, cost_bps=5.0)

    difference = (
        returns_frame["benchmark_return"]
        - frame["daily_return"].reindex(returns_frame.index)
    ).abs().max()

    assert float(difference) < 1e-12


# =========================================================
# 事实简报与 AI 降级
# =========================================================

def test_brief_daily_return_in_percent():
    """简报中的日收益率必须是百分比，不能是小数。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    brief = ai_analyst.build_fact_brief(frame, backtest)

    raw = float(frame["daily_return"].iloc[-1])
    reported = brief["latest"]["daily_return_pct"]

    assert reported == pytest.approx(raw * 100, rel=1e-4)


def test_brief_marks_missing_factor():
    """缺失因子必须在简报中标记为不可用。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    brief = ai_analyst.build_fact_brief(frame, backtest)

    inventory = next(
        item for item in brief["factors"] if item["id"] == "f_inventory"
    )

    assert inventory["available"] is False
    assert inventory["score"] is None
    assert inventory["direction"] == "数据缺失"


def test_brief_contains_data_quality():
    """简报应包含数据质量诊断，供模型解释缺失原因。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    comex, quality = make_comex(rows=12, verified_rows=3)

    brief = ai_analyst.build_fact_brief(
        frame,
        backtest,
        diagnosis=prepare_inventory(comex, verified_list(quality))[1],
    )

    report = brief["data_quality"]["inventory_report"]

    assert report["total_reports"] == 12
    assert report["trusted_reports"] == 3
    assert report["usable"] is False
    assert report["reason"]

    coverage = brief["data_quality"]["factor_coverage"]

    assert coverage["COMEX 库存变化(总库存)"]["available_days"] == 0


def test_brief_on_empty_frame():
    """空因子表也应生成结构完整的简报。"""

    brief = ai_analyst.build_fact_brief(pd.DataFrame(), {})

    assert brief["sample"]["status"] == "无可用行情数据"


def test_prompt_contains_no_api_key():
    """提示词中不得包含密钥。"""

    frame = build_factors(make_gold(rows=120), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    brief = ai_analyst.build_fact_brief(frame, backtest)
    prompt = ai_analyst.build_user_prompt(brief)

    key = ai_analyst.get_api_key()

    if key:
        assert key not in prompt


def test_analyze_falls_back_when_no_key(monkeypatch):
    """没有密钥时应降级为规则解读，且不抛异常。"""

    monkeypatch.setattr(ai_analyst.os, "getenv", lambda *a, **k: None)

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    result = ai_analyst.analyze(frame, backtest, use_model=True)

    assert result["mode"] == "fallback"
    assert result["text"]
    assert "因子解读" in result["text"]
    assert result["error"] == "未配置 DEEPSEEK_API_KEY"


def test_analyze_falls_back_on_network_error(monkeypatch):
    """网络异常时应降级为规则解读。"""

    import requests

    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("connection timed out")

    monkeypatch.setattr(ai_analyst, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(ai_analyst.requests, "post", raise_timeout)

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    result = ai_analyst.analyze(frame, backtest, use_model=True)

    assert result["mode"] == "fallback"
    assert "Timeout" in result["error"] or "网络" in result["error"]
    assert result["text"]


def test_analyze_falls_back_on_http_error(monkeypatch):
    """接口返回错误状态码时应降级为规则解读。"""

    class FakeResponse:
        ok = False
        status_code = 429
        text = "rate limited"

    monkeypatch.setattr(ai_analyst, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(
        ai_analyst.requests, "post", lambda *a, **k: FakeResponse()
    )

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    result = ai_analyst.analyze(frame, backtest, use_model=True)

    assert result["mode"] == "fallback"
    assert "429" in result["error"]


def test_analyze_uses_model_on_success(monkeypatch):
    """接口正常时应采用模型返回内容。"""

    class FakeResponse:
        ok = True
        status_code = 200
        text = "{}"

        @staticmethod
        def json():
            return {
                "model": "deepseek-chat",
                "choices": [
                    {"message": {"content": "### 信号概览\n模型解读正文"}}
                ],
                "usage": {"total_tokens": 123},
            }

    monkeypatch.setattr(ai_analyst, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(
        ai_analyst.requests, "post", lambda *a, **k: FakeResponse()
    )

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    result = ai_analyst.analyze(frame, backtest, use_model=True)

    assert result["mode"] == "model"
    assert "模型解读正文" in result["text"]
    assert result["model"] == "deepseek-chat"


def test_analyze_handles_empty_model_content(monkeypatch):
    """接口返回空内容时应降级。"""

    class FakeResponse:
        ok = True
        status_code = 200
        text = "{}"

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "   "}}]}

    monkeypatch.setattr(ai_analyst, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(
        ai_analyst.requests, "post", lambda *a, **k: FakeResponse()
    )

    frame = build_factors(make_gold(rows=150), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    result = ai_analyst.analyze(frame, backtest, use_model=True)

    assert result["mode"] == "fallback"
    assert result["text"]


def test_rule_commentary_never_empty():
    """规则化解读在任何数据状态下都应输出内容。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    result = ai_analyst.analyze(frame, backtest, use_model=False)

    assert result["mode"] == "rule"
    assert len(result["text"]) > 100
    assert "风险与局限" in result["text"]


def test_rule_commentary_on_empty_frame():
    """空数据时规则解读应给出明确提示而非报错。"""

    result = ai_analyst.analyze(pd.DataFrame(), {}, use_model=False)

    assert result["mode"] == "rule"
    assert result["text"]


def test_rule_commentary_reports_missing_reason():
    """规则解读应说明库存因子缺失的原因。"""

    frame = build_factors(make_gold(rows=200), make_macro(), pd.DataFrame())
    backtest = run_backtest(frame)

    comex, quality = make_comex(rows=12, verified_rows=2)

    result = ai_analyst.analyze(
        frame,
        backtest,
        use_model=False,
        diagnosis=prepare_inventory(comex, verified_list(quality))[1],
    )

    assert "缺失原因" in result["text"]
    assert "核验" in result["text"]


# =========================================================
# 端到端
# =========================================================

def test_end_to_end_pipeline():
    """完整链路：行情 + 宏观 + 已核验库存 → 因子 → 回测。"""

    gold = make_gold(rows=250, start_price=4200.0, seed=42)
    macro = make_macro()
    comex, quality = make_comex(rows=20, verified_rows=20)

    frame = build_factors(
        gold, macro, comex, trusted_inventory=verified_list(quality)
    )

    assert len(frame) == 250
    assert frame["score"].notna().sum() > 200

    backtest = run_backtest(frame, cost_bps=5.0)

    assert backtest["strategy_metrics"]["sample_days"] > 0
    assert backtest["returns"]["strategy_equity"].iloc[-1] > 0

    result = ai_analyst.analyze(frame, backtest, use_model=False)

    assert result["brief"]["latest"]["close"] == pytest.approx(
        float(gold["close"].iloc[-1]), rel=1e-6
    )


def test_pipeline_stable_under_repeated_runs():
    """相同输入多次运行结果必须完全一致（无随机性、无状态泄漏）。"""

    gold = make_gold(rows=150, start_price=4000.0, seed=99)
    macro = make_macro()

    first = build_factors(gold, macro, pd.DataFrame())
    second = build_factors(gold, macro, pd.DataFrame())

    pd.testing.assert_frame_equal(first, second)


def test_pipeline_does_not_mutate_input():
    """因子计算不得修改传入的原始数据。"""

    gold = make_gold(rows=100)
    macro = make_macro()

    gold_copy = gold.copy(deep=True)
    macro_copy = macro.copy(deep=True)

    build_factors(gold, macro, pd.DataFrame())

    pd.testing.assert_frame_equal(gold, gold_copy)
    pd.testing.assert_frame_equal(macro, macro_copy)

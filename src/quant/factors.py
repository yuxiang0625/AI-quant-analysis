# 本文件负责把行情、宏观与库存数据转换成可解释的量化因子
"""
黄金多因子打分引擎。

设计原则：
1. 所有数值都由本地 pandas 计算得出，AI 只负责解读，不参与任何数字生成。
2. 宏观与库存数据一律按「公布可用日」对齐，禁止使用未来数据（look-ahead bias）。
3. 因子缺失时按可用权重重新归一化，避免早期样本被人为压低得分。
"""

import numpy as np
import pandas as pd


# =========================================================
# 1. 因子定义
# =========================================================

# 每个因子的默认权重，页面侧边可调
DEFAULT_WEIGHTS = {
    "f_trend_ma": 1.5,
    "f_price_ma": 1.0,
    "f_macd": 1.0,
    "f_momentum_20": 1.0,
    "f_rsi": 0.8,
    "f_real_rate": 1.5,
    "f_nominal_rate": 0.8,
    "f_inflation": 0.8,
    "f_fedfunds": 0.7,
    "f_inventory": 0.7,
}

# 因子中文名
FACTOR_LABELS = {
    "f_trend_ma": "均线趋势(MA20/MA50)",
    "f_price_ma": "价格位置(收盘价/MA20)",
    "f_macd": "MACD 柱动能",
    "f_momentum_20": "20日波动率调整动量",
    "f_rsi": "RSI14 超买超卖",
    "f_real_rate": "10年期实际利率方向",
    "f_nominal_rate": "10年期名义利率方向",
    "f_inflation": "CPI 同比通胀水平",
    "f_fedfunds": "联邦基金利率周期",
    "f_inventory": "COMEX 库存变化(总库存)",
}

# 因子分组，用于页面分区展示
FACTOR_GROUPS = {
    "技术面": [
        "f_trend_ma",
        "f_price_ma",
        "f_macd",
        "f_momentum_20",
        "f_rsi",
    ],
    "宏观面": [
        "f_real_rate",
        "f_nominal_rate",
        "f_inflation",
        "f_fedfunds",
    ],
    "实物面": [
        "f_inventory",
    ],
}

# +1 代表的含义，用于页面和提示词解释方向
FACTOR_MEANING = {
    "f_trend_ma": "MA20 在 MA50 之上，中期趋势偏多",
    "f_price_ma": "收盘价站上 MA20，短期结构偏多",
    "f_macd": "MACD 柱为正且继续走高，动能偏多",
    "f_momentum_20": "近20日涨幅相对自身波动率偏高，动量偏多",
    "f_rsi": "RSI14 处于超卖区，存在均值回归的反弹空间",
    "f_real_rate": "实际利率下行，持有黄金的机会成本降低",
    "f_nominal_rate": "名义利率下行，对金价形成支撑",
    "f_inflation": "通胀高于常态，黄金的抗通胀需求上升",
    "f_fedfunds": "政策利率处于下行通道，流动性环境偏宽松",
    "f_inventory": (
        "COMEX 总库存环比下降，实物交割需求偏强"
        "（口径为 total；注册/非注册拆分仅周频可得，未用于打分）"
    ),
}

# 宏观数据的公布滞后天数（保守取值，只使用当时已公布的观测值）
MACRO_LAG_DAYS = {
    "DGS10": 1,
    "DFII10": 1,
    "CPIAUCSL": 45,
    "FEDFUNDS": 35,
}

# 库存报告公布滞后天数
INVENTORY_LAG_DAYS = 1

# 库存变化回看窗口（日历天）。
# 用日历天而非「报告条数」，才能让日频（东方财富只有 total）与
# 周频（metalcharts 含 registered/eligible）两种节奏语义一致，
# 都表示「一周变化」，不会因为数据频率不同而变成 3 天 vs 3 周。
INVENTORY_LOOKBACK_DAYS = 7

# 指标优先级：因子序列必须单一口径，不能混用 registered 与 total。
# 优先 total —— 日频源（东方财富）覆盖长、密度高，7 天回看窗口才有意义；
# registered/eligible 拆分只有周频源提供，保留用于页面展示与交叉验证。
INVENTORY_METRIC_PREFERENCE = ("total", "registered")

# 选用某指标所需的最少观测数，否则回退到下一优先级
INVENTORY_MIN_OBSERVATIONS = 8

# 多源同一业务日数值的相对容差，超过则视为冲突并弃用该日
INVENTORY_SOURCE_TOLERANCE = 0.0001

# 来源优先级：同一业务日有多个来源时，按此顺序取第一个可用值。
# 东方财富是日频主源（1200+ 条、口径已验证），metalcharts 周频用于补空。
# 因子序列必须来自单一连贯口径，因此由优先级决定取数，而不是同日比对判冲突
# ——两源对「同一观测挂在哪一天」的约定相差一天（周频源常用发布日标注），
# 同日精确比对会把这种约定差异误判成数值冲突，白白丢掉主源的有效日频数据。
# 数据质量的裁决（跨源验证、极端变化、核验状态）已在采集层完成。
INVENTORY_SOURCE_PRIORITY = (
    "eastmoney",
    "metalcharts",
)

# 参与分析的核验状态
TRUSTED_INVENTORY_STATUSES = ("VERIFIED", "ACCEPTED")

# 各因子的中性死区，避免噪声频繁翻转信号
DEADBAND_RATE_DAILY = 0.05      # 日频利率20日变化，单位百分点
DEADBAND_RATE_MONTHLY = 0.10    # 月频政策利率3个月变化，单位百分点
DEADBAND_INVENTORY_PCT = 0.005  # 库存变化幅度死区，0.5%

# 通胀因子的锚点：同比 2% 视为中性，每偏离 3 个百分点记满分
CPI_NEUTRAL_YOY = 2.0
CPI_FULL_SCALE_YOY = 3.0


# =========================================================
# 2. 数据整形
# =========================================================

def to_daily_gold(gold_df: pd.DataFrame) -> pd.DataFrame:
    """把带时区的行情压缩成按自然日去重的日线表。"""

    if gold_df is None or gold_df.empty:
        return pd.DataFrame()

    df = gold_df.copy()

    df["date"] = (
        pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    df = df.dropna(subset=["date", "close"])

    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )

    return df


def _as_of_align(frame: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
    """
    按「可用日期」把序列对齐到交易日索引。

    frame 需要包含 available_date 与 value 两列，
    只有 available_date <= 交易日 的观测值才会被使用。
    """

    empty = pd.Series(np.nan, index=target_index, dtype="float64")

    if frame is None or frame.empty or len(target_index) == 0:
        return empty

    prepared = frame.dropna(subset=["available_date", "value"]).copy()

    if prepared.empty:
        return empty

    left = pd.DataFrame({"date": target_index}).sort_values("date")
    right = prepared.sort_values("available_date")

    merged = pd.merge_asof(
        left,
        right[["available_date", "value"]],
        left_on="date",
        right_on="available_date",
        direction="backward",
    )

    return merged.set_index("date")["value"].reindex(target_index)


def _macro_frame(macro_df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """取出单个宏观指标并按观测日期去重排序。"""

    if macro_df is None or macro_df.empty:
        return pd.DataFrame(columns=["observation_date", "value"])

    subset = macro_df[macro_df["indicator"] == indicator].copy()

    if subset.empty:
        return pd.DataFrame(columns=["observation_date", "value"])

    subset["observation_date"] = pd.to_datetime(
        subset["observation_date"], errors="coerce"
    ).dt.normalize()

    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")

    subset = (
        subset.dropna(subset=["observation_date", "value"])
        .sort_values("observation_date")
        .drop_duplicates(subset=["observation_date"], keep="last")
    )

    return subset[["observation_date", "value"]]


def _with_available_date(subset: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """为宏观观测值加上公布可用日期。"""

    if subset.empty:
        return pd.DataFrame(columns=["available_date", "value"])

    prepared = subset.copy()

    prepared["available_date"] = (
        prepared["observation_date"] + pd.Timedelta(days=lag_days)
    )

    return prepared[["available_date", "value"]]


# =========================================================
# 3. 技术面因子
# =========================================================

def add_technical_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """在日线表上计算技术指标并生成技术面因子。"""

    df = daily.copy()

    close = df["close"]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    df["daily_return"] = close.pct_change()

    # --- RSI14 ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - 100 / (1 + rs)

    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI14"] = 100

    # --- MACD ---
    # 必须设置 min_periods：否则首日 EMA 直接等于收盘价，
    # MACD 与柱状图会得到虚假的 0 值，进而污染动能因子。
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(
        span=9, min_periods=9, adjust=False
    ).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # --- ATR 与波动率，用于仓位缩放 ---
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["ATR_pct"] = df["ATR14"] / close * 100

    df["vol_20d_annual"] = (
        df["daily_return"].rolling(20).std() * np.sqrt(252) * 100
    )

    # --- 因子：均线趋势 ---
    df["f_trend_ma"] = np.where(
        df["MA20"].isna() | df["MA50"].isna(),
        np.nan,
        np.sign(df["MA20"] - df["MA50"]),
    )

    # --- 因子：价格位置 ---
    df["f_price_ma"] = np.where(
        df["MA20"].isna(),
        np.nan,
        np.sign(close - df["MA20"]),
    )

    # --- 因子：MACD 动能（方向与变化需一致才给分）---
    hist_rising = df["MACD_hist"].diff() > 0
    hist_falling = df["MACD_hist"].diff() < 0

    df["f_macd"] = np.nan
    df.loc[(df["MACD_hist"] > 0) & hist_rising, "f_macd"] = 1.0
    df.loc[(df["MACD_hist"] < 0) & hist_falling, "f_macd"] = -1.0
    df.loc[
        df["MACD_hist"].notna() & df["f_macd"].isna(),
        "f_macd",
    ] = 0.0

    # --- 因子：波动率调整后的20日动量 ---
    return_20d = close.pct_change(20)
    rolling_std = df["daily_return"].rolling(20).std()
    scaled_std = rolling_std * np.sqrt(20)

    momentum_z = return_20d / scaled_std.replace(0, np.nan)

    df["f_momentum_20"] = momentum_z.clip(-1, 1)

    # --- 因子：RSI 均值回归 ---
    df["f_rsi"] = np.nan
    df.loc[df["RSI14"] <= 30, "f_rsi"] = 1.0
    df.loc[df["RSI14"] >= 70, "f_rsi"] = -1.0
    df.loc[df["RSI14"].notna() & df["f_rsi"].isna(), "f_rsi"] = 0.0

    return df


# =========================================================
# 4. 宏观面因子
# =========================================================

def add_macro_factors(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """按公布可用日对齐宏观指标，并生成宏观面因子。"""

    out = df.copy()
    index = out.index

    # --- 实际利率 DFII10 ---
    real_rate = _macro_frame(macro_df, "DFII10")
    out["real_rate"] = _as_of_align(
        _with_available_date(real_rate, MACRO_LAG_DAYS["DFII10"]), index
    )

    real_rate_change = out["real_rate"].diff(20)
    out["real_rate_change_20d"] = real_rate_change

    out["f_real_rate"] = np.where(
        real_rate_change.abs() < DEADBAND_RATE_DAILY,
        0.0,
        -np.sign(real_rate_change),
    )
    out.loc[real_rate_change.isna(), "f_real_rate"] = np.nan

    # --- 名义利率 DGS10 ---
    nominal_rate = _macro_frame(macro_df, "DGS10")
    out["nominal_rate"] = _as_of_align(
        _with_available_date(nominal_rate, MACRO_LAG_DAYS["DGS10"]), index
    )

    nominal_change = out["nominal_rate"].diff(20)
    out["nominal_rate_change_20d"] = nominal_change

    out["f_nominal_rate"] = np.where(
        nominal_change.abs() < DEADBAND_RATE_DAILY,
        0.0,
        -np.sign(nominal_change),
    )
    out.loc[nominal_change.isna(), "f_nominal_rate"] = np.nan

    # --- CPI 同比 ---
    cpi = _macro_frame(macro_df, "CPIAUCSL")

    if not cpi.empty:
        cpi_series = cpi.set_index("observation_date")["value"]
        cpi_yoy = (cpi_series.pct_change(12) * 100).dropna()
        cpi_frame = pd.DataFrame({"value": cpi_yoy}).reset_index()
        cpi_frame = cpi_frame.rename(columns={"index": "observation_date"})
        cpi_frame["observation_date"] = pd.to_datetime(
            cpi_frame["observation_date"]
        )
    else:
        cpi_frame = pd.DataFrame(columns=["observation_date", "value"])

    out["cpi_yoy"] = _as_of_align(
        _with_available_date(cpi_frame, MACRO_LAG_DAYS["CPIAUCSL"]), index
    )

    out["f_inflation"] = (
        (out["cpi_yoy"] - CPI_NEUTRAL_YOY) / CPI_FULL_SCALE_YOY
    ).clip(-1, 1)

    # --- 联邦基金利率 ---
    fed = _macro_frame(macro_df, "FEDFUNDS")

    if not fed.empty:
        fed_series = fed.set_index("observation_date")["value"]
        fed_change = fed_series.diff(3).dropna()
        fed_frame = pd.DataFrame({"value": fed_change}).reset_index()
        fed_frame = fed_frame.rename(columns={"index": "observation_date"})
        fed_frame["observation_date"] = pd.to_datetime(
            fed_frame["observation_date"]
        )
    else:
        fed_frame = pd.DataFrame(columns=["observation_date", "value"])

    out["fedfunds_change_3m"] = _as_of_align(
        _with_available_date(fed_frame, MACRO_LAG_DAYS["FEDFUNDS"]), index
    )

    out["f_fedfunds"] = np.where(
        out["fedfunds_change_3m"].abs() < DEADBAND_RATE_MONTHLY,
        0.0,
        -np.sign(out["fedfunds_change_3m"]),
    )
    out.loc[out["fedfunds_change_3m"].isna(), "f_fedfunds"] = np.nan

    return out


# =========================================================
# 5. 实物面因子
# =========================================================

def normalize_trusted_keys(trusted):
    """
    把「可信记录」入参统一成可匹配的形式。

    支持两种写法：
        None                     不做过滤（调用方自行保证数据可信）
        [(date, source), ...]    精确到来源的可信键
        [date, ...]              只给日期，表示该日期任意来源均可信

    返回 (pairs, dates)；两者皆空集合时表示没有任何可信记录。
    """

    if trusted is None:
        return None, None

    pairs = set()
    dates = set()

    for item in trusted:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            parsed = pd.to_datetime(item[0], errors="coerce")

            if pd.isna(parsed):
                continue

            pairs.add((parsed.normalize(), str(item[1])))
        else:
            parsed = pd.to_datetime(item, errors="coerce")

            if pd.isna(parsed):
                continue

            dates.add(parsed.normalize())

    return pairs, dates


def prepare_inventory(comex_df: pd.DataFrame, trusted=None):
    """
    整理 COMEX 库存：过滤不可信来源、合并多源、剔除跨源冲突日期。

    参数：
        comex_df 需包含 report_date / source，以及 registered、
                 eligible、total 中至少一项
        trusted  可信记录键，见 normalize_trusted_keys

    返回 (整理后的 DataFrame, 诊断信息 dict)。
    """

    diagnosis = {
        "total_reports": 0,
        "trusted_reports": 0,
        "required_reports": INVENTORY_MIN_OBSERVATIONS,
        "usable": False,
        "reason": "",
        "sources": [],
        "metric_used": None,
        "trust_filter_enabled": False,
        "source_discrepancies": [],
        "lookback_days": INVENTORY_LOOKBACK_DAYS,
    }

    if comex_df is None or comex_df.empty:
        diagnosis["reason"] = "数据库中没有 COMEX 库存记录。"
        return pd.DataFrame(), diagnosis

    inventory = comex_df.copy()

    inventory["date"] = pd.to_datetime(
        inventory["report_date"], errors="coerce"
    ).dt.normalize()

    for column in INVENTORY_METRIC_PREFERENCE:
        if column in inventory.columns:
            inventory[column] = pd.to_numeric(
                inventory[column], errors="coerce"
            )
        else:
            inventory[column] = np.nan

    inventory = inventory.dropna(subset=["date"])

    # 至少要有一个可用的库存数值
    has_value = inventory[list(INVENTORY_METRIC_PREFERENCE)].notna().any(axis=1)
    inventory = inventory[has_value]

    diagnosis["total_reports"] = int(len(inventory))

    if inventory.empty:
        diagnosis["reason"] = "库存记录中没有任何可用的数值字段。"
        return pd.DataFrame(), diagnosis

    if "source" not in inventory.columns:
        inventory["source"] = "unknown"

    inventory["source"] = inventory["source"].astype(str)

    # --- 可信度过滤 ---
    pairs, dates = normalize_trusted_keys(trusted)

    trust_note = ""

    if pairs is not None or dates is not None:
        pairs = pairs or set()
        dates = dates or set()

        diagnosis["trust_filter_enabled"] = True

        # 必须按行顺序逐行判定。早期实现用 set(zip(...)) 生成键集合，
        # set 无序且会去重，得到的判定结果与 inventory.index 错位，
        # 等于随机过滤 —— 已作废(REJECTED)的脏数据因此泄漏进了因子计算。
        keys = list(zip(inventory["date"], inventory["source"]))

        keep = pd.Series(
            [
                (key in pairs) or (key[0] in dates)
                for key in keys
            ],
            index=inventory.index,
        )

        inventory = inventory[keep]

        trust_note = (
            f"共 {diagnosis['total_reports']} 条库存记录，"
            f"其中 {len(inventory)} 条处于可信核验状态"
            "（VERIFIED 跨源一致 / ACCEPTED 单源通过全部自动校验）。"
        )
    else:
        diagnosis["trust_filter_enabled"] = False

        trust_note = (
            f"共 {diagnosis['total_reports']} 条库存记录，"
            "未启用核验状态过滤（调用方未提供可信记录键）。"
        )

    diagnosis["trusted_reports"] = int(len(inventory))
    diagnosis["sources"] = sorted(inventory["source"].unique().tolist())

    if inventory.empty:
        diagnosis["reason"] = (
            trust_note
            + "没有任何一条通过核验，库存因子不参与打分。"
            "请先运行采集程序完成跨源验证或人工核验。"
        )
        return pd.DataFrame(), diagnosis

    # --- 多源合并：按来源优先级为每个业务日选出单一取值 ---
    priority_rank = {
        source: index
        for index, source in enumerate(INVENTORY_SOURCE_PRIORITY)
    }

    unknown_rank = len(INVENTORY_SOURCE_PRIORITY)

    inventory = inventory.copy()

    inventory["source_rank"] = inventory["source"].map(
        lambda item: priority_rank.get(item, unknown_rank)
    )

    inventory = inventory.sort_values(
        ["date", "source_rank"], kind="mergesort"
    )

    merged_rows = []
    discrepancies = []

    for date, group in inventory.groupby("date", sort=True):
        primary = group.iloc[0]

        # 所有字段只取自主源的同一行，绝不跨源拼字段。
        # 旧的 cngold 脏数据正是「注册/非注册/总量来自不同日期」拼接而成，
        # 加总自洽却与权威源逐字段对不上，这类错误不能在新代码里重演。
        record = {
            "date": date,
            "source": str(primary["source"]),
            "registered": (
                float(primary["registered"])
                if pd.notna(primary["registered"])
                else np.nan
            ),
            "eligible": (
                float(primary["eligible"])
                if pd.notna(primary["eligible"])
                else np.nan
            ),
            "total": (
                float(primary["total"])
                if pd.notna(primary["total"])
                else np.nan
            ),
        }

        if "source_url" in group.columns:
            record["source_url"] = next(
                (
                    item
                    for item in group["source_url"]
                    if isinstance(item, str) and item
                ),
                None,
            )

        # 同一天存在多个来源时，记录数值分歧供诊断，但不因此弃用该日：
        # 两源对「同一观测挂在哪一天」的约定相差一天（周频源常用发布日标注），
        # 同日精确比对会把这种约定差异误判成数值冲突。
        # 数据质量的裁决已在采集层用 ±1 天窗口完成并写入核验状态。
        if len(group) > 1:
            totals = [
                float(value)
                for value in group["total"].dropna().tolist()
            ]

            if len(totals) > 1:
                spread = max(totals) - min(totals)
                base = max(totals)

                if base > 0 and spread / base > INVENTORY_SOURCE_TOLERANCE:
                    discrepancies.append(
                        {
                            "report_date": date.strftime("%Y-%m-%d"),
                            "totals": {
                                str(item["source"]): float(item["total"])
                                for _, item in group.iterrows()
                                if pd.notna(item["total"])
                            },
                            "spread_pct": round(spread / base * 100, 4),
                            "used_source": record["source"],
                        }
                    )

        merged_rows.append(record)

    if not merged_rows:
        diagnosis["reason"] = "库存记录合并后为空，无法计算因子。"
        diagnosis["source_discrepancies"] = discrepancies
        return pd.DataFrame(), diagnosis

    merged = pd.DataFrame(merged_rows).sort_values("date")

    diagnosis["source_discrepancies"] = discrepancies

    # --- 选择计算口径：优先注册库存，覆盖不足则回退总库存 ---
    metric = None

    for candidate in INVENTORY_METRIC_PREFERENCE:
        coverage = int(merged[candidate].notna().sum())

        if coverage >= INVENTORY_MIN_OBSERVATIONS:
            metric = candidate
            break

    if metric is None:
        diagnosis["reason"] = (
            trust_note
            + f" 但注册库存与总库存的可用观测数都不足 "
            f"{INVENTORY_MIN_OBSERVATIONS} 条，"
            "无法可靠计算库存变化因子，实物面因子暂缺。"
        )
        # 与其他不可用分支保持一致，统一返回空表
        return pd.DataFrame(), diagnosis

    merged = merged[merged[metric].notna()].copy()

    # 统一口径列：下游只需读 metric_value，不必关心选中的是哪个指标
    merged["metric_value"] = merged[metric]

    diagnosis["metric_used"] = metric
    diagnosis["usable"] = True

    # 未启用可信过滤时不能把这些记录称作「可信」，否则会掩盖数据来源未经核验的事实
    if diagnosis.get("trust_filter_enabled"):
        diagnosis["reason"] = (
            f"采用 {metric} 口径，{len(merged)} 条可信记录，"
            f"回看窗口 {INVENTORY_LOOKBACK_DAYS} 个日历天。"
        )
    else:
        diagnosis["reason"] = (
            f"采用 {metric} 口径，{len(merged)} 条记录"
            "（未启用核验状态过滤，记录可信度未经核验），"
            f"回看窗口 {INVENTORY_LOOKBACK_DAYS} 个日历天。"
        )

    if discrepancies:
        diagnosis["reason"] += (
            f" 另有 {len(discrepancies)} 个业务日存在跨源数值分歧，"
            f"已按来源优先级取 {INVENTORY_SOURCE_PRIORITY[0]} 的值。"
        )

    return merged, diagnosis


def add_inventory_factors_from_frame(df, inventory_frame, diagnosis):
    """
    基于已整理好的库存数据生成实物面因子。

    与 add_inventory_factors 的区别：本函数不重复调用 prepare_inventory，
    便于主入口复用同一次整理结果，同时取得诊断信息。
    """

    out = df.copy()
    index = out.index

    out["comex_registered"] = np.nan
    out["comex_registered_change_pct"] = np.nan
    out["f_inventory"] = np.nan

    if inventory_frame is None or inventory_frame.empty:
        return out

    if not diagnosis or not diagnosis.get("usable"):
        return out

    # metric_value 是 prepare_inventory 选定的统一口径列
    series = inventory_frame[["date", "metric_value"]].copy()

    series["date"] = pd.to_datetime(series["date"])

    series = series.sort_values("date").reset_index(drop=True)

    # 基期匹配：对每个观测日，取「date - LOOKBACK_DAYS 当天或之前」最近的观测。
    # 左右表方向不能弄反——若把右表键设为 date-LOOKBACK_DAYS 并配上当天的值，
    # backward 匹配会取到未来观测，构成 look-ahead bias。
    left = pd.DataFrame(
        {
            "date": series["date"],
            "metric_value": series["metric_value"],
            "baseline_target": series["date"]
            - pd.Timedelta(days=INVENTORY_LOOKBACK_DAYS),
        }
    )

    right = series.rename(
        columns={
            "date": "baseline_date",
            "metric_value": "metric_value_baseline",
        }
    )

    merged = pd.merge_asof(
        left.sort_values("baseline_target"),
        right.sort_values("baseline_date"),
        left_on="baseline_target",
        right_on="baseline_date",
        direction="backward",
    )

    merged = merged.sort_values("date").reset_index(drop=True)

    merged = merged.dropna(subset=["metric_value", "metric_value_baseline"])

    if merged.empty:
        return out

    merged["change_pct"] = (
        (merged["metric_value"] - merged["metric_value_baseline"])
        / merged["metric_value_baseline"]
    )

    # 库存下降视为利多（实物需求走强），故取负号
    merged["score"] = (
        -merged["change_pct"] / DEADBAND_INVENTORY_PCT / 4
    ).clip(-1, 1)

    merged.loc[
        merged["change_pct"].abs() < DEADBAND_INVENTORY_PCT,
        "score",
    ] = 0.0

    merged["available_date"] = (
        merged["date"] + pd.Timedelta(days=INVENTORY_LAG_DAYS)
    )

    # 水平值与变化率分开对齐：
    # 水平值在首个观测日即可用，不应被「必须有基期」的条件限制；
    # 变化率与得分则需要基期存在才有意义。
    series = series.copy()

    series["available_date"] = (
        series["date"] + pd.Timedelta(days=INVENTORY_LAG_DAYS)
    )

    align_level = _as_of_align(
        series[["available_date", "metric_value"]].rename(
            columns={"metric_value": "value"}
        ),
        index,
    )

    align_change = _as_of_align(
        merged[["available_date", "change_pct"]].rename(
            columns={"change_pct": "value"}
        ),
        index,
    )

    align_score = _as_of_align(
        merged[["available_date", "score"]].rename(
            columns={"score": "value"}
        ),
        index,
    )

    # 列名保留 comex_registered 以兼容既有页面与导出；
    # 实际口径可能是 registered 或 total，由 metric_used 标明。
    out["comex_registered"] = align_level
    out["comex_registered_change_pct"] = align_change * 100
    out["f_inventory"] = align_score

    out.attrs["inventory_metric_used"] = diagnosis.get("metric_used")

    return out


def add_inventory_factors(
    df: pd.DataFrame,
    comex_df: pd.DataFrame,
    trusted=None,
) -> pd.DataFrame:
    """
    按公布可用日对齐 COMEX 库存，并生成实物面因子。

    变化幅度按「日历天回看窗口」计算，因此日频与周频数据都表示
    同一时间跨度（默认一周）的变化，不会因数据频率不同而失真。
    """

    inventory, diagnosis = prepare_inventory(comex_df, trusted)

    return add_inventory_factors_from_frame(df, inventory, diagnosis)


# =========================================================
# 6. 综合打分
# =========================================================

def add_composite_score(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """按可用权重归一化，计算综合得分。"""

    out = df.copy()

    factor_columns = [
        column for column in DEFAULT_WEIGHTS if column in out.columns
    ]

    effective_weights = {
        column: float(weights.get(column, DEFAULT_WEIGHTS[column]))
        for column in factor_columns
    }

    total_weight = sum(abs(value) for value in effective_weights.values())

    if total_weight <= 0:
        out["score"] = np.nan
        out["active_factor_count"] = 0
        out["active_weight"] = 0.0
        return out

    weighted_sum = pd.Series(0.0, index=out.index)
    active_weight = pd.Series(0.0, index=out.index)
    active_count = pd.Series(0.0, index=out.index)

    for column, weight in effective_weights.items():
        values = pd.to_numeric(out[column], errors="coerce")
        available = values.notna()

        contribution = values.fillna(0.0) * weight

        weighted_sum = weighted_sum + contribution.where(available, 0.0)
        active_weight = active_weight + np.where(available, abs(weight), 0.0)
        active_count = active_count + available.astype(float)

        out[f"w_{column}"] = contribution.where(available, np.nan)

    out["score"] = np.where(
        active_weight > 0,
        weighted_sum / active_weight.replace(0, np.nan),
        np.nan,
    )

    out["active_factor_count"] = active_count
    out["active_weight"] = active_weight

    return out


def add_position(
    df: pd.DataFrame,
    long_threshold: float = 0.15,
    short_threshold: float = -0.15,
    allow_short: bool = False,
    use_vol_target: bool = True,
    target_vol: float = 12.0,
) -> pd.DataFrame:
    """把综合得分转换为仓位，并可选按波动率目标缩放。"""

    out = df.copy()

    position = pd.Series(0.0, index=out.index)

    position[out["score"] >= long_threshold] = 1.0

    if allow_short:
        position[out["score"] <= short_threshold] = -1.0

    out["signal"] = position

    if use_vol_target:
        realized = out["vol_20d_annual"]

        scale = (target_vol / realized.replace(0, np.nan)).clip(upper=1.0)

        # 波动率尚未估出时缩放系数记 0，即「无风险信息则不开仓」，
        # 不能退化为满仓，否则样本前段会承担未经缩放的风险敞口。
        out["vol_scale"] = scale.fillna(0.0)
    else:
        out["vol_scale"] = 1.0

    out["position"] = position * out["vol_scale"]

    return out


# =========================================================
# 7. 对外主入口
# =========================================================

def build_factors(
    gold_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    comex_df: pd.DataFrame,
    weights: dict = None,
    trusted_inventory=None,
    long_threshold: float = 0.15,
    short_threshold: float = -0.15,
    allow_short: bool = False,
    use_vol_target: bool = True,
    target_vol: float = 12.0,
) -> pd.DataFrame:
    """
    生成完整的因子表，索引为交易日。

    trusted_inventory：可信库存记录键，可为
        [(date, source), ...] 或 [date, ...]，见 normalize_trusted_keys；
        传 None 表示不按可信度过滤库存来源。
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS

    daily = to_daily_gold(gold_df)

    if daily.empty:
        return pd.DataFrame()

    df = add_technical_factors(daily)
    df = add_macro_factors(df, macro_df)

    # prepare_inventory 只做一次，因子与诊断共用同一结果
    inventory_frame, diagnosis = prepare_inventory(
        comex_df, trusted_inventory
    )

    df = add_inventory_factors_from_frame(df, inventory_frame, diagnosis)

    # 把库存数据可用性诊断挂在 attrs 上，供页面提示缺失原因
    df.attrs["inventory_diagnosis"] = diagnosis

    df = add_composite_score(df, weights)
    df = add_position(
        df,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        allow_short=allow_short,
        use_vol_target=use_vol_target,
        target_vol=target_vol,
    )

    return df


def describe_latest_factors(df: pd.DataFrame) -> list:
    """输出最新一天的因子明细，供页面表格与提示词使用。"""

    if df is None or df.empty:
        return []

    latest = df.iloc[-1]

    rows = []

    for column in DEFAULT_WEIGHTS:
        if column not in df.columns:
            continue

        value = latest.get(column, np.nan)

        rows.append(
            {
                "factor": column,
                "label": FACTOR_LABELS[column],
                "value": None if pd.isna(value) else round(float(value), 3),
                "available": bool(pd.notna(value)),
            }
        )

    return rows


if __name__ == "__main__":
    from database.db import get_connection

    with get_connection() as conn:
        gold = pd.read_sql_query(
            "SELECT * FROM gold_prices WHERE symbol='XAU/USD' "
            "AND source='XAUS' ORDER BY timestamp",
            conn,
        )
        macro = pd.read_sql_query(
            "SELECT * FROM macro_data WHERE source='FRED' "
            "ORDER BY observation_date",
            conn,
        )
        try:
            comex = pd.read_sql_query(
                "SELECT * FROM comex_inventory ORDER BY report_date", conn
            )
        except Exception:
            comex = pd.DataFrame()

    result = build_factors(gold, macro, comex)

    print("因子表构建完成，行数：", len(result))
    print("最新得分：", result["score"].iloc[-1])
    print("最新仓位：", result["position"].iloc[-1])

    for row in describe_latest_factors(result):
        print(row)

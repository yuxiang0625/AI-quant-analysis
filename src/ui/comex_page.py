# 本文件负责 COMEX 库存页面的渲染与人工核验交互
"""
COMEX 库存页面。

为什么必须按来源分别处理：
    库存数据现在有多个来源，频率与字段都不同：
        eastmoney   日频，只有 total，覆盖长
        metalcharts 周频，含 registered / eligible 拆分
        cngold      旧的搜索解析来源，已证伪并标记 REJECTED
    把它们混在一张表里直接 diff()，会得到跨来源的错误变化量
    （相邻两行未必是相邻业务日）。因此图表、变化量与联动分析
    都必须先锁定单一来源。

为什么需要人工核验工作台：
    自动核验只能覆盖「与第二来源在同一业务日数值一致」的记录（VERIFIED）
    和「单源通过全部自动校验」的记录（ACCEPTED）。跨源冲突、极端变化、
    日期落在周末等告警会落到 REVIEW，需要人对照原始来源判断。
    此前质量表里存在「用户已在界面确认」的记录，但页面上没有任何入口，
    核验链路是断的——本模块补上这个入口，并写入可追溯的审计信息。
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.db import get_connection


# =========================================================
# 1. 常量
# =========================================================

PAPER_BG = "#0B1120"
PLOT_BG = "#111827"
GOLD = "#F5C451"
MUTED = "#9CA3AF"

STATUS_VERIFIED = "VERIFIED"
STATUS_ACCEPTED = "ACCEPTED"
STATUS_REVIEW = "REVIEW"
STATUS_REJECTED = "REJECTED"

# 旧库可能残留 UNVERIFIED 等历史取值，统一按待复核处理
STATUS_LABELS = {
    STATUS_VERIFIED: "已验证（跨源一致）",
    STATUS_ACCEPTED: "已接受（单源通过校验）",
    STATUS_REVIEW: "待复核",
    STATUS_REJECTED: "已作废",
    "UNVERIFIED": "待复核（历史取值）",
}

STATUS_COLORS = {
    STATUS_VERIFIED: "#22C55E",
    STATUS_ACCEPTED: "#60A5FA",
    STATUS_REVIEW: "#F59E0B",
    STATUS_REJECTED: "#EF4444",
    "UNVERIFIED": "#F59E0B",
}

# 参与量化分析的状态
TRUSTED_STATUSES = (STATUS_VERIFIED, STATUS_ACCEPTED)

SOURCE_LABELS = {
    "eastmoney": "东方财富（日频·总库存）",
    "metalcharts": "MetalCharts（周频·含注册拆分）",
    "cngold": "金投网文章解析（已作废）",
}

# 库存公布滞后：CME 业务日次一日才可获取
INVENTORY_LAG_DAYS = 1


# =========================================================
# 2. 辅助函数
# =========================================================

def _fmt(value, digits=0, suffix="", dash="—"):
    """格式化数值，缺失时统一显示破折号。"""

    if value is None or pd.isna(value):
        return dash

    return f"{float(value):,.{digits}f}{suffix}"


def _source_label(source):
    """把来源代号翻译成中文说明。"""

    return SOURCE_LABELS.get(str(source), str(source))


def _base_layout(fig, height=430):
    """统一深色主题布局。"""

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )

    return fig


def normalize_status(status):
    """把历史状态取值归一到四级分级。"""

    if status is None or pd.isna(status):
        return None

    text = str(status).strip().upper()

    if text in (STATUS_VERIFIED, STATUS_ACCEPTED, STATUS_REVIEW, STATUS_REJECTED):
        return text

    # UNVERIFIED 等旧取值一律视为待复核
    return STATUS_REVIEW


def is_rejected_source(single_source_df):
    """
    判断该来源是否整体处于「已作废」状态。

    只要还有任意一条记录不是 REJECTED，就不视为整体作废，
    以便对部分作废的来源仍可查看其可信部分。
    """

    if single_source_df is None or single_source_df.empty:
        return False

    frame = single_source_df

    if "status" not in frame.columns:
        return False

    statuses = frame["status"].map(normalize_status)

    return bool((statuses == STATUS_REJECTED).all())


def pick_primary_source(comex_df):
    """
    选出默认展示的来源：优先有可信记录的来源，其次记录数多者。

    整体处于 REJECTED（已作废）的来源不得作为默认选择，
    否则页面一打开就用已证伪的数据做趋势与联动分析。

    返回来源代号；无可用来源时返回 None。
    """

    if comex_df is None or comex_df.empty:
        return None

    frame = comex_df.copy()

    if "status" not in frame.columns:
        frame["status"] = None

    frame["norm_status"] = frame["status"].map(normalize_status)

    frame["trusted"] = frame["norm_status"].isin(TRUSTED_STATUSES)
    frame["rejected"] = frame["norm_status"] == STATUS_REJECTED

    ranked = (
        frame.groupby("source")
        .agg(
            trusted_count=("trusted", "sum"),
            rejected_count=("rejected", "sum"),
            total_count=("trusted", "size"),
        )
    )

    # 全部记录都已作废的来源排除在默认选择之外
    not_rejected = ranked[
        ranked["rejected_count"] < ranked["total_count"]
    ]

    candidates = not_rejected if not not_rejected.empty else ranked

    candidates = candidates.sort_values(
        ["trusted_count", "total_count"], ascending=False
    )

    if candidates.empty:
        return None

    return str(candidates.index[0])


# =========================================================
# 3. 人工核验写回
# =========================================================

def record_manual_review(report_date, source, status, note, source_url=None):
    """
    写入人工核验结论。

    参数：
        report_date  CME 业务日（YYYY-MM-DD）
        source       来源代号
        status       新状态，只允许 VERIFIED / REVIEW / REJECTED
        note         核验说明，会写入 reason 以便日后追溯
        source_url   来源网址，缺失时保留原值

    返回写入的行数。
    """

    allowed = (STATUS_VERIFIED, STATUS_REVIEW, STATUS_REJECTED)

    if status not in allowed:
        raise ValueError(f"不支持的人工核验状态：{status}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reason = f"人工核验（{now}）：{note}".strip()

    if status == STATUS_VERIFIED:
        reason += "｜人工确认与原始来源一致"
    elif status == STATUS_REJECTED:
        reason += "｜人工判定该记录不可用，不参与任何分析"

    reviewed_at = now if status == STATUS_VERIFIED else None

    with get_connection() as conn:
        if source_url:
            conn.execute(
                """
                INSERT INTO comex_inventory_quality (
                    report_date, source, source_url, status,
                    reason, flags_json, reviewed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(report_date, source) DO UPDATE SET
                    source_url = COALESCE(excluded.source_url, source_url),
                    status = excluded.status,
                    reason = excluded.reason,
                    reviewed_at = excluded.reviewed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    report_date, source, source_url, status,
                    reason, reviewed_at, now,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO comex_inventory_quality (
                    report_date, source, source_url, status,
                    reason, flags_json, reviewed_at, updated_at
                ) VALUES (?, ?, NULL, ?, ?, NULL, ?, ?)
                ON CONFLICT(report_date, source) DO UPDATE SET
                    status = excluded.status,
                    reason = excluded.reason,
                    reviewed_at = excluded.reviewed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    report_date, source, status,
                    reason, reviewed_at, now,
                ),
            )

        conn.commit()

    return 1


# =========================================================
# 4. 各区块渲染
# =========================================================

def render_source_overview(comex_df):
    """渲染数据来源与核验状态概况。"""

    st.subheader("一、数据来源与核验概况")

    frame = comex_df.copy()

    if "status" not in frame.columns:
        frame["status"] = None

    frame["norm_status"] = frame["status"].map(normalize_status)

    summary = (
        frame.groupby(["source", "norm_status"], dropna=False)
        .size()
        .reset_index(name="records")
    )

    rows = []

    for source in sorted(frame["source"].unique()):
        subset = frame[frame["source"] == source]

        trusted = int(
            subset["norm_status"].isin(TRUSTED_STATUSES).sum()
        )

        dates = pd.to_datetime(subset["report_date"], errors="coerce")

        rows.append(
            {
                "来源": _source_label(source),
                "记录数": int(len(subset)),
                "可信记录": trusted,
                "日期范围": (
                    f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}"
                    if dates.notna().any()
                    else "—"
                ),
                "含注册拆分": "是" if subset["registered"].notna().any() else "否",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    total = int(len(frame))
    trusted_total = int(frame["norm_status"].isin(TRUSTED_STATUSES).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("库存记录总数", f"{total:,}")
    col2.metric("可信记录数", f"{trusted_total:,}")
    col3.metric(
        "可信占比",
        f"{trusted_total / total * 100:.1f}%" if total else "—",
    )

    st.caption(
        "可信 = VERIFIED（与第二来源在同一 CME 业务日数值一致）"
        "或 ACCEPTED（单源通过全部自动校验）。"
        "REVIEW 与 REJECTED 不参与量化打分。"
    )

    # 状态分布
    status_counts = (
        frame["norm_status"].fillna("(无状态)").value_counts()
    )

    status_frame = pd.DataFrame(
        {
            "核验状态": [
                STATUS_LABELS.get(item, str(item))
                for item in status_counts.index
            ],
            "记录数": status_counts.values,
        }
    )

    st.dataframe(status_frame, use_container_width=True, hide_index=True)


def render_trend_chart(single_source_df, source):
    """渲染单一来源的库存趋势图。"""

    st.subheader("二、库存趋势")

    if is_rejected_source(single_source_df):
        st.error(
            f"⚠️ {_source_label(source)} 已被标记为「已作废」："
            "其日期口径比 CME 业务日早一天，且同一行内的注册 / 非注册 / 总量"
            "存在跨日期拼接错误。以下图表仅供审计追溯，"
            "不得用于任何分析或决策。"
        )

    frame = single_source_df.sort_values("report_date")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frame["report_date"],
            y=frame["total"],
            mode="lines",
            name="总库存",
            line=dict(color=GOLD, width=2.4),
            hovertemplate=(
                "业务日：%{x|%Y-%m-%d}<br>总库存：%{y:,.0f} oz<extra></extra>"
            ),
        )
    )

    has_split = frame["registered"].notna().any()

    if has_split:
        fig.add_trace(
            go.Scatter(
                x=frame["report_date"],
                y=frame["registered"],
                mode="lines",
                name="注册库存",
                line=dict(color="#22C55E", width=1.8),
                hovertemplate=(
                    "业务日：%{x|%Y-%m-%d}<br>注册库存：%{y:,.0f} oz"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=frame["report_date"],
                y=frame["eligible"],
                mode="lines",
                name="非注册库存",
                line=dict(color="#60A5FA", width=1.8),
                hovertemplate=(
                    "业务日：%{x|%Y-%m-%d}<br>非注册库存：%{y:,.0f} oz"
                    "<extra></extra>"
                ),
            )
        )
    else:
        st.info(
            f"{_source_label(source)} 只提供总库存，"
            "不含注册 / 非注册拆分。拆分数据来自 MetalCharts 周频来源。"
        )

    _base_layout(fig, height=470)

    fig.update_layout(
        title=f"COMEX 黄金库存趋势 · {_source_label(source)}",
        xaxis_title="CME 业务日",
        yaxis_title="库存（金衡盎司）",
    )

    st.plotly_chart(fig, use_container_width=True)

    latest = frame.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "最新业务日",
        f"{latest['report_date']:%Y-%m-%d}",
    )

    col2.metric("总库存", f"{_fmt(latest['total'])} oz")

    col3.metric(
        "注册库存",
        f"{_fmt(latest['registered'])} oz" if has_split else "该来源不提供",
    )

    col4.metric(
        "非注册库存",
        f"{_fmt(latest['eligible'])} oz" if has_split else "该来源不提供",
    )


def render_cross_source_check(comex_df):
    """渲染多来源的一致性对照。"""

    st.subheader("三、跨源一致性对照")

    # 直接复用采集层的 cross_validate，保证页面结论与入库时的判定完全一致。
    # 页面若另写一套「同日精确比对」，会把周频源用发布日标注、
    # 因而与日频源相差一天的正常情况误判为分歧（实测会多报 24 个），
    # 与采集层「冲突 0 组」的结论自相矛盾。
    from src.data_sources.comex_inventory_api import (
        CROSS_SOURCE_DATE_WINDOW_DAYS,
        cross_validate,
    )

    frame = comex_df.copy()

    structured = frame[
        frame["source"].isin(["eastmoney", "metalcharts"])
    ]

    if structured.empty:
        st.info("暂无可用于交叉比对的结构化来源数据。")
        return

    rows = [
        {
            "source": str(row["source"]),
            "report_date": pd.Timestamp(row["report_date"]).strftime("%Y-%m-%d"),
            "total": (
                float(row["total"]) if pd.notna(row["total"]) else None
            ),
        }
        for _, row in structured.iterrows()
        if pd.notna(row["report_date"]) and pd.notna(row["total"])
    ]

    passed_pairs, conflicts = cross_validate(rows)

    passed_dates = {item[0] for item in passed_pairs}
    conflict_dates = {item["report_date"] for item in conflicts}

    col1, col2, col3 = st.columns(3)

    col1.metric("参与比对的业务日", f"{len(passed_dates | conflict_dates):,}")
    col2.metric("数值一致", f"{len(passed_dates):,}")
    col3.metric("存在分歧", f"{len(conflict_dates):,}")

    st.caption(
        f"一致判定标准：在 ±{CROSS_SOURCE_DATE_WINDOW_DAYS} 天窗口内，"
        "另一来源存在总库存相对偏差不超过 0.01% 的观测。"
        "之所以允许窗口，是因为周频源常用发布日（周五）标注，"
        "其数值对应日频源前一业务日的库存，两者相差一天属于口径约定差异，"
        "不是数据错误。真正无法解释的分歧会被标记为待复核。"
    )

    if conflicts:
        st.warning(f"发现 {len(conflicts)} 处跨源分歧，已标记为待复核。")

        conflict_table = pd.DataFrame(
            [
                {
                    "业务日": item["report_date"],
                    "来源": item["source"],
                    "对照来源": item["compared_source"],
                    "总库存(oz)": _fmt(
                        item["totals"].get(item["source"])
                    ),
                    "最大偏差(%)": _fmt(item["spread_pct"], 4),
                }
                for item in conflicts
            ]
        )

        st.dataframe(conflict_table, use_container_width=True, hide_index=True)
    else:
        st.success(
            "两个来源在所有重叠观测上一致，没有发现需要复核的数值分歧。"
        )

    # 展示重叠业务日的逐日对照
    pivot = structured.pivot_table(
        index="report_date",
        columns="source",
        values="total",
        aggfunc="last",
    )

    overlap = pivot.dropna()

    if not overlap.empty:
        st.markdown("**同一业务日的逐日对照**")

        overlap = overlap.copy()

        columns = list(overlap.columns)

        first, second = columns[0], columns[1]

        overlap["差额(oz)"] = overlap[first] - overlap[second]
        overlap["相对偏差(%)"] = (
            overlap["差额(oz)"].abs()
            / overlap[[first, second]].max(axis=1)
            * 100
        )

        st.dataframe(
            overlap.sort_index(ascending=False).round(3),
            use_container_width=True,
        )

        st.caption(
            "此表按完全相同的业务日对齐，仅用于逐日查看数值差异；"
            "上方的一致性判定采用 ±1 天窗口，两者口径不同属预期。"
        )


def render_daily_changes(single_source_df, source):
    """渲染单一来源的库存变化，变化量已在加载时按来源分组计算。"""

    st.subheader("四、库存变化")

    frame = single_source_df.sort_values("report_date", ascending=False)

    has_split = frame["registered"].notna().any()

    columns = {
        "report_date": "CME 业务日",
        "total": "总库存(oz)",
        "total_change": "总库存变化(oz)",
    }

    if has_split:
        columns.update(
            {
                "registered": "注册库存(oz)",
                "registered_change": "注册库存变化(oz)",
                "eligible": "非注册库存(oz)",
                "eligible_change": "非注册库存变化(oz)",
            }
        )

    display = frame[list(columns.keys())].rename(columns=columns)

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption(
        f"变化量按 {_source_label(source)} 内部相邻业务日计算，"
        "不跨来源比较，避免不同频率数据产生错误的差值。"
    )


def render_gold_linkage(single_source_df, gold_df, source):
    """渲染库存变化与金价的联动分析。"""

    st.subheader("五、库存变化与黄金价格")

    if is_rejected_source(single_source_df):
        st.error(
            f"{_source_label(source)} 已被标记为「已作废」，"
            "不参与任何相关性计算与量化打分。请切换到可信来源。"
        )
        return

    if gold_df is None or gold_df.empty:
        st.info("黄金行情表暂无数据，暂时无法进行联动分析。")
        return

    gold_daily = gold_df[["timestamp", "close"]].copy()

    gold_daily["date"] = (
        gold_daily["timestamp"].dt.tz_convert(None).dt.normalize()
    )

    gold_daily = (
        gold_daily.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )

    gold_daily["gold_return_pct"] = (
        gold_daily["close"].pct_change() * 100
    )

    inventory = single_source_df[["report_date", "total"]].copy()

    inventory["date"] = pd.to_datetime(
        inventory["report_date"], errors="coerce"
    ).dt.normalize()

    inventory = inventory.dropna(subset=["date"]).sort_values("date")

    # 变化量按同一来源的相邻业务日计算
    inventory["inventory_change"] = inventory["total"].diff()

    # 库存按公布滞后对齐：业务日次一日才可用，避免未来函数
    inventory["available_date"] = (
        inventory["date"] + pd.Timedelta(days=INVENTORY_LAG_DAYS)
    )

    inventory = inventory.sort_values("available_date")

    # 每个库存观测只配对「其可用日当天或之后第一个行情日」，
    # 一对一匹配。若用 merge_asof 让每个行情日都回溯到最近库存，
    # 稀疏的库存值会被重复匹配到多个交易日，算出的相关性是失真的。
    right = inventory[
        ["available_date", "date", "total", "inventory_change"]
    ].rename(
        columns={
            "available_date": "available_date",
            "date": "business_date",
        }
    )

    gold_sorted = gold_daily[
        ["date", "close", "gold_return_pct"]
    ].sort_values("date")

    paired_rows = []

    for _, item in right.iterrows():
        available = item["available_date"]

        if pd.isna(available) or pd.isna(item["inventory_change"]):
            continue

        matched = gold_sorted[gold_sorted["date"] >= available]

        if matched.empty:
            continue

        first = matched.iloc[0]

        paired_rows.append(
            {
                "date": first["date"],
                "business_date": item["business_date"],
                "close": first["close"],
                "gold_return_pct": first["gold_return_pct"],
                "total": item["total"],
                "inventory_change": item["inventory_change"],
            }
        )

    paired = pd.DataFrame(paired_rows)

    if paired.empty:
        st.info(
            f"{_source_label(source)} 的库存可用日之后没有对应的行情交易日，"
            "暂时无法进行联动分析。"
        )
        return

    paired = paired.dropna(subset=["gold_return_pct", "inventory_change"])

    # 多个库存观测可能映射到同一交易日，只保留该日最近的一条，
    # 确保配对结果严格一对一，相关系数不被重复样本主导。
    paired = (
        paired.sort_values(["date", "business_date"])
        .drop_duplicates(subset=["date"], keep="last")
    )

    if len(paired) < 5:
        st.info(
            f"黄金行情与 {_source_label(source)} 的有效配对不足 5 个，"
            "暂不计算相关系数。"
        )
        return

    corr = paired["gold_return_pct"].corr(paired["inventory_change"])

    col1, col2 = st.columns(2)

    col1.metric("同期变化相关系数", _fmt(corr, 3))
    col2.metric("有效配对数", f"{len(paired):,}")

    st.caption(
        "库存已按「CME 业务日 + 1 天」的公布滞后对齐，不使用未来数据。"
        "相关性不代表因果关系，且样本区间较短，结果仅供探索性参考。"
    )

    fig = go.Figure(
        go.Scatter(
            x=paired["inventory_change"],
            y=paired["gold_return_pct"],
            mode="markers",
            name="同期变化",
            # date 是对齐后的可用日（业务日 + 公布滞后），
            # business_date 才是库存真实的 CME 业务日，标签应显示后者
            text=paired["business_date"].dt.strftime("%Y-%m-%d"),
            customdata=paired["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate=(
                "库存业务日：%{text}<br>"
                "对齐到行情日：%{customdata}<br>"
                "库存变化：%{x:,.0f} oz<br>"
                "黄金日收益率：%{y:.3f}%<extra></extra>"
            ),
            marker=dict(size=6, opacity=0.7),
        )
    )

    fig.update_layout(
        title=f"COMEX 总库存变化与黄金日收益率 · {_source_label(source)}",
        xaxis_title="总库存变化（盎司）",
        yaxis_title="黄金日收益率（%）",
        height=440,
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_review_workbench(single_source_df, source):
    """渲染人工核验工作台。"""

    st.subheader("六、人工核验工作台")

    st.caption(
        "自动核验只能覆盖跨源一致（VERIFIED）与单源通过校验（ACCEPTED）的记录。"
        "存在告警的记录会落到「待复核」，需要你对照原始来源判断后在此确认。"
    )

    frame = single_source_df.copy()

    if "status" not in frame.columns:
        frame["status"] = None

    frame["norm_status"] = frame["status"].map(normalize_status)

    pending = frame[frame["norm_status"] == STATUS_REVIEW]

    if pending.empty:
        st.success(
            f"{_source_label(source)} 没有待复核记录。"
        )
        return

    st.warning(
        f"{_source_label(source)} 有 {len(pending)} 条待复核记录。"
    )

    batch_options = [10, 25, 50, 100, len(pending)]

    # 记录数可能与预设值重复，需去重并保持升序，否则控件会报错
    batch_options = sorted(
        {option for option in batch_options if option > 0}
    )

    limit = st.select_slider(
        "本次处理多少条",
        options=batch_options,
        value=min(25, batch_options[-1]),
    )

    batch = pending.sort_values("report_date", ascending=False).head(limit)

    for _, item in batch.iterrows():
        report_date = pd.to_datetime(item["report_date"]).strftime("%Y-%m-%d")

        with st.container():
            st.markdown(
                f"**{report_date}** · "
                f"总库存 {_fmt(item['total'])} oz · "
                f"注册 {_fmt(item['registered'])} oz · "
                f"非注册 {_fmt(item['eligible'])} oz"
            )

            url = item.get("source_url")

            if isinstance(url, str) and url:
                st.markdown(f"原始来源：[{url}]({url})")
            else:
                st.caption("该记录缺少来源网址，无法溯源核对。")

            cols = st.columns([3, 1, 1])

            with cols[0]:
                note = st.text_input(
                    "核验说明",
                    key=f"note_{source}_{report_date}",
                    placeholder="例如：已对照东方财富原始页面，数值与业务日均一致",
                )

            with cols[1]:
                confirm = st.button(
                    "✅ 标记已验证",
                    key=f"ok_{source}_{report_date}",
                    use_container_width=True,
                )

            with cols[2]:
                reject = st.button(
                    "🚫 标记作废",
                    key=f"no_{source}_{report_date}",
                    use_container_width=True,
                )

            if confirm or reject:
                if not note.strip():
                    st.error("请先填写核验说明，否则日后无法追溯判断依据。")
                else:
                    status = STATUS_VERIFIED if confirm else STATUS_REJECTED

                    record_manual_review(
                        report_date=report_date,
                        source=str(source),
                        status=status,
                        note=note.strip(),
                        source_url=url if isinstance(url, str) else None,
                    )

                    st.success(f"已写入：{report_date} → {STATUS_LABELS[status]}")

                    st.cache_data.clear()
                    st.rerun()

            st.divider()


def render_raw_table(comex_df):
    """渲染原始数据与下载。"""

    st.subheader("七、库存原始数据")

    frame = comex_df.sort_values(
        ["source", "report_date"], ascending=[True, False]
    )

    st.dataframe(frame, use_container_width=True, hide_index=True)

    st.download_button(
        "下载 COMEX 库存 CSV",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="aurum_comex_inventory.csv",
        mime="text/csv",
    )


# =========================================================
# 5. 页面主入口
# =========================================================

def render_comex_page(comex_df, gold_df):
    """渲染整个 COMEX 库存页面。"""

    st.title("COMEX 黄金库存分析")

    st.caption(
        "COMEX GOLD INVENTORY | 多源库存、跨源一致性与数据核验"
    )

    if comex_df is None or comex_df.empty:
        st.warning("数据库中暂无可读取的 COMEX 黄金库存数据。")
        st.info(
            "请运行以下命令抓取结构化库存数据：\n\n"
            "```\n"
            "python -m src.data_sources.comex_inventory_api           # 预览\n"
            "python -m src.data_sources.comex_inventory_api --commit   # 入库\n"
            "```\n\n"
            "采集时会自动做日期偏移自检与跨源交叉验证。"
        )
        return

    st.info(
        "本页库存日期为 CME 业务日。数据来源已从「搜索文章 + 正则解析」"
        "改为结构化接口直读，并在入库前完成日期偏移自检与跨源交叉验证。"
        "旧的 cngold 数据已被证明日期口径错误，标记为「已作废」保留审计，"
        "不参与任何分析与打分。",
        icon="ℹ️",
    )

    sources = sorted(comex_df["source"].unique().tolist())

    default_source = pick_primary_source(comex_df)

    selected = st.selectbox(
        "选择分析来源（图表与联动分析只使用单一来源，避免跨频率混算）",
        sources,
        index=sources.index(default_source) if default_source in sources else 0,
        format_func=_source_label,
    )

    single_source_df = comex_df[comex_df["source"] == selected].copy()

    st.divider()

    render_source_overview(comex_df)

    st.divider()

    render_trend_chart(single_source_df, selected)

    st.divider()

    render_cross_source_check(comex_df)

    st.divider()

    render_daily_changes(single_source_df, selected)

    st.divider()

    render_gold_linkage(single_source_df, gold_df, selected)

    st.divider()

    render_review_workbench(single_source_df, selected)

    st.divider()

    render_raw_table(comex_df)

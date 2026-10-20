import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


from pathlib import Path
from plotly.subplots import make_subplots
from database.db import get_connection
from src.quant.dashboard import render_ai_quant_page
from src.ui.comex_page import render_comex_page


# ============================================================
# 1. 页面配置
# ============================================================

st.set_page_config(
    page_title="长征量化系统",
    page_icon=Path(__file__).parent / "long_march" / "img.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. 页面样式
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1120;
        color: #E5E7EB;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    div[data-testid="stMetric"] {
        background-color: #151F32;
        border: 1px solid #263449;
        padding: 18px;
        border-radius: 10px;
    }

    h1, h2, h3 {
        color: #F5C451;
    }

    div[data-testid="stMetricLabel"] {
        color: #9CA3AF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. 数据读取
# ============================================================

@st.cache_data(ttl=300)
def load_gold_data():
    """读取黄金历史行情。"""

    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                source
            FROM gold_prices
            WHERE symbol = 'XAU/USD'
              AND source = 'XAUS'
            ORDER BY timestamp
            """,
            conn,
        )

    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(
            subset=["timestamp", "close"]
        )

    return df


@st.cache_data(ttl=300)
def load_macro_data():
    """读取已保存的宏观经济数据。"""

    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                indicator,
                observation_date,
                value,
                release_date,
                source
            FROM macro_data
            WHERE source = 'FRED'
            ORDER BY observation_date
            """,
            conn,
        )

    if not df.empty:
        df["observation_date"] = pd.to_datetime(
            df["observation_date"],
            errors="coerce",
        )

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["observation_date", "value"]
        )

    return df


@st.cache_data(ttl=300)
def load_comex_data():
    """
    读取 COMEX 黄金库存历史数据；表不存在时返回空表。

    注意两个多来源场景下的陷阱：
    1. 日频来源（东方财富）只提供 total，registered/eligible 为空，
       因此不能要求三者同时非空，否则整批日频数据会被丢弃。
    2. 变化量必须按 source 分组计算，跨来源做 diff 会得到错误结果
       （不同来源的日期节奏不同，相邻行未必是相邻业务日）。
    """
    columns = [
        "report_date", "registered", "eligible", "total", "source",
        "source_url", "status", "reviewed_at",
    ]

    try:
        with get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    i.report_date,
                    i.registered,
                    i.eligible,
                    i.total,
                    i.source,
                    q.source_url,
                    q.status,
                    q.reviewed_at
                FROM comex_inventory i
                LEFT JOIN comex_inventory_quality q
                    ON q.report_date = i.report_date
                   AND q.source = i.source
                ORDER BY i.report_date
                """,
                conn,
            )
    except Exception:
        # 质量表缺失时退回只读库存表，保证页面仍可用
        try:
            with get_connection() as conn:
                df = pd.read_sql_query(
                    """
                    SELECT report_date, registered, eligible, total, source
                    FROM comex_inventory
                    ORDER BY report_date
                    """,
                    conn,
                )

            for column in ("source_url", "status", "reviewed_at"):
                df[column] = None
        except Exception:
            return pd.DataFrame(columns=columns)

    if df.empty:
        return pd.DataFrame(columns=columns)

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    for column in ["registered", "eligible", "total"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 至少要有 total 才算有效记录；拆分字段允许缺失
    df = df.dropna(subset=["report_date", "total"])

    df = df.sort_values("report_date").drop_duplicates(
        subset=["report_date", "source"], keep="last"
    )

    # 变化量按来源分别计算，避免跨来源的错误 diff
    for column in ["registered", "eligible", "total"]:
        df[f"{column}_change"] = (
            df.groupby("source")[column].diff()
        )

    return df


@st.cache_data(ttl=300)
def load_inventory_quality_status():
    """读取库存核验状态的分布，用于页面展示数据可信度概况。"""

    try:
        with get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT status, source, COUNT(*) AS records
                FROM comex_inventory_quality
                GROUP BY status, source
                ORDER BY status, source
                """,
                conn,
            )
    except Exception:
        return pd.DataFrame(columns=["status", "source", "records"])

    return df


# 参与量化分析的核验状态：跨源一致(VERIFIED) 或 单源通过全部自动校验(ACCEPTED)
TRUSTED_INVENTORY_STATUSES = ("VERIFIED", "ACCEPTED")


@st.cache_data(ttl=300)
def load_trusted_inventory_keys():
    """
    读取可信的 COMEX 库存记录键 (report_date, source)。

    量化打分只使用 VERIFIED / ACCEPTED 记录；
    REVIEW（待复核）与 REJECTED（已证伪）一律不参与因子计算。
    质量表不存在时返回 None，表示不做可信度过滤。
    """

    placeholders = ",".join("?" for _ in TRUSTED_INVENTORY_STATUSES)

    try:
        with get_connection() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT report_date, source
                FROM comex_inventory_quality
                WHERE status IN ({placeholders})
                """,
                conn,
                params=TRUSTED_INVENTORY_STATUSES,
            )
    except Exception:
        # 质量表尚未创建，交由库存因子内部按数据量降级处理
        return None

    if df.empty:
        return []

    keys = []

    for _, row in df.iterrows():
        parsed = pd.to_datetime(row["report_date"], errors="coerce")

        if pd.isna(parsed):
            continue

        keys.append((parsed.normalize(), str(row["source"])))

    return keys


# ============================================================
# 4. 指标名称
# ============================================================
MACRO_NAMES = {
    "DGS10": "美国10年期国债收益率",
    "DFII10": "美国10年期实际利率",
    "CPIAUCSL": "美国消费者价格指数",
    "FEDFUNDS": "美国联邦基金有效利率",
}

MACRO_UNITS = {
    "DGS10": "%",
    "DFII10": "%",
    "CPIAUCSL": "指数",
    "FEDFUNDS": "%",
}


# ============================================================
# 5. 技术指标
# ============================================================

def calculate_indicators(df):
    """计算黄金价格的技术指标。"""

    df = df.copy()

    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - 100 / (1 + rs)

    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI14",
    ] = 100

    ema12 = df["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["MACD_hist"] = (
        df["MACD"] - df["MACD_signal"]
    )

    return df


# ============================================================
# 6. 基本面分析辅助函数
# ============================================================

def latest_macro_value(macro_df, indicator):
    """获取指定指标的最新观测值及日期。"""

    subset = macro_df[
        macro_df["indicator"] == indicator
    ].sort_values("observation_date")

    if subset.empty:
        return None, None

    row = subset.iloc[-1]

    return row["value"], row["observation_date"]


def make_macro_chart(macro_df, indicator):
    """生成单项宏观指标历史走势图。"""

    subset = macro_df[
        macro_df["indicator"] == indicator
    ].sort_values("observation_date")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=subset["observation_date"],
            y=subset["value"],
            mode="lines",
            name=MACRO_NAMES[indicator],
            line=dict(width=2),
            hovertemplate=(
                "日期：%{x|%Y-%m-%d}<br>"
                "数值：%{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=MACRO_NAMES[indicator],
        height=350,
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        xaxis_title="观测日期",
        yaxis_title=MACRO_UNITS[indicator],
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


# ============================================================
# 7. 侧边栏
# ============================================================

with st.sidebar:
    st.image(Path(__file__).parent / "long_march" / "img_1.png", width=120)

    st.markdown("长征量化分析")
    st.caption("GOLD INTELLIGENCE SYSTEM")

    st.divider()

    page = st.radio(
        "导航菜单",
        [
            "市场总览",
            "行情分析",
            "AI量化分析",
            "基本面分析",
            "COMEX库存分析",
            "数据中心",
        ],
    )

    st.divider()

    st.markdown("### 数据配置")
    st.caption("交易品种：XAU/USD")
    st.caption("行情来源：XAUS")
    st.caption("宏观数据来源：FRED")

    if st.button("刷新页面数据"):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# 8. 读取数据
# ============================================================

gold_df = load_gold_data()
macro_df = load_macro_data()
comex_df = load_comex_data()
trusted_inventory = load_trusted_inventory_keys()


# ============================================================
# 9. 市场总览
# ============================================================

if page == "市场总览":

    st.title("长征|黄金市场总览")

    st.caption(
        "Aurum Intelligence | GOLD MARKET INTELLIGENCE"
    )

    if gold_df.empty:
        st.warning("数据库中暂无黄金行情，请先运行黄金行情入库程序。")
        st.stop()

    gold_df = calculate_indicators(gold_df)

    latest = gold_df.iloc[-1]

    if len(gold_df) >= 2:
        previous_close = gold_df.iloc[-2]["close"]

        change_pct = (
            latest["close"] / previous_close - 1
        ) * 100
    else:
        change_pct = 0.0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "黄金最新收盘价",
        f"${latest['close']:,.2f}",
        f"{change_pct:+.2f}%",
    )

    rsi_value = latest["RSI14"]

    col2.metric(
        "RSI14",
        f"{rsi_value:.2f}" if pd.notna(rsi_value) else "暂无",
    )

    col3.metric(
        "MA20",
        f"${latest['MA20']:,.2f}"
        if pd.notna(latest["MA20"])
        else "暂无",
    )

    col4.metric(
        "历史记录数量",
        f"{len(gold_df):,}",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=gold_df["timestamp"],
            y=gold_df["close"],
            name="黄金收盘价",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=gold_df["timestamp"],
            y=gold_df["MA20"],
            name="MA20",
            line=dict(width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=gold_df["timestamp"],
            y=gold_df["MA50"],
            name="MA50",
            line=dict(width=1.5),
        )
    )

    fig.update_layout(
        title="黄金价格与移动平均线",
        height=500,
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        f"行情观测时间：{latest['timestamp']} | 数据来源：XAUS"
    )


# ============================================================
# 10. 行情分析
# ============================================================

elif page == "行情分析":

    st.title("技术分析")

    if gold_df.empty:
        st.warning("数据库中暂无黄金行情。")
        st.stop()

    df = calculate_indicators(gold_df)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(
            "黄金K线与均线",
            "RSI14",
            "MACD",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="黄金K线",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["MA20"],
            name="MA20",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["MA50"],
            name="MA50",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["RSI14"],
            name="RSI14",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        row=2,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["MACD_hist"],
            name="MACD柱",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["MACD"],
            name="MACD",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["MACD_signal"],
            name="Signal",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=850,
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# 11. AI 量化分析
# ============================================================

elif page == "AI量化分析":

    render_ai_quant_page(
        gold_df=gold_df,
        macro_df=macro_df,
        comex_df=comex_df,
        trusted_inventory=trusted_inventory,
    )


# ============================================================
# 12. 基本面分析
# ============================================================

elif page == "基本面分析":

    st.title("基本面分析")

    st.caption(
        "MACROECONOMIC FACTORS | FUNDAMENTAL RESEARCH"
    )

    if macro_df.empty:
        st.warning(
            "数据库中暂无宏观数据，请先运行宏观数据入库程序。"
        )
        st.stop()

    st.subheader("一、宏观指标总览")

    indicators = [
        "DGS10",
        "DFII10",
        "CPIAUCSL",
        "FEDFUNDS",
    ]

    cols = st.columns(4)

    for col, indicator in zip(cols, indicators):

        value, date = latest_macro_value(
            macro_df,
            indicator,
        )

        with col:
            st.markdown(
                f"**{MACRO_NAMES[indicator]}**"
            )

            if value is None:
                st.metric("最新数值", "暂无")
            else:
                unit = MACRO_UNITS[indicator]

                st.metric(
                    "最新数值",
                    f"{value:.3f} {unit}",
                )

                st.caption(
                    f"观测日期：{date:%Y-%m-%d}"
                )

    st.divider()

    st.subheader("二、宏观经济指标走势")

    selected_indicator = st.selectbox(
        "选择需要查看的指标",
        indicators,
        format_func=lambda x: MACRO_NAMES[x],
    )

    st.plotly_chart(
        make_macro_chart(
            macro_df,
            selected_indicator,
        ),
        use_container_width=True,
    )

    st.divider()

    st.subheader("三、黄金价格与实际利率")

    gold_returns = pd.DataFrame()
    real_rate = macro_df[
        macro_df["indicator"] == "DFII10"
    ].copy()

    if gold_df.empty:
        st.warning(
            "没有黄金行情数据，暂时无法进行对比。"
        )

    elif real_rate.empty:
        st.warning(
            "没有实际利率数据，暂时无法进行对比。"
        )

    else:
        gold_returns = gold_df[
            ["timestamp", "close"]
        ].copy()

        gold_returns["date"] = (
            gold_returns["timestamp"]
            .dt.tz_convert(None)
            .dt.normalize()
        )

        # 使用黄金日收益率与实际利率变化进行对比
        gold_returns["gold_return_pct"] = (
            gold_returns["close"].pct_change() * 100
        )

        real_rate["date"] = (
            real_rate["observation_date"]
            .dt.normalize()
        )

        real_rate = real_rate.sort_values("date")

        real_rate["rate_change"] = (
            real_rate["value"].diff()
        )

        # 只对齐双方都实际存在观测值的日期
        comparison = pd.merge(
            gold_returns[
                ["date", "gold_return_pct"]
            ],
            real_rate[
                ["date", "value", "rate_change"]
            ],
            on="date",
            how="inner",
        ).dropna()

        if len(comparison) >= 3:

            correlation = comparison[
                ["gold_return_pct", "rate_change"]
            ].corr().iloc[0, 1]

            col1, col2 = st.columns(2)

            col1.metric(
                "同期日变化相关系数",
                f"{correlation:.3f}",
            )

            col2.metric(
                "有效配对数量",
                f"{len(comparison):,}",
            )

            st.caption(
                "相关系数用于描述同期变化之间的线性关系，"
                "不代表因果关系，也不能直接作为买卖信号。"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=comparison["rate_change"],
                    y=comparison["gold_return_pct"],
                    mode="markers",
                    name="同期变化",
                    hovertemplate=(
                        "实际利率变化：%{x:.3f} 个百分点<br>"
                        "黄金日收益率：%{y:.3f}%"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title="实际利率日变化与黄金日收益率",
                xaxis_title="实际利率日变化（百分点）",
                yaxis_title="黄金日收益率（%）",
                height=450,
                template="plotly_dark",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#111827",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "当前数据的共同观测日期不足，"
                "暂时无法可靠计算相关系数。"
            )

    st.divider()

    st.subheader("四、宏观数据明细")

    st.dataframe(
        macro_df.sort_values(
            "observation_date",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = macro_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="下载宏观数据 CSV",
        data=csv_data,
        file_name="aurum_macro_data.csv",
        mime="text/csv",
    )


# ============================================================
# 13. COMEX 库存分析
# ============================================================

elif page == "COMEX库存分析":

    # 页面逻辑（含人工核验工作台）已抽到 src/ui/comex_page.py
    render_comex_page(comex_df, gold_df)


# ============================================================
# 14. 数据中心
# ============================================================

elif page == "数据中心":

    st.title("数据中心")

    tab1, tab2, tab3 = st.tabs(
        ["黄金行情", "宏观经济数据", "COMEX库存"]
    )

    with tab1:

        if gold_df.empty:
            st.warning("暂无黄金行情数据。")

        else:
            st.metric(
                "黄金行情记录数",
                f"{len(gold_df):,}",
            )

            st.dataframe(
                gold_df.sort_values(
                    "timestamp",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_data = gold_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "下载黄金行情 CSV",
                data=csv_data,
                file_name="aurum_gold_prices.csv",
                mime="text/csv",
            )

    with tab2:

        if macro_df.empty:
            st.warning("暂无宏观经济数据。")

        else:
            st.metric(
                "宏观数据记录数",
                f"{len(macro_df):,}",
            )

            st.dataframe(
                macro_df.sort_values(
                    "observation_date",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_data = macro_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "下载宏观数据 CSV",
                data=csv_data,
                file_name="aurum_macro_data.csv",
                mime="text/csv",
            )

    with tab3:
        if comex_df.empty:
            st.warning("暂无 COMEX 库存数据。")
        else:
            st.metric("COMEX 库存记录数", f"{len(comex_df):,}")
            st.dataframe(
                comex_df.sort_values("report_date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            csv_data = comex_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载 COMEX 库存 CSV",
                data=csv_data,
                file_name="aurum_comex_inventory.csv",
                mime="text/csv",
            )


# ============================================================
# 15. 免责声明
# ============================================================

st.divider()

st.caption(
    "Aurum Intelligence | 黄金量化研究平台"
)

st.caption(
    "本平台用于金融数据研究和历史分析。"
    "相关性不代表因果关系，历史表现不代表未来结果。"
    "数据可能存在发布时间差异、延迟或修订。"
)

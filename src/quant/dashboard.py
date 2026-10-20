# 本文件负责「AI量化分析」页面的渲染，数值全部来自本地量化引擎
"""
AI 量化分析页面。

页面结构：
1. 信号总览（综合得分、方向、仓位）
2. 因子归因（最新一日的因子得分与加权贡献）
3. 价格 / 得分 / 持仓三联图
4. 历史回测绩效与净值回撤
5. AI 解读（调用大模型，失败时降级为本地规则解读）
6. 参数配置与数据质量说明

设计约束：
- AI 只做文字解读，页面上所有数字均由 factors.py / backtest.py 本地算出。
- 仓位使用 shift(1)，杜绝未来函数。
- COMEX 库存只采用核验状态为 VERIFIED（跨源一致）或 ACCEPTED
  （单源通过全部自动校验）的记录；REVIEW 与 REJECTED 不参与打分。
"""

import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from src.quant.ai_analyst import (
    DISCLAIMER,
    analyze,
    get_api_key,
)
from src.quant.backtest import run_backtest
from src.quant.factors import (
    DEFAULT_WEIGHTS,
    FACTOR_GROUPS,
    FACTOR_LABELS,
    build_factors,
)


# =========================================================
# 1. 样式常量
# =========================================================

PAPER_BG = "#0B1120"
PLOT_BG = "#111827"
GOLD = "#F5C451"
BULL_COLOR = "#22C55E"
BEAR_COLOR = "#EF4444"
MUTED = "#9CA3AF"


# =========================================================
# 2. 通用辅助
# =========================================================

def _fmt(value, digits=2, suffix="", dash="—"):
    """格式化数值，缺失时统一显示破折号。"""

    if value is None:
        return dash

    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash

    if math.isnan(number) or math.isinf(number):
        return dash

    return f"{number:,.{digits}f}{suffix}"


def _direction_label(score) -> str:
    """把综合得分翻译成中文方向。"""

    if score is None:
        return "数据缺失"

    try:
        number = float(score)
    except (TypeError, ValueError):
        return "数据缺失"

    if math.isnan(number):
        return "数据缺失"

    if number >= 0.15:
        return "偏多"
    if number <= -0.15:
        return "偏空"
    if number > 0.05:
        return "略偏多"
    if number < -0.05:
        return "略偏空"
    return "中性"


def _direction_color(score) -> str:
    """根据方向返回配色。"""

    label = _direction_label(score)

    if "多" in label:
        return BULL_COLOR
    if "空" in label:
        return BEAR_COLOR
    return MUTED


def _base_layout(fig, height=420):
    """统一深色主题布局。"""

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


# =========================================================
# 3. 信号总览
# =========================================================

def render_overview(factor_df: pd.DataFrame, backtest: dict):
    """渲染顶部信号指标卡。"""

    latest = factor_df.iloc[-1]

    score = latest.get("score")
    score_value = None if pd.isna(score) else float(score)

    color = _direction_color(score_value)
    label = _direction_label(score_value)

    st.markdown(
        f"""
        <div style="background:#151F32;border:1px solid #263449;
                    border-left:5px solid {color};border-radius:10px;
                    padding:16px 20px;margin-bottom:18px;">
            <div style="font-size:13px;color:{MUTED};letter-spacing:1px;">
                当前多因子综合信号
            </div>
            <div style="font-size:30px;font-weight:700;color:{color};">
                {label}
                <span style="font-size:18px;color:#E5E7EB;margin-left:10px;">
                    {_fmt(score_value, 3)}
                </span>
            </div>
            <div style="font-size:12px;color:{MUTED};margin-top:4px;">
                得分区间 -1（极度偏空）~ +1（极度偏多），
                由 {int(latest.get('active_factor_count', 0))} 个可用因子加权得出
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    strategy = backtest.get("strategy_metrics", {})
    trades = backtest.get("trade_stats", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("最新收盘价", f"${_fmt(latest.get('close'), 2)}")

    col2.metric(
        "模型仓位",
        _fmt(latest.get("position"), 2),
        "满仓为 1.00",
        delta_color="off",
    )

    col3.metric(
        "波动率缩放",
        _fmt(latest.get("vol_scale"), 2),
        f"年化波动 {_fmt(latest.get('vol_20d_annual'), 1, '%')}",
        delta_color="off",
    )

    col4.metric(
        "回测累计收益",
        _fmt(strategy.get("total_return_pct"), 2, "%"),
        f"基准 {_fmt(backtest.get('benchmark_metrics', {}).get('total_return_pct'), 2, '%')}",
    )

    col5.metric(
        "区间换手次数",
        f"{_fmt(trades.get('trade_count'), 0)}",
        f"持仓占比 {_fmt(trades.get('exposure_pct'), 1, '%')}",
        delta_color="off",
    )


# =========================================================
# 4. 因子归因
# =========================================================

def render_factor_attribution(factor_df: pd.DataFrame, weights: dict):
    """渲染最新一日的因子得分与加权贡献。"""

    latest = factor_df.iloc[-1]

    rows = []

    for group_name, keys in FACTOR_GROUPS.items():
        for key in keys:
            if key not in factor_df.columns:
                continue

            raw = pd.to_numeric(latest.get(key), errors="coerce")
            weight = float(weights.get(key, DEFAULT_WEIGHTS[key]))

            available = bool(pd.notna(raw))

            score_value = float(raw) if available else None

            contribution_column = f"w_{key}"

            contribution = (
                pd.to_numeric(
                    latest.get(contribution_column), errors="coerce"
                )
                if contribution_column in factor_df.columns
                else np.nan
            )

            rows.append(
                {
                    "因子分组": group_name,
                    "因子": FACTOR_LABELS[key],
                    "得分": score_value,
                    "方向": _direction_label(score_value)
                    if available
                    else "数据缺失",
                    "权重": weight,
                    "加权贡献": (
                        float(contribution) if pd.notna(contribution) else None
                    ),
                    "_key": key,
                }
            )

    table = pd.DataFrame(rows)

    display = table.drop(columns=["_key"]).copy()

    st.dataframe(
        display.style.format(
            {
                "得分": lambda v: "—" if v is None else f"{v:+.3f}",
                "权重": "{:.2f}",
                "加权贡献": lambda v: (
                    "—" if v is None else f"{v:+.3f}"
                ),
            }
        ).map(
            lambda v: (
                f"color: {BULL_COLOR}"
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and v > 0.05
                else (
                    f"color: {BEAR_COLOR}"
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and v < -0.05
                    else ""
                )
            ),
            subset=["得分", "加权贡献"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 加权贡献条形图
    plot_table = table.copy()

    plot_table["加权贡献"] = plot_table["加权贡献"].fillna(0.0)
    plot_table = plot_table.sort_values("加权贡献")

    colors = [
        BULL_COLOR if value > 0 else BEAR_COLOR if value < 0 else MUTED
        for value in plot_table["加权贡献"]
    ]

    fig = go.Figure(
        go.Bar(
            x=plot_table["加权贡献"],
            y=plot_table["因子"],
            orientation="h",
            marker_color=colors,
            text=[
                _fmt(value, 3) for value in plot_table["加权贡献"]
            ],
            textposition="outside",
            hovertemplate=(
                "%{y}<br>加权贡献：%{x:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_color=MUTED, line_width=1)

    _base_layout(fig, height=430)

    fig.update_layout(
        title="最新一日各因子对综合得分的加权贡献",
        xaxis_title="加权贡献（正值利多，负值利空）",
        margin=dict(l=160, r=40, t=60, b=40),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 5. 价格 / 得分 / 持仓
# =========================================================

def render_signal_chart(
    factor_df: pd.DataFrame,
    long_threshold: float,
    short_threshold: float,
):
    """渲染价格、综合得分与持仓的三联图，并标出信号切换点。"""

    df = factor_df.copy()

    index = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.46, 0.28, 0.26],
        subplot_titles=(
            "黄金收盘价与信号切换点",
            "多因子综合得分",
            "模型持仓（已按波动率缩放）",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=index,
            y=df["close"],
            mode="lines",
            name="收盘价",
            line=dict(color=GOLD, width=2),
        ),
        row=1,
        col=1,
    )

    # 信号切换点：仓位方向发生变化时标记
    signal_change = df["signal"].diff()

    entry = df[(signal_change > 0) & df["signal"].notna()]
    exit_point = df[(signal_change < 0) & df["signal"].notna()]

    if not entry.empty:
        fig.add_trace(
            go.Scatter(
                x=entry.index,
                y=entry["close"],
                mode="markers",
                name="开仓 / 转多",
                marker=dict(
                    symbol="triangle-up",
                    size=11,
                    color=BULL_COLOR,
                    line=dict(width=1, color="#0B1120"),
                ),
                hovertemplate="日期：%{x|%Y-%m-%d}<br>价格：%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if not exit_point.empty:
        fig.add_trace(
            go.Scatter(
                x=exit_point.index,
                y=exit_point["close"],
                mode="markers",
                name="平仓 / 转空",
                marker=dict(
                    symbol="triangle-down",
                    size=11,
                    color=BEAR_COLOR,
                    line=dict(width=1, color="#0B1120"),
                ),
                hovertemplate="日期：%{x|%Y-%m-%d}<br>价格：%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=index,
            y=df["score"],
            mode="lines",
            name="综合得分",
            line=dict(color="#60A5FA", width=1.8),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=long_threshold,
        line_dash="dash",
        line_color=BULL_COLOR,
        row=2,
        col=1,
    )

    fig.add_hline(
        y=short_threshold,
        line_dash="dash",
        line_color=BEAR_COLOR,
        row=2,
        col=1,
    )

    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=index,
            y=df["position"],
            mode="lines",
            name="持仓",
            fill="tozeroy",
            # 阶梯线用 line.shape="hv"，Scatter 没有 step 参数
            line=dict(color="#A78BFA", width=1.5, shape="hv"),
            fillcolor="rgba(167,139,250,0.22)",
        ),
        row=3,
        col=1,
    )

    _base_layout(fig, height=820)

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 6. 回测绩效
# =========================================================

def render_backtest(backtest: dict, cost_bps: float):
    """渲染净值曲线、回撤与绩效对照表。"""

    returns_frame = backtest.get("returns")

    if returns_frame is None or returns_frame.empty:
        st.info("回测样本不足，暂无绩效数据。")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.62, 0.38],
        subplot_titles=("策略净值 vs 买入持有", "回撤（%）"),
    )

    fig.add_trace(
        go.Scatter(
            x=returns_frame.index,
            y=returns_frame["strategy_equity"],
            mode="lines",
            name="多因子策略",
            line=dict(color=GOLD, width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=returns_frame.index,
            y=returns_frame["benchmark_equity"],
            mode="lines",
            name="买入持有",
            line=dict(color="#60A5FA", width=1.6, dash="dot"),
        ),
        row=1,
        col=1,
    )

    fig.add_hline(y=1, line_color=MUTED, line_width=1, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=backtest["strategy_drawdown"].index,
            y=backtest["strategy_drawdown"],
            mode="lines",
            name="策略回撤",
            fill="tozeroy",
            line=dict(color=BEAR_COLOR, width=1.2),
            fillcolor="rgba(239,68,68,0.18)",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=backtest["benchmark_drawdown"].index,
            y=backtest["benchmark_drawdown"],
            mode="lines",
            name="基准回撤",
            line=dict(color="#60A5FA", width=1, dash="dot"),
        ),
        row=2,
        col=1,
    )

    _base_layout(fig, height=640)

    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    strategy = backtest["strategy_metrics"]
    benchmark = backtest["benchmark_metrics"]
    trades = backtest["trade_stats"]

    metric_rows = [
        ("累计收益率", "total_return_pct", "%"),
        ("年化收益率", "annual_return_pct", "%"),
        ("年化波动率", "annual_vol_pct", "%"),
        ("夏普比率", "sharpe", ""),
        ("索提诺比率", "sortino", ""),
        ("最大回撤", "max_drawdown_pct", "%"),
        ("卡玛比率", "calmar", ""),
        ("持仓日胜率", "win_rate_pct", "%"),
    ]

    table_rows = []

    for label, key, suffix in metric_rows:
        table_rows.append(
            {
                "指标": label,
                "多因子策略": _fmt(strategy.get(key), 2, suffix),
                "买入持有": _fmt(benchmark.get(key), 2, suffix),
            }
        )

    table_rows.extend(
        [
            {
                "指标": "样本交易日",
                "多因子策略": _fmt(strategy.get("sample_days"), 0),
                "买入持有": _fmt(benchmark.get("sample_days"), 0),
            },
            {
                "指标": "信号切换次数",
                "多因子策略": _fmt(trades.get("trade_count"), 0),
                "买入持有": "—",
            },
            {
                "指标": "平均持仓占比",
                "多因子策略": _fmt(trades.get("exposure_pct"), 1, "%"),
                "买入持有": "100.0%",
            },
            {
                "指标": "累计换手（仓位变动）",
                "多因子策略": _fmt(trades.get("total_turnover"), 2),
                "买入持有": "—",
            },
            {
                "指标": f"累计交易成本（{cost_bps:.0f} bps/边）",
                "多因子策略": _fmt(trades.get("total_cost_pct"), 3, "%"),
                "买入持有": "—",
            },
        ]
    )

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "回测口径：信号在收盘后生成，仓位次日生效（shift 1），"
        "已按换手计提双边交易成本；未包含滑点、资金费率与流动性冲击。"
    )


# =========================================================
# 7. AI 解读
# =========================================================

def render_ai_commentary(factor_df: pd.DataFrame, backtest: dict, params: dict):
    """渲染 AI 解读区块。"""

    st.subheader("五、AI 智能解读")

    api_key = get_api_key()

    if not api_key:
        st.warning(
            "未检测到 DEEPSEEK_API_KEY，当前仅能输出本地规则化解读。"
            "请在项目根目录 .env 文件中配置该密钥后刷新页面。"
        )

    with st.expander("解读设置", expanded=False):
        setting_col1, setting_col2 = st.columns([1, 1])

        with setting_col1:
            use_model = st.toggle(
                "调用大模型生成解读",
                value=bool(api_key),
                disabled=not api_key,
                help="关闭后使用本地规则模板生成解读，不产生任何 API 调用。",
            )

            model_name = st.selectbox(
                "模型",
                ["deepseek-chat", "deepseek-reasoner"],
                index=0,
                disabled=not (use_model and bool(api_key)),
                help="deepseek-reasoner 推理更强但更慢、消耗更高。",
            )

        with setting_col2:
            temperature = st.slider(
                "解读发散度",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                disabled=not (use_model and bool(api_key)),
                help="数值越低，解读越贴近给定数据、越保守。",
            )

        st.caption(
            "AI 只会收到本地量化引擎算出的数字简报，并被明确要求"
            "不得编造数值、不得给出买卖指令。"
        )

    generate = st.button(
        "🤖 生成 AI 解读",
        type="primary",
        use_container_width=True,
    )

    if generate:
        with st.spinner("正在分析多因子信号…"):
            result = analyze(
                factor_df,
                backtest,
                weights=params["weights"],
                params={
                    "多头阈值": params["long_threshold"],
                    "空头阈值": params["short_threshold"],
                    "允许做空": params["allow_short"],
                    "波动率目标年化%": params["target_vol"],
                    "交易成本bps": params["cost_bps"],
                },
                use_model=use_model,
                model=model_name,
                temperature=temperature,
                diagnosis=params["diagnosis"],
            )

        st.session_state["ai_result"] = result

    result = st.session_state.get("ai_result")

    if not result:
        st.info("点击上方按钮，基于当前因子与参数生成本轮解读。")
        return

    mode = result.get("mode")

    if mode == "model":
        st.success(
            f"AI 解读已生成 · 模型：{result.get('model') or model_name}"
        )
    elif mode == "rule":
        st.info("当前为本地规则化解读（未调用大模型）。")
    else:
        st.warning(
            f"大模型调用未成功，已降级为本地规则化解读。原因：{result.get('error')}"
        )

    st.markdown(result.get("text", ""))

    st.caption(DISCLAIMER)

    with st.expander("查看发送给模型的事实简报（JSON）"):
        st.json(result.get("brief", {}))

        st.download_button(
            "下载事实简报 JSON",
            data=json.dumps(
                result.get("brief", {}), ensure_ascii=False, indent=2
            ).encode("utf-8"),
            file_name="aurum_quant_fact_brief.json",
            mime="application/json",
        )


# =========================================================
# 8. 参数配置
# =========================================================

def render_parameter_panel() -> dict:
    """渲染参数配置面板，返回当前参数。"""

    st.subheader("六、模型参数配置")

    params = {}

    with st.expander("因子权重与信号参数", expanded=False):
        st.markdown("**因子权重**")
        st.caption("权重为 0 表示该因子不参与打分；缺失因子会自动按可用权重重新归一化。")

        keys = list(DEFAULT_WEIGHTS.keys())

        weights = {}

        for start in range(0, len(keys), 2):
            pair = keys[start:start + 2]
            columns = st.columns(2)

            for column, key in zip(columns, pair):
                with column:
                    weights[key] = st.slider(
                        FACTOR_LABELS[key],
                        min_value=0.0,
                        max_value=3.0,
                        value=float(DEFAULT_WEIGHTS[key]),
                        step=0.1,
                        key=f"weight_{key}",
                    )

        st.divider()
        st.markdown("**信号与仓位**")

        signal_col1, signal_col2, signal_col3 = st.columns(3)

        with signal_col1:
            params["long_threshold"] = st.slider(
                "做多阈值",
                min_value=-1.0,
                max_value=1.0,
                value=0.15,
                step=0.05,
                help="综合得分高于该值时开多仓。",
            )

            params["allow_short"] = st.checkbox(
                "允许做空",
                value=False,
                help="开启后，得分低于空头阈值时建立负仓位。",
            )

        with signal_col2:
            params["short_threshold"] = st.slider(
                "做空阈值",
                min_value=-1.0,
                max_value=0.0,
                value=-0.15,
                step=0.05,
                help="仅在「允许做空」开启时生效。",
            )

            params["use_vol_target"] = st.checkbox(
                "波动率目标缩放",
                value=True,
                help="按 20 日已实现波动率把仓位缩放到目标水平，波动越大仓位越低。",
            )

        with signal_col3:
            params["target_vol"] = st.slider(
                "目标年化波动率(%)",
                min_value=5.0,
                max_value=30.0,
                value=12.0,
                step=1.0,
                disabled=not params["use_vol_target"],
            )

            params["cost_bps"] = st.slider(
                "单边交易成本(bps)",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=1.0,
                help="按仓位变动计提，双边均计。",
            )

        if params["long_threshold"] <= params["short_threshold"]:
            st.error("做多阈值必须大于做空阈值，请重新调整。")
            st.stop()

    params["weights"] = weights

    return params


# =========================================================
# 9. 数据质量
# =========================================================

def render_data_quality(factor_df: pd.DataFrame, diagnosis: dict):
    """渲染数据质量与因子覆盖情况。"""

    st.subheader("七、数据质量与口径说明")

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:
        st.markdown("**样本区间**")

        start = factor_df.index.min()
        end = factor_df.index.max()

        st.write(
            f"- 交易日：{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
        )
        st.write(f"- 样本天数：{len(factor_df):,} 个交易日")
        st.write(
            "- 宏观对齐：按公布滞后天数（利率 1 天、CPI 45 天、"
            "政策利率 35 天）取「当时已公布」的观测值，避免未来函数"
        )

    with quality_col2:
        st.markdown("**COMEX 库存核验状态**")

        total = diagnosis.get("total_reports", 0)
        trusted = diagnosis.get("trusted_reports", 0)
        required = diagnosis.get("required_reports", 0)
        usable = diagnosis.get("usable", False)

        sources = diagnosis.get("sources") or []
        metric_used = diagnosis.get("metric_used")
        discrepancies = diagnosis.get("source_discrepancies") or []

        if usable:
            st.success(
                f"可信记录 {trusted} / {total} 条，"
                f"库存因子已启用（{metric_used} 口径）。"
            )
        else:
            st.warning(
                f"可信记录 {trusted} / {total} 条，"
                f"计算库存变化因子至少需要 {required} 条。"
            )

        if sources:
            st.caption(f"参与来源：{', '.join(sources)}")

        st.caption(diagnosis.get("reason") or "")

        if discrepancies:
            st.caption(
                f"另有 {len(discrepancies)} 个业务日存在跨源数值分歧，"
                "已按来源优先级取主源的值。"
            )

        if not usable:
            st.info(
                "补全方式：运行 "
                "python -m src.data_sources.comex_inventory_api --commit "
                "抓取结构化库存数据；采集时会自动做日期偏移自检与跨源交叉验证。"
            )

    st.markdown("**因子数据覆盖**")

    coverage_rows = []

    for group_name, keys in FACTOR_GROUPS.items():
        for key in keys:
            if key not in factor_df.columns:
                continue

            series = pd.to_numeric(factor_df[key], errors="coerce")

            available = int(series.notna().sum())

            coverage_rows.append(
                {
                    "因子分组": group_name,
                    "因子": FACTOR_LABELS[key],
                    "可用天数": available,
                    "总天数": int(len(factor_df)),
                    "覆盖率": f"{available / max(len(factor_df), 1) * 100:.1f}%",
                }
            )

    st.dataframe(
        pd.DataFrame(coverage_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "覆盖率不足 100% 属于正常情况：均线、波动率等指标需要预热期，"
        "早期样本不参与打分，综合得分会自动按可用权重归一化。"
    )


# =========================================================
# 10. 因子明细导出
# =========================================================

def render_factor_table(factor_df: pd.DataFrame):
    """渲染最近因子明细并提供下载。"""

    st.subheader("八、因子明细")

    export_columns = [
        "close",
        "MA20",
        "MA50",
        "RSI14",
        "MACD_hist",
        "ATR_pct",
        "vol_20d_annual",
        "real_rate",
        "nominal_rate",
        "cpi_yoy",
        "fedfunds_change_3m",
        "comex_registered",
        "comex_registered_change_pct",
    ] + list(DEFAULT_WEIGHTS.keys()) + [
        "score",
        "active_factor_count",
        "signal",
        "vol_scale",
        "position",
    ]

    available_columns = [
        column for column in export_columns if column in factor_df.columns
    ]

    export_df = factor_df[available_columns].copy()
    export_df.insert(0, "date", export_df.index.strftime("%Y-%m-%d"))

    row_options = [10, 20, 50, 100, len(export_df)]

    # 样本量可能与预设值重复，需去重并保持升序
    row_options = sorted({option for option in row_options if option > 0})

    display_count = st.select_slider(
        "展示最近多少条记录",
        options=row_options,
        value=min(20, row_options[-1]),
    )

    st.dataframe(
        export_df.tail(display_count).sort_index(ascending=False).round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "下载完整因子明细 CSV",
        data=export_df.round(6).to_csv(index=False).encode("utf-8-sig"),
        file_name="aurum_quant_factors.csv",
        mime="text/csv",
    )


# =========================================================
# 11. 页面主入口
# =========================================================

def render_ai_quant_page(
    gold_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    comex_df: pd.DataFrame,
    trusted_inventory=None,
):
    """渲染整个「AI量化分析」页面。"""

    st.title("AI量化分析")

    st.caption(
        "MULTI-FACTOR QUANT ENGINE | 本地计算 + AI 解读"
    )

    if gold_df is None or gold_df.empty:
        st.warning(
            "数据库中暂无黄金行情，请先运行黄金行情入库程序。"
        )
        st.stop()

    st.info(
        "本页所有数字均由本地 pandas 引擎计算：多因子打分 → 信号 → "
        "波动率缩放 → 回测绩效。AI 只负责把已经算好的结果用自然语言解释清楚，"
        "不参与任何数值生成，也不提供买卖建议。",
        icon="ℹ️",
    )

    # --- 参数面板（放在最前，用户可先调参再看结果）---
    params = render_parameter_panel()

    # --- 构建因子与回测 ---
    factor_df = build_factors(
        gold_df,
        macro_df,
        comex_df,
        weights=params["weights"],
        trusted_inventory=trusted_inventory,
        long_threshold=params["long_threshold"],
        short_threshold=params["short_threshold"],
        allow_short=params["allow_short"],
        use_vol_target=params["use_vol_target"],
        target_vol=params["target_vol"],
    )

    if factor_df.empty:
        st.error("因子表构建失败，请检查行情数据是否有效。")
        st.stop()

    # 诊断结果由 build_factors 计算并挂在 attrs 上，直接复用，
    # 不再重复调用 prepare_inventory，避免出现两份不一致的结论。
    diagnosis = factor_df.attrs.get("inventory_diagnosis", {}) or {}

    params["diagnosis"] = diagnosis

    backtest = run_backtest(factor_df, cost_bps=params["cost_bps"])

    # --- 各区块渲染 ---
    st.subheader("一、信号总览")

    render_overview(factor_df, backtest)

    st.divider()

    st.subheader("二、因子归因")

    render_factor_attribution(factor_df, params["weights"])

    st.divider()

    st.subheader("三、信号与持仓走势")

    render_signal_chart(
        factor_df,
        params["long_threshold"],
        params["short_threshold"],
    )

    st.divider()

    st.subheader("四、历史回测绩效")

    render_backtest(backtest, params["cost_bps"])

    st.divider()

    render_ai_commentary(factor_df, backtest, params)

    st.divider()

    render_data_quality(factor_df, diagnosis)

    st.divider()

    render_factor_table(factor_df)

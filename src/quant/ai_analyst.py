
# 本文件负责生成数据简报、调用大模型解读，并在不可用时降级为规则化结论
"""
AI 解读模块。

职责边界（重要）：
1. 所有数字都由 factors.py / backtest.py 在本地算好，本模块只负责
   「组织事实简报」+「请大模型用自然语言解读」。
2. 提示词明确要求模型不得编造数值、不得给出买卖指令。
3. 没有密钥、超时或接口报错时，降级为本地规则化结论，页面不会崩。
"""

import json
import os

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.quant.factors import (
    DEFAULT_WEIGHTS,
    FACTOR_GROUPS,
    FACTOR_LABELS,
    FACTOR_MEANING,
    describe_latest_factors,
)


# =========================================================
# 1. 基础配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

CHAT_URL = "https://api.deepseek.com/chat/completions"

DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

TIMEOUT = 60

MAX_HISTORY_FACTS = 12

DISCLAIMER = (
    "以上内容由 AI 基于本地量化模型的输出自动解读，"
    "仅用于研究学习，不构成任何投资建议；市场有风险，决策需独立判断。"
)

SYSTEM_PROMPT = (
    "你是黄金量化研究平台的解读助手，服务对象是有金融基础的量化研究者。\n"
    "严格遵守以下规则：\n"
    "1. 参考使用用户提供的 JSON 简报中出现的数据，禁止任何数值，可自行从网络获取官方可信数值并对你的结论进行补充\n"
    "   价格点位、时间、机构名称或新闻事件。\n"
    "2. 简报中标记为 null 或 available=false 的因子视为数据缺失，说明缺失，\n"
    "   不得推测其取值。简报 data_quality 段说明了缺失原因与数据覆盖天数，\n"
    "   如需解释缺失，请引用其中的 reason、trusted_reports 与 metric_used 等字段。\n"
    "3. 根据专业判断给出买卖指令、目标价、仓位建议或回归分析下的收益；涉及风险时必须明确指出。\n"
    "4. 综合得分取值范围 -1 到 1，正为偏多、负为偏空、接近 0 为中性；\n"
    "   因子得分同理。解读时必须说明方向含义。\n"
    "5. 用简体中文，Markdown 格式，分「信号概览」「因子解读」「风险与局限」\n"
    "   三个小标题，总长度控制在 500 字以内。\n"
    "6. 语气克制客观，不使用夸张词汇，不添加免责声明以外的套话。\n"
    "7. 涉及权重合计、覆盖天数等汇总数字时，直接引用简报中的 "
    "active_weight_sum、factor_coverage 等字段，禁止自行加减求和。"
)


# =========================================================
# 2. 数值格式化
# =========================================================

def _round(value, digits=3):
    """安全四舍五入，非数值返回 None。"""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(number) or np.isinf(number):
        return None

    return round(number, digits)


def _safe_int(value):
    """安全转整数，失败返回 None。"""

    if value is None:
        return None

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    return number


def _direction(score) -> str:
    """把因子得分翻译成中文方向词。"""

    value = _round(score, 3)

    if value is None:
        return "数据缺失"

    if value > 0.05:
        return "偏多"
    if value < -0.05:
        return "偏空"
    return "中性"


# =========================================================
# 3. 数据简报
# =========================================================

def build_fact_brief(
    factor_df: pd.DataFrame,
    backtest: dict,
    weights: dict = None,
    params: dict = None,
    diagnosis: dict = None,
) -> dict:
    """把因子表与回测结果压缩成结构化事实简报。"""

    brief = {
        "meta": {
            "asset": "XAU/USD 现货黄金",
            "generator": "Aurum Intelligence 本地量化引擎",
            "note": "所有数值均由本地 pandas 计算，AI负责专业投资分析",
        },
        "sample": {},
        "latest": {},
        "factors": [],
        "backtest": {},
        "data_quality": {},
        "params": params or {},
        "weights": {},
    }

    if factor_df is None or factor_df.empty:
        brief["sample"]["status"] = "无可用行情数据"
        return brief

    effective_weights = weights or DEFAULT_WEIGHTS

    start = factor_df.index.min()
    end = factor_df.index.max()

    brief["sample"] = {
        "start_date": start.strftime("%Y-%m-%d") if pd.notna(start) else None,
        "end_date": end.strftime("%Y-%m-%d") if pd.notna(end) else None,
        "trading_days": int(len(factor_df)),
    }

    latest = factor_df.iloc[-1]

    # daily_return 是收益率小数，转成百分比后再交给模型，
    # 否则涨跌幅会被错误解读一个数量级。
    daily_return_raw = _round(latest.get("daily_return"), 6)

    daily_return_pct = (
        round(daily_return_raw * 100, 4)
        if daily_return_raw is not None
        else None
    )

    brief["latest"] = {
        "date": end.strftime("%Y-%m-%d") if pd.notna(end) else None,
        "close": _round(latest.get("close"), 2),
        "daily_return_pct": daily_return_pct,
        "composite_score": _round(latest.get("score"), 3),
        "score_direction": _direction(latest.get("score")),
        "signal": _round(latest.get("signal"), 2),
        "position": _round(latest.get("position"), 3),
        "vol_scale": _round(latest.get("vol_scale"), 3),
        "rsi14": _round(latest.get("RSI14"), 2),
        "ma20": _round(latest.get("MA20"), 2),
        "ma50": _round(latest.get("MA50"), 2),
        "macd_hist": _round(latest.get("MACD_hist"), 3),
        "atr_pct": _round(latest.get("ATR_pct"), 3),
        "vol_20d_annual_pct": _round(
            latest.get("vol_20d_annual"), 2
        ),
        "real_rate": _round(latest.get("real_rate"), 3),
        "real_rate_change_20d": _round(
            latest.get("real_rate_change_20d"), 3
        ),
        "nominal_rate": _round(latest.get("nominal_rate"), 3),
        "cpi_yoy_pct": _round(latest.get("cpi_yoy"), 2),
        "fedfunds_change_3m": _round(
            latest.get("fedfunds_change_3m"), 3
        ),
        "comex_registered_oz": _round(
            latest.get("comex_registered"), 0
        ),
        "comex_registered_change_pct": _round(
            latest.get("comex_registered_change_pct"), 2
        ),
        "active_factor_count": _safe_int(
            latest.get("active_factor_count")
        ),
        "total_factor_count": len(DEFAULT_WEIGHTS),
        # 直接给出权重合计，避免模型自行求和时出现算术误差
        "active_weight_sum": _round(
            latest.get("active_weight"), 2
        ),
    }

    # 因子明细
    for row in describe_latest_factors(factor_df):
        key = row["factor"]

        brief["factors"].append(
            {
                "id": key,
                "label": row["label"],
                "score": row["value"],
                "available": row["available"],
                "direction": _direction(row["value"]),
                "weight": _round(effective_weights.get(key), 2),
                "meaning_when_positive": FACTOR_MEANING.get(key),
            }
        )

    brief["weights"] = {
        FACTOR_LABELS.get(key, key): _round(value, 2)
        for key, value in effective_weights.items()
    }

    # 数据质量诊断：说明哪些因子缺失、为什么缺失
    # Streamlit 缓存会丢失 DataFrame.attrs，因此允许调用方显式传入
    if diagnosis is None:
        try:
            diagnosis = (
                factor_df.attrs.get("inventory_diagnosis", {}) or {}
            )
        except AttributeError:
            diagnosis = {}

    coverage = {}
    for key in DEFAULT_WEIGHTS:
        if key in factor_df.columns:
            series = pd.to_numeric(factor_df[key], errors="coerce")
            coverage[FACTOR_LABELS.get(key, key)] = {
                "available_days": int(series.notna().sum()),
                "total_days": int(len(factor_df)),
            }

    brief["data_quality"] = {
        "inventory_report": {
            "total_reports": _safe_int(diagnosis.get("total_reports")),
            "trusted_reports": _safe_int(diagnosis.get("trusted_reports")),
            "required_reports": _safe_int(
                diagnosis.get("required_reports")
            ),
            "usable": bool(diagnosis.get("usable", False)),
            "sources": diagnosis.get("sources") or [],
            "metric_used": diagnosis.get("metric_used"),
            "lookback_days": _safe_int(diagnosis.get("lookback_days")),
            "source_discrepancies_count": len(
                diagnosis.get("source_discrepancies") or []
            ),
            "reason": diagnosis.get("reason") or None,
        },
        "factor_coverage": coverage,
        "note": (
            "宏观数据按公布滞后天数对齐，只使用当日已公布的观测值；"
            "库存数据只采用核验状态为 VERIFIED（跨源数值一致）或 "
            "ACCEPTED（单源通过全部自动校验）的记录，"
            "REVIEW 与 REJECTED 一律不参与打分。"
        ),
    }

    # 回测指标
    strategy = backtest.get("strategy_metrics", {}) if backtest else {}
    benchmark = backtest.get("benchmark_metrics", {}) if backtest else {}
    trades = backtest.get("trade_stats", {}) if backtest else {}

    def pack(metrics: dict) -> dict:
        return {
            "total_return_pct": _round(metrics.get("total_return_pct"), 2),
            "annual_return_pct": _round(metrics.get("annual_return_pct"), 2),
            "annual_vol_pct": _round(metrics.get("annual_vol_pct"), 2),
            "sharpe": _round(metrics.get("sharpe"), 2),
            "sortino": _round(metrics.get("sortino"), 2),
            "max_drawdown_pct": _round(
                metrics.get("max_drawdown_pct"), 2
            ),
            "calmar": _round(metrics.get("calmar"), 2),
            "win_rate_pct": _round(metrics.get("win_rate_pct"), 2),
            "sample_days": _safe_int(metrics.get("sample_days")),
        }

    brief["backtest"] = {
        "strategy": pack(strategy),
        "buy_and_hold": pack(benchmark),
        "trades": {
            "trade_count": _safe_int(trades.get("trade_count")),
            "exposure_pct": _round(trades.get("exposure_pct"), 2),
            "total_turnover": _round(trades.get("total_turnover"), 2),
            "total_cost_pct": _round(trades.get("total_cost_pct"), 3),
        },
        "caveats": [
            "样本区间较短，指标统计意义有限",
            "回测未包含滑点、资金费率与流动性冲击",
            "相关性不代表因果关系，历史表现不代表未来结果",
        ],
    }

    return brief


def recent_factor_history(factor_df: pd.DataFrame, days: int = 10) -> list:
    """输出最近若干日的得分轨迹，帮助模型描述趋势变化。"""

    if factor_df is None or factor_df.empty:
        return []

    tail = factor_df.tail(max(days, 1))

    history = []

    for date, row in tail.iterrows():
        history.append(
            {
                "date": date.strftime("%Y-%m-%d") if pd.notna(date) else None,
                "close": _round(row.get("close"), 2),
                "score": _round(row.get("score"), 3),
                "signal": _round(row.get("signal"), 2),
            }
        )

    return history


# =========================================================
# 4. 提示词组装
# =========================================================

def build_user_prompt(brief: dict, history: list = None) -> str:
    """把事实简报序列化为提示词正文。"""

    payload = {
        "fact_brief": brief,
        "recent_score_history": history or [],
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)

    return (
        "以下是本地量化引擎输出的事实简报（JSON）。\n"
        "请严格按照系统提示中的规则，用中文解读当前黄金多因子信号状态。\n\n"
        f"{text}\n"
    )


# =========================================================
# 5. 调用大模型
# =========================================================

def get_api_key() -> str:
    """读取密钥，缺失时返回空字符串。"""

    return (os.getenv("DEEPSEEK_API_KEY") or "").strip()


def call_model(
    brief: dict,
    history: list = None,
    model: str = None,
    temperature: float = 0.3,
    timeout: int = TIMEOUT,
) -> dict:
    """
    调用 DeepSeek 生成解读。

    返回 dict：
        {"ok": bool, "text": str, "mode": str, "model": str, "error": str|None}
    """

    api_key = get_api_key()

    target_model = model or DEFAULT_MODEL

    if not api_key:
        return {
            "ok": False,
            "text": "",
            "mode": "unavailable",
            "model": target_model,
            "error": "未配置 DEEPSEEK_API_KEY",
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": target_model,
        "temperature": temperature,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(brief, history),
            },
        ],
    }

    try:
        response = requests.post(
            CHAT_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return {
            "ok": False,
            "text": "",
            "mode": "error",
            "model": target_model,
            "error": f"网络请求失败：{exc}",
        }

    if not response.ok:
        return {
            "ok": False,
            "text": "",
            "mode": "error",
            "model": target_model,
            "error": (
                f"接口返回 HTTP {response.status_code}："
                f"{response.text[:300]}"
            ),
        }

    try:
        data = response.json()
    except ValueError:
        return {
            "ok": False,
            "text": "",
            "mode": "error",
            "model": target_model,
            "error": f"返回内容不是有效 JSON：{response.text[:300]}",
        }

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    if not content or not content.strip():
        return {
            "ok": False,
            "text": "",
            "mode": "error",
            "model": target_model,
            "error": f"接口返回空内容：{json.dumps(data, ensure_ascii=False)[:300]}",
        }

    usage = data.get("usage", {})

    return {
        "ok": True,
        "text": content.strip(),
        "mode": "model",
        "model": data.get("model", target_model),
        "error": None,
        "usage": usage,
    }


# =========================================================
# 6. 离线降级：规则化解读
# =========================================================

def rule_based_commentary(brief: dict) -> str:
    """在无法调用模型时，用本地规则生成结构化解读。"""

    lines = ["### 信号概览", ""]

    latest = brief.get("latest", {})
    sample = brief.get("sample", {})

    if not latest:
        return "暂无可用数据，无法生成解读。"

    score = latest.get("composite_score")
    direction = latest.get("score_direction", "数据缺失")

    lines.append(
        f"截至 {latest.get('date')}，黄金收盘价 "
        f"{latest.get('close')}，多因子综合得分 {score}，"
        f"方向判定为**{direction}**。"
    )

    lines.append(
        f"当前生效因子 {latest.get('active_factor_count')} / "
        f"{latest.get('total_factor_count')} 个，"
        f"模型信号 {latest.get('signal')}，"
        f"经波动率缩放后的目标仓位 {latest.get('position')}。"
    )

    lines.append("")
    lines.append("### 因子解读")
    lines.append("")

    factors = brief.get("factors", [])

    missing = []

    for group_name, keys in FACTOR_GROUPS.items():
        group_rows = [f for f in factors if f["id"] in keys]

        if not group_rows:
            continue

        lines.append(f"**{group_name}**")

        for row in group_rows:
            if not row["available"]:
                missing.append(row["label"])
                lines.append(f"- {row['label']}：数据缺失")
                continue

            meaning = row.get("meaning_when_positive") or ""
            score_value = row["score"]
            row_direction = row["direction"]

            note = meaning if score_value > 0 else ""

            if not note and score_value < 0:
                note = f"与「{meaning}」方向相反" if meaning else ""

            lines.append(
                f"- {row['label']}：{score_value}（{row_direction}）"
                + (f" — {note}" if note else "")
            )

        lines.append("")

    if missing:
        lines.append(
            f"> 缺失因子：{'、'.join(missing)}，已按可用权重重新归一化。"
        )

        inventory_report = (
            brief.get("data_quality", {}).get("inventory_report", {})
        )
        reason = inventory_report.get("reason")

        if reason:
            lines.append(f"> 缺失原因：{reason}")

        lines.append("")

    lines.append("### 风险与局限")
    lines.append("")

    backtest = brief.get("backtest", {})
    strategy = backtest.get("strategy", {})
    benchmark = backtest.get("buy_and_hold", {})

    if strategy:
        lines.append(
            f"- 回测样本 {strategy.get('sample_days')} 个交易日，"
            f"策略累计收益 {strategy.get('total_return_pct')}%，"
            f"买入持有 {benchmark.get('total_return_pct')}%；"
            f"最大回撤 {strategy.get('max_drawdown_pct')}%，"
            f"夏普 {strategy.get('sharpe')}。"
        )

    for caveat in backtest.get("caveats", []):
        lines.append(f"- {caveat}")

    lines.append("")
    lines.append(f"> {DISCLAIMER}")

    return "\n".join(lines)


def analyze(
    factor_df: pd.DataFrame,
    backtest: dict,
    weights: dict = None,
    params: dict = None,
    use_model: bool = True,
    model: str = None,
    temperature: float = 0.3,
    diagnosis: dict = None,
) -> dict:
    """
    对外主入口：生成简报并尝试取得 AI 解读。

    返回 dict：
        brief / history / text / mode / model / error / fallback
    """

    brief = build_fact_brief(
        factor_df, backtest, weights, params, diagnosis
    )
    history = recent_factor_history(factor_df)

    result = {
        "brief": brief,
        "history": history,
        "text": "",
        "mode": "fallback",
        "model": None,
        "error": None,
        "fallback": "",
    }

    result["fallback"] = rule_based_commentary(brief)

    if not use_model:
        result["text"] = result["fallback"]
        result["mode"] = "rule"
        return result

    response = call_model(
        brief,
        history=history,
        model=model,
        temperature=temperature,
    )

    result["model"] = response.get("model")
    result["error"] = response.get("error")

    if response.get("ok"):
        result["text"] = response["text"]
        result["mode"] = "model"
    else:
        result["text"] = result["fallback"]
        result["mode"] = "fallback"

    return result


if __name__ == "__main__":
    from database.db import get_connection
    from src.quant.factors import build_factors
    from src.quant.backtest import run_backtest

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

    frame = build_factors(gold, macro, comex)
    result_backtest = run_backtest(frame)

    output = analyze(frame, result_backtest, use_model=False)

    print(json.dumps(output["brief"], ensure_ascii=False, indent=2))
    print("\n---- 规则化解读 ----\n")
    print(output["text"])
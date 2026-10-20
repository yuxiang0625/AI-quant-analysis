# 本文件负责对因子信号做历史回测与绩效统计
"""
轻量级向量化回测引擎。

关键约定：
1. 仓位一律 shift(1)，即用昨日收盘后算出的信号交易今日，杜绝未来函数。
2. 交易成本按仓位变化量（换手）计提，双边均计。
3. 所有指标同时给出策略与「买入持有」基准，便于对照。
"""

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 252


# =========================================================
# 1. 收益与净值
# =========================================================

def build_returns(
    df: pd.DataFrame,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """根据仓位生成策略日收益、基准日收益与净值曲线。"""

    out = pd.DataFrame(index=df.index)

    daily_return = pd.to_numeric(
        df["daily_return"], errors="coerce"
    )

    position = pd.to_numeric(df["position"], errors="coerce")

    # 昨日仓位决定今日收益
    held_position = position.shift(1)

    out["benchmark_return"] = daily_return
    out["gross_return"] = held_position * daily_return

    turnover = position.diff().abs().fillna(position.abs())

    out["turnover"] = turnover.shift(1).fillna(0.0)
    out["cost"] = out["turnover"] * (cost_bps / 10_000)

    out["strategy_return"] = out["gross_return"] - out["cost"]

    out = out.dropna(subset=["strategy_return", "benchmark_return"])

    out["strategy_equity"] = (
        1 + out["strategy_return"]
    ).cumprod()

    out["benchmark_equity"] = (
        1 + out["benchmark_return"]
    ).cumprod()

    out["position_held"] = held_position.reindex(out.index)

    return out


def drawdown_series(equity: pd.Series) -> pd.Series:
    """由净值曲线计算回撤序列（百分比，负值）。"""

    if equity is None or equity.empty:
        return pd.Series(dtype="float64")

    running_max = equity.cummax()

    return (equity / running_max - 1) * 100


# =========================================================
# 2. 绩效指标
# =========================================================

def performance_metrics(
    returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """计算一组常用风险收益指标。"""

    returns = pd.to_numeric(returns, errors="coerce").dropna()

    if returns.empty:
        return {
            "total_return_pct": None,
            "annual_return_pct": None,
            "annual_vol_pct": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown_pct": None,
            "calmar": None,
            "win_rate_pct": None,
            "sample_days": 0,
        }

    equity = (1 + returns).cumprod()

    total_return = float(equity.iloc[-1] - 1)

    sample_days = int(len(returns))

    years = sample_days / periods_per_year

    if years > 0 and equity.iloc[-1] > 0:
        annual_return = float(equity.iloc[-1] ** (1 / years) - 1)
    else:
        annual_return = None

    std = float(returns.std(ddof=1)) if sample_days > 1 else 0.0

    annual_vol = std * np.sqrt(periods_per_year)

    if annual_vol > 0 and annual_return is not None:
        sharpe = float(annual_return / annual_vol)
    else:
        sharpe = None

    downside = returns[returns < 0]

    if len(downside) > 1:
        downside_std = float(downside.std(ddof=1)) * np.sqrt(periods_per_year)
    else:
        downside_std = 0.0

    if downside_std > 0 and annual_return is not None:
        sortino = float(annual_return / downside_std)
    else:
        sortino = None

    drawdown = drawdown_series(equity)

    max_drawdown = (
        float(drawdown.min()) if not drawdown.empty else None
    )

    if (
        max_drawdown is not None
        and max_drawdown < 0
        and annual_return is not None
    ):
        calmar = float(annual_return / abs(max_drawdown / 100))
    else:
        calmar = None

    traded = returns[returns != 0]

    win_rate = (
        float((traded > 0).mean() * 100) if not traded.empty else None
    )

    return {
        "total_return_pct": total_return * 100,
        "annual_return_pct": (
            annual_return * 100 if annual_return is not None else None
        ),
        "annual_vol_pct": annual_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown,
        "calmar": calmar,
        "win_rate_pct": win_rate,
        "sample_days": sample_days,
    }


def trade_statistics(returns_frame: pd.DataFrame) -> dict:
    """统计换手、持仓占比与交易次数。"""

    if returns_frame is None or returns_frame.empty:
        return {
            "trade_count": 0,
            "exposure_pct": None,
            "total_turnover": 0.0,
            "total_cost_pct": 0.0,
        }

    held = returns_frame["position_held"].fillna(0.0)

    changes = held.diff().fillna(held)

    trade_count = int((changes.abs() > 1e-9).sum())

    exposure = float((held.abs() > 1e-9).mean() * 100)

    total_turnover = float(returns_frame["turnover"].sum())

    total_cost = float(returns_frame["cost"].sum() * 100)

    return {
        "trade_count": trade_count,
        "exposure_pct": exposure,
        "total_turnover": total_turnover,
        "total_cost_pct": total_cost,
    }


# =========================================================
# 3. 主入口
# =========================================================

def run_backtest(df: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """执行回测并返回净值、回撤与全部指标。"""

    if df is None or df.empty:
        return {
            "returns": pd.DataFrame(),
            "strategy_drawdown": pd.Series(dtype="float64"),
            "benchmark_drawdown": pd.Series(dtype="float64"),
            "strategy_metrics": performance_metrics(pd.Series(dtype="float64")),
            "benchmark_metrics": performance_metrics(
                pd.Series(dtype="float64")
            ),
            "trade_stats": trade_statistics(pd.DataFrame()),
        }

    returns_frame = build_returns(df, cost_bps=cost_bps)

    strategy_drawdown = drawdown_series(
        returns_frame["strategy_equity"]
    )

    benchmark_drawdown = drawdown_series(
        returns_frame["benchmark_equity"]
    )

    return {
        "returns": returns_frame,
        "strategy_drawdown": strategy_drawdown,
        "benchmark_drawdown": benchmark_drawdown,
        "strategy_metrics": performance_metrics(
            returns_frame["strategy_return"]
        ),
        "benchmark_metrics": performance_metrics(
            returns_frame["benchmark_return"]
        ),
        "trade_stats": trade_statistics(returns_frame),
    }


if __name__ == "__main__":
    from database.db import get_connection
    from src.quant.factors import build_factors

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

    factor_frame = build_factors(gold, macro, comex)
    result = run_backtest(factor_frame)

    print("回测样本天数：", result["strategy_metrics"]["sample_days"])
    print("策略指标：", result["strategy_metrics"])
    print("基准指标：", result["benchmark_metrics"])
    print("交易统计：", result["trade_stats"])

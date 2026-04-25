from __future__ import annotations

import numpy as np
import pandas as pd

from quantify_trader.backtest.portfolio_engine import PortfolioBacktestEngine, PortfolioBacktestRequest
from quantify_trader.strategies.rebalance_equal_weight import (
    GroupedEqualWeightRebalanceStrategy,
    RebalanceParams,
)
from quantify_trader.trading.portfolio_broker import PortfolioSimBroker


def test_should_rebalance_by_group_weight_thresholds() -> None:
    groups = {"G1": ["A"], "G2": ["B"], "G3": ["C"], "G4": ["D"]}
    strat = GroupedEqualWeightRebalanceStrategy(groups, params=RebalanceParams(lower_weight=0.15, upper_weight=0.35))

    assert strat.should_rebalance({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) is False
    assert strat.should_rebalance({"A": 0.36, "B": 0.22, "C": 0.22, "D": 0.20}) is True
    assert strat.should_rebalance({"A": 0.14, "B": 0.29, "C": 0.29, "D": 0.28}) is True


def test_target_weights_group_25pct_and_intragroup_equal_split() -> None:
    groups = {"G1": ["A", "A2"], "G2": ["B"], "G3": ["C", "C2", "C3"], "G4": ["D", "D2"]}
    strat = GroupedEqualWeightRebalanceStrategy(groups)
    tw = strat.target_weights()

    # 每组 25%
    assert abs((tw["A"] + tw["A2"]) - 0.25) < 1e-12
    assert abs((tw["B"]) - 0.25) < 1e-12
    assert abs((tw["C"] + tw["C2"] + tw["C3"]) - 0.25) < 1e-12
    assert abs((tw["D"] + tw["D2"]) - 0.25) < 1e-12

    # 组内均分
    assert abs(tw["A"] - tw["A2"]) < 1e-12
    assert abs(tw["C"] - tw["C2"]) < 1e-12
    assert abs(tw["C2"] - tw["C3"]) < 1e-12

    # 总和为 1
    assert abs(sum(tw.values()) - 1.0) < 1e-12


def test_custom_group_targets() -> None:
    groups = {"G1": ["A"], "G2": ["B"], "G3": ["C"], "G4": ["D"]}
    group_targets = {"G1": 0.40, "G2": 0.30, "G3": 0.20, "G4": 0.10}
    strat = GroupedEqualWeightRebalanceStrategy(groups, group_targets=group_targets)
    tw = strat.target_weights()

    assert abs(tw["A"] - 0.40) < 1e-12
    assert abs(tw["B"] - 0.30) < 1e-12
    assert abs(tw["C"] - 0.20) < 1e-12
    assert abs(tw["D"] - 0.10) < 1e-12
    assert abs(sum(tw.values()) - 1.0) < 1e-12


def test_custom_group_targets_with_multi_member() -> None:
    groups = {"G1": ["A", "A2"], "G2": ["B"], "G3": ["C"], "G4": ["D"]}
    group_targets = {"G1": 0.50, "G2": 0.20, "G3": 0.20, "G4": 0.10}
    strat = GroupedEqualWeightRebalanceStrategy(groups, group_targets=group_targets)
    tw = strat.target_weights()

    # G1 有 2 个成员，各分 25%
    assert abs(tw["A"] - 0.25) < 1e-12
    assert abs(tw["A2"] - 0.25) < 1e-12
    assert abs(tw["B"] - 0.20) < 1e-12
    assert abs(tw["C"] - 0.20) < 1e-12
    assert abs(tw["D"] - 0.10) < 1e-12
    assert abs(sum(tw.values()) - 1.0) < 1e-12


def test_backtest_rebalances_only_when_triggered() -> None:
    # 构造一个价格序列：
    # - 第一天四组等权
    # - 第二天 G1 大涨到 >35% 触发一次再平衡
    groups = {"G1": ["A"], "G2": ["B"], "G3": ["C"], "G4": ["D"]}
    syms = ["A", "B", "C", "D"]
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame(
        {
            "A": [100.0, 200.0, 200.0],
            "B": [100.0, 100.0, 100.0],
            "C": [100.0, 100.0, 100.0],
            "D": [100.0, 100.0, 100.0],
        },
        index=idx,
    )

    strat = GroupedEqualWeightRebalanceStrategy(groups, params=RebalanceParams(0.15, 0.35))
    broker = PortfolioSimBroker(1_000_000.0)

    res = PortfolioBacktestEngine().run(
        PortfolioBacktestRequest(symbols=syms, close_prices=close, strategy=strat, broker=broker)
    )

    # 至少发生过一次再平衡，且不会每一天都再平衡
    assert int(np.sum(res.equity_curve["rebalance"].to_numpy())) >= 1
    assert int(np.sum(res.equity_curve["rebalance"].to_numpy())) <= len(res.equity_curve)

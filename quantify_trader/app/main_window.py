from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from quantify_trader.backtest.portfolio_engine import (
    PortfolioBacktestEngine,
    PortfolioBacktestRequest,
)
from quantify_trader.data.demo_data import load_real_close_data
from quantify_trader.monitor.waterline import WaterlineSnapshot
from quantify_trader.strategies.rebalance_equal_weight import (
    GroupedEqualWeightRebalanceStrategy,
    RebalanceParams,
)
from quantify_trader.trading.portfolio_broker import PortfolioSimBroker

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class UiState:
    years: int


class MainWindow(QMainWindow):
    _GROUPS: ClassVar[dict[str, list[str]]] = {
        "股票": ["510300"],
        "国债": ["511010"],
        "短债或货币基金": ["511880"],
        "黄金": ["518880"],
    }
    _SYMBOLS: ClassVar[list[str]] = ["510300", "511010", "511880", "518880"]
    _GROUP_NAMES: ClassVar[list[str]] = list(_GROUPS.keys())

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quantify Trader")
        self.resize(1200, 720)

        self._engine = PortfolioBacktestEngine()

        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        form = QFormLayout()

        self.portfolio = QComboBox()
        self.portfolio.addItems(["等权再平衡"])

        self.group_weight_spins: dict[str, QDoubleSpinBox] = {}
        group_weight_row = QVBoxLayout()
        group_weight_row.setSpacing(4)
        for g in self._GROUP_NAMES:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(2)
            spin.setValue(0.25)
            spin.setSuffix(f"  ({', '.join(self._GROUPS[g])})")
            self.group_weight_spins[g] = spin
            group_weight_row.addWidget(QLabel(g))
            group_weight_row.addWidget(spin)

        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setRange(0.0, 1.0)
        self.lower_spin.setSingleStep(0.01)
        self.lower_spin.setDecimals(2)
        self.lower_spin.setValue(0.20)

        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setRange(0.0, 1.0)
        self.upper_spin.setSingleStep(0.01)
        self.upper_spin.setDecimals(2)
        self.upper_spin.setValue(0.30)

        self.run_btn = QPushButton("运行回测")
        self.run_btn.clicked.connect(self._on_run_backtest)

        form.addRow("策略", self.portfolio)
        form.addRow("分组目标权重", group_weight_row)
        form.addRow("再平衡下限", self.lower_spin)
        form.addRow("再平衡上限", self.upper_spin)
        left_layout.addLayout(form)
        left_layout.addWidget(self.run_btn)

        self.metrics = QLabel("")
        self.metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left_layout.addWidget(self.metrics, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)

        pg.setConfigOptions(antialias=True)
        self.equity_plot = pg.PlotWidget(title="Equity Curve")
        self.equity_plot.showGrid(x=True, y=True, alpha=0.25)
        right_layout.addWidget(self.equity_plot, 1)

        self.drawdown_plot = pg.PlotWidget(title="Drawdown")
        self.drawdown_plot.showGrid(x=True, y=True, alpha=0.25)
        right_layout.addWidget(self.drawdown_plot, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([320, 880])

        root_layout = QHBoxLayout(root)
        root_layout.addWidget(splitter)

        self._on_run_backtest()

    def _on_run_backtest(self) -> None:
        close = load_real_close_data(_DATA_DIR)

        group_targets = {g: self.group_weight_spins[g].value() for g in self._GROUP_NAMES}
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()

        strategy = GroupedEqualWeightRebalanceStrategy(
            groups=self._GROUPS,
            group_targets=group_targets,
            params=RebalanceParams(lower_weight=lower, upper_weight=upper),
        )

        broker = PortfolioSimBroker(initial_cash=1_000_000.0, commission_rate=0.00015)

        req = PortfolioBacktestRequest(
            symbols=self._SYMBOLS,
            close_prices=close,
            strategy=strategy,
            broker=broker,
        )
        result = self._engine.run(req)

        snapshot = WaterlineSnapshot.from_equity_series(result.equity_curve["equity"])
        self._render(result.equity_curve, snapshot, result.stats)

    def _render(self, equity_curve: pd.DataFrame, snap: WaterlineSnapshot, stats: dict) -> None:
        self.equity_plot.clear()
        self.drawdown_plot.clear()

        x = np.arange(len(equity_curve), dtype=float)
        equity = equity_curve["equity"].to_numpy(dtype=float)
        dd = equity_curve["drawdown"].to_numpy(dtype=float)

        self.equity_plot.plot(x, equity, pen=pg.mkPen(color=(66, 133, 244), width=2))
        self.drawdown_plot.plot(x, dd, pen=pg.mkPen(color=(234, 67, 53), width=2))

        annual_returns = stats.get("annual_returns", {})
        annual_lines = [f"  {year}: {ret:+.2%}" for year, ret in annual_returns.items()]

        self.metrics.setText(
            "\n".join(
                [
                    f"初始资金: 1,000,000.00",
                    f"净值终值: {equity[-1]:,.2f}",
                    f"总收益率: {stats.get('total_return', float('nan')):.2%}",
                    f"年化收益: {stats.get('cagr', float('nan')):.2%}",
                    f"最大回撤: {snap.max_drawdown_pct:.2%}",
                    f"夏普比率: {stats.get('sharpe', float('nan')):.2f}",
                    f"交易次数: {stats.get('trades', 0)}",
                    f"再平衡次数: {stats.get('rebalances', 0)}",
                    f"回测天数: {len(equity_curve)}",
                    "",
                    "各年收益:",
                    *annual_lines,
                    "",
                    f"最终收益: {stats.get('total_return', float('nan')):.2%}",
                ]
            )
        )

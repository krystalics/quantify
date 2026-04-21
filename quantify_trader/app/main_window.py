from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from quantify_trader.backtest.portfolio_engine import (
    PortfolioBacktestEngine,
    PortfolioBacktestRequest,
)
from quantify_trader.data.demo_data import make_demo_multi_close
from quantify_trader.monitor.waterline import WaterlineSnapshot
from quantify_trader.strategies.rebalance_equal_weight import (
    GroupedEqualWeightRebalanceStrategy,
    RebalanceParams,
)
from quantify_trader.trading.portfolio_broker import PortfolioSimBroker


@dataclass(frozen=True)
class UiState:
    years: int


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quantify Trader")
        self.resize(1200, 720)

        self._engine = PortfolioBacktestEngine()
        self._groups: dict[str, list[str]] = {
            "G1": ["AAA", "AAB", "AAC"],
            "G2": ["BBA", "BBB"],
            "G3": ["CCA", "CCB", "CCC", "CCD"],
            "G4": ["DDA", "DDB"],
        }
        self._symbols = [s for members in self._groups.values() for s in members]

        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        form = QFormLayout()
        self.portfolio = QComboBox()
        self.portfolio.addItems(["等权再平衡(15%/35%触发 -> 25%)"])

        self.years = QSpinBox()
        self.years.setRange(1, 20)
        self.years.setValue(3)

        self.run_btn = QPushButton("运行回测")
        self.run_btn.clicked.connect(self._on_run_backtest)

        form.addRow("策略", self.portfolio)
        form.addRow("回测年数", self.years)
        form.addRow(
            "分组",
            QLabel(" | ".join([f"{g}: {', '.join(ms)}" for g, ms in self._groups.items()])),
        )
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

    def _get_state(self) -> UiState:
        return UiState(
            years=int(self.years.value()),
        )

    def _on_run_backtest(self) -> None:
        st = self._get_state()

        close = make_demo_multi_close(self._symbols, years=st.years, seed=7)
        strategy = GroupedEqualWeightRebalanceStrategy(
            groups=self._groups, params=RebalanceParams(lower_weight=0.15, upper_weight=0.35)
        )

        broker = PortfolioSimBroker(initial_cash=1_000_000.0)

        req = PortfolioBacktestRequest(
            symbols=self._symbols,
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

        self.metrics.setText(
            "\n".join(
                [
                    f"净值终值: {equity[-1]:,.2f}",
                    f"最大回撤: {snap.max_drawdown_pct:.2%}",
                    f"年化收益(近似): {stats.get('cagr', float('nan')):.2%}",
                    f"夏普(近似): {stats.get('sharpe', float('nan')):.2f}",
                    f"交易次数: {stats.get('trades', 0)}",
                    f"再平衡次数: {stats.get('rebalances', 0)}",
                ]
            )
        )


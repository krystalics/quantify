# Quantify Trader (个人量化交易台)

一个 **Python + Qt（PySide6）** 的最小可运行量化交易桌面应用骨架，面向个人交易者：

- 回测（可视化权益曲线）
- 风险控制（仓位/最大回撤/止损等规则入口）
- 资产水位监控（净值、回撤、风险暴露聚合）
- 自动交易（下单接口抽象，便于对接券商/模拟盘）

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m quantify_trader
```

## 目录结构

```text
quantify_trader/
  app/                 # Qt UI 与应用组装
  backtest/            # 回测引擎与结果对象
  risk/                # 风控规则与风控执行器
  monitor/             # 资产水位/风险指标监控聚合
  trading/             # 下单接口（模拟/实盘适配器）
  strategies/          # 策略示例
  data/                # 数据接口（示例为内置生成）
```

## 下一步建议

- 接入真实行情/历史数据源（Tushare/聚宽/券商API/本地CSV/Parquet）
- 将 `SimBroker` 替换为你的券商下单实现
- 丰富 `risk` 规则：单票上限、行业暴露、波动率目标、动态杠杆等


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RebalanceParams:
    lower_weight: float = 0.15
    upper_weight: float = 0.35


class GroupedEqualWeightRebalanceStrategy:
    """
    组合再平衡策略：
    - 固定 4 个分组（标签组），每组目标权重 25%
    - 触发条件：任一“组权重” > upper 或 < lower
    - 触发后：把每组拉回 25%，再在组内按成员数等分
    """

    def __init__(self, groups: dict[str, list[str]], params: RebalanceParams | None = None) -> None:
        if len(groups) != 4:
            raise ValueError("groups must contain exactly 4 groups")
        symbols: list[str] = []
        for g, members in groups.items():
            if not g:
                raise ValueError("group name cannot be empty")
            if not members:
                raise ValueError(f"group {g} has no members")
            symbols.extend(members)
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbol appears in multiple groups")

        self.groups = groups
        self.symbols = symbols
        self.params = params or RebalanceParams()

    def should_rebalance(self, weights: dict[str, float]) -> bool:
        group_weights = self._group_weights(weights)
        for gw in group_weights.values():
            if gw > self.params.upper_weight or gw < self.params.lower_weight:
                return True
        return False

    def target_weights(self) -> dict[str, float]:
        group_target = 0.25
        targets: dict[str, float] = {}
        for _, members in self.groups.items():
            w = group_target / float(len(members))
            for s in members:
                targets[s] = w
        return targets

    def _group_weights(self, weights: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for g, members in self.groups.items():
            out[g] = float(sum(float(weights.get(s, 0.0)) for s in members))
        return out


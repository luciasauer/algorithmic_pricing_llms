# src/environment/penalty_demand_environment.py
import logging
import numpy as np


class PenaltyDemandEnvironment:
    def __init__(
        self,
        name: str,
        description: str,
        penalty_lambda: float = 0.0622,
        logger: logging.Logger = None,
    ):
        self.name = name
        self.description = description
        self.penalty_lambda = penalty_lambda
        self.logger = logger or logging.getLogger("experiment_logger")

        self.round = 0
        self.c_series = None
        self.c = None  # fallback costs (static)
        self.nbr_agents = None

    def _current_c(
        self, c_override: np.ndarray = None, agent_order: list[tuple[str, int]] = None
    ) -> np.ndarray:
        if c_override is not None:
            return c_override
        elif self.c_series is not None:
            return np.array(
                [
                    self.c_series[idx][self.round - 1]
                    for _, idx in sorted(agent_order, key=lambda x: x[1])
                ]
            )
        elif self.c is not None:
            return np.array(
                [self.c[idx] for _, idx in sorted(agent_order, key=lambda x: x[1])]
            )
        else:
            raise ValueError(
                "No cost data available (c_override, c_series, or c must be set)."
            )

    def compute_quantities_and_profits(
        self,
        agent_order: list[tuple[str, int]],
        prices: dict[str, float],
        c_override: np.ndarray = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        try:
            sorted_agents = sorted(agent_order, key=lambda x: x[1])
            sorted_names = [name for name, _ in sorted_agents]
            price_values = np.array([prices[name] for name in sorted_names])
            cost_values = self._current_c(
                c_override=c_override, agent_order=agent_order
            )

            profits = {}
            quantities = {}

            for i, name in enumerate(sorted_names):
                P_i = price_values[i]
                MC_i = cost_values[i]
                S_i = self.market_share[i]

                # Exclude i-th price and market share
                mask = np.arange(len(self.market_share)) != i
                prices_excl_i = price_values[mask]
                weights_excl_i = self.market_share[mask]

                # Normalize weights to sum to 1
                normalized_weights = weights_excl_i / np.sum(weights_excl_i)

                P_avg_weighted_competitors = np.average(
                    prices_excl_i,
                    weights=normalized_weights,
                )

                penalty = np.exp(
                    -self.penalty_lambda * abs(P_i - P_avg_weighted_competitors)
                )
                profit = (P_i - MC_i) * S_i * penalty

                profits[name] = float(profit)
                quantities[name] = float(S_i * penalty)

                self.logger.info(
                    f"Agent: {name}, Price: {P_i}, MC: {MC_i}, S: {S_i}, P_avg_others: {P_avg_weighted_competitors}"
                )

            return quantities, profits

        except Exception as e:
            self.logger.error(
                f"Error computing quantities and profits in PenaltyDemandEnvironment: {e}"
            )
            raise

    def set_round(self, round_num: int):
        self.round = round_num
        if self.c_series is not None:
            self.c = self.c_series[:, round_num - 1]

    def register_time_series(self, c_series: np.ndarray):
        self.c_series = c_series

    def get_environment_params(self):
        return {
            "penalty_lambda": self.penalty_lambda,
            "description": self.description,
        }

    def _compute_benchmarks(self):
        pass  # Not applicable for this environment

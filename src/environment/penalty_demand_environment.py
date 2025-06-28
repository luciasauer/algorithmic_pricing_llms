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
        self.logger = logger or logging.getLogger("experiment_logger")
        self.penalty_lambda = penalty_lambda
        self.round = 0
        self.c_series = None
        self.c = None  # current marginal costs per agent
        self.nbr_agents = None

    def compute_quantities_and_profits(
        self, agent_order: list[tuple[str, int]], prices: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        try:
            # Sort by env_index
            sorted_agents = sorted(agent_order, key=lambda x: x[1])
            sorted_names = [name for name, _ in sorted_agents]
            price_values = np.array([prices[name] for name in sorted_names])
            if self.c_series is not None:
                cost_values = np.array(
                    [self.c_series[idx][self.round - 1] for _, idx in sorted_agents]
                )
                self.logger.info(f"ROUND MARGINAL COST: {cost_values}")
            else:
                cost_values = np.array([self.c[idx] for _, idx in sorted_agents])

            # cost_values = np.array([self.c[idx] for _, idx in sorted_agents]) #NOTE! CHANGE THIS!

            # Compute average competitor price for each agent
            profits = {}
            quantities = {}

            for i, name in enumerate(sorted_names):
                P_i = price_values[i]
                MC_i = cost_values[i]
                # All competitors
                other_prices = np.delete(price_values, i)
                P_others_avg = np.mean(other_prices)

                # s_i = 1 by default unless you want custom logic
                penalty = np.exp(-self.penalty_lambda * abs(P_i - P_others_avg))
                profit = (P_i - MC_i) * 1.0 * penalty  # NOTE!CHANGE THIS!

                profits[name] = float(profit)
                quantities[name] = penalty  # you can also model s_i as this

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
        pass  # No monopoly/Nash concept in this environment

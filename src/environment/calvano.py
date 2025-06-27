#src/environment/environment.py
import logging
import numpy as np
from src.environment.pricing_market_logic_multiproduct import (
    get_quantities,
    get_profits,
    get_monopoly_prices,
    get_nash_prices,
)

class CalvanoDemandEnvironment:
    def __init__(self, name: str, description: str, logger: logging.Logger = None):
        self.name = name
        self.description = description
        self.logger = logger or logging.getLogger("experiment_logger")

        self.a_0 = None
        self.a = None
        self.mu = None
        self.alpha = None
        self.beta = None
        self.sigma = None
        self.c = None
        self.group_idxs = None

        self.monopoly_prices = None
        self.nash_prices = None
        self.round = 0


    def compute_quantities_and_profits(self, agent_order: list[tuple[str, int]], prices: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        try:
            sorted_names = [name for name, _ in sorted(agent_order, key=lambda x: x[1])]
            price_values = [prices[name] for name in sorted_names]

            quantities = get_quantities(
                p=tuple(price_values), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )
            profits = get_profits(
                p=tuple(price_values), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, c=self.c, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            return {
                name: q for name, q in zip(sorted_names, quantities)
            }, {
                name: pi for name, pi in zip(sorted_names, profits)
            }

        except Exception as e:
            self.logger.error(f"Error computing quantities and profits: {e}")
            raise
    
    def _compute_benchmarks(self):
        try:
            # Monopoly solution
            self.monopoly_prices = get_monopoly_prices(
                a0=self.a_0, a=self.a, mu=self.mu, alpha=self.alpha,
                c=self.c, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            self.monopoly_quantities = get_quantities(
                p=tuple(self.monopoly_prices), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            self.monopoly_profits = get_profits(
                p=tuple(self.monopoly_prices), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, c=self.c, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            # Nash solution
            self.nash_prices = get_nash_prices(
                a0=self.a_0, a=self.a, mu=self.mu, alpha=self.alpha, c=self.c,
                multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            self.nash_quantities = get_quantities(
                p=tuple(self.nash_prices), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            self.nash_profits = get_profits(
                p=tuple(self.nash_prices), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, c=self.c, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )
        except Exception as e:
            self.logger.error(f"Error computing benchmarks when initializing environment: {e}")
            raise e
    
    def get_environment_params(self):
        return {
            "a_0": self.a_0,
            "a": self.a.tolist(),
            "mu": self.mu,
            "alpha": self.alpha.tolist(),
            "beta": self.beta,
            "sigma": self.sigma,
            "c": self.c.tolist(),
            "group_idxs": self.group_idxs,
            "monopoly_prices": self.monopoly_prices if self.monopoly_prices is not None else None,
            "monopoly_quantities": self.monopoly_quantities if self.monopoly_quantities is not None else None,
            "monopoly_profits": self.monopoly_profits if self.monopoly_profits is not None else None,
            "nash_prices": self.nash_prices if self.nash_prices is not None else None,
            "nash_quantities": self.nash_quantities if self.nash_quantities is not None else None,
            "nash_profits": self.nash_profits if self.nash_profits is not None else None,
        }

    def register_time_series(self, c_series: np.ndarray):
        """Supply full (num_agents, num_rounds) arrays of time-varying parameters"""
        self.c_series = c_series

    def set_round(self, round_num: int):
        """Update time-varying parameters before using them"""
        self.round = round_num
        if hasattr(self, 'c_series') and self.c_series is not None:
            self.c = self.c_series[:, round_num - 1]  # 0-indexed access

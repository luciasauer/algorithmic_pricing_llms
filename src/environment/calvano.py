# src/environment/environment.py
"""
Calvano demand environment implementation.

This module implements the market environment based on the demand specification
from Calvano et al. (2020). The environment manages market parameters, computes
quantities and profits using nested logit demand, and provides benchmarks for
monopoly and Nash equilibrium pricing.

References:
    Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).
    "Artificial intelligence, algorithmic pricing, and collusion."
    American Economic Review, 110(10), 3267-3297.
"""

import logging
import numpy as np
from src.environment.pricing_market_logic_multiproduct import (
    get_quantities,
    get_profits,
    get_monopoly_prices,
    get_nash_prices,
)


class CalvanoDemandEnvironment:
    """
    Market environment implementing Calvano et al. (2020) demand specification.

    This class manages the economic environment for the pricing simulation,
    including market parameters, demand calculations, and profit computations.
    It supports both static and time-varying cost structures.

    Attributes:
        name: Descriptive name for this environment instance
        description: Detailed description of the environment configuration
        logger: Logger instance for debugging and monitoring
        a_0: Outside option demand intercept (baseline utility)
        a: Array of demand intercepts for each product
        mu: Price sensitivity parameter (higher = more price sensitive)
        alpha: Array of quality/markup parameters for each product
        beta: Parameter for demand specification (if used)
        sigma: Within-group correlation parameter for nested logit
        c: Array of marginal costs (fallback for static costs)
        group_idxs: Array indicating which group each product belongs to
        c_series: Time series of costs for dynamic cost scenarios
        round: Current round number (1-indexed)
        monopoly_prices: Precomputed monopoly benchmark prices
        nash_prices: Precomputed Nash equilibrium benchmark prices
    """

    def __init__(self, name: str, description: str, logger: logging.Logger = None):
        """
        Initialize the Calvano demand environment.

        Args:
            name: Descriptive name for this environment instance
            description: Detailed description of the environment configuration
            logger: Logger instance for debugging and monitoring (optional)
        """
        self.name = name
        self.description = description
        self.logger = logger or logging.getLogger("experiment_logger")

        # Core parameters
        self.a_0 = None
        self.a = None
        self.mu = None
        self.alpha = None
        self.beta = None
        self.sigma = None
        self.c = None  # This will store fallback static values
        self.group_idxs = None

        # Time-series cost data
        self.c_series = None
        self.round = 0

        # Benchmarks
        self.monopoly_prices = None
        self.nash_prices = None

    def _current_c(self):
        """
        Returns the cost vector for the current round.
        Prioritizes time-series if available, else defaults to static `self.c`.
        """
        if self.c_series is not None:
            return self.c_series[:, self.round - 1]  # round is 1-based
        return self.c

    def compute_quantities_and_profits(
        self,
        agent_order: list[tuple[str, int]],
        prices: dict[str, float],
        c_override: np.ndarray = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        try:
            sorted_names = [name for name, _ in sorted(agent_order, key=lambda x: x[1])]
            price_values = [prices[name] for name in sorted_names]

            current_c = c_override if c_override is not None else self._current_c()

            quantities = get_quantities(
                p=tuple(price_values),
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )
            profits = get_profits(
                p=tuple(price_values),
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                c=current_c,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )

            return {name: q for name, q in zip(sorted_names, quantities)}, {
                name: pi for name, pi in zip(sorted_names, profits)
            }

        except Exception as e:
            self.logger.error(f"Error computing quantities and profits: {e}")
            raise

    def _compute_benchmarks(self):
        try:
            # Benchmarks use current round's cost
            current_c = self._current_c()

            self.monopoly_prices = get_monopoly_prices(
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                c=current_c,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )

            self.monopoly_quantities = get_quantities(
                p=tuple(self.monopoly_prices),
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )

            self.monopoly_profits = get_profits(
                p=tuple(self.monopoly_prices),
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                c=current_c,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )

            self.nash_prices = get_nash_prices(
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                c=current_c,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )

            self.nash_quantities = get_quantities(
                p=tuple(self.nash_prices),
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )

            self.nash_profits = get_profits(
                p=tuple(self.nash_prices),
                a0=self.a_0,
                a=self.a,
                mu=self.mu,
                alpha=self.alpha,
                c=current_c,
                multiplier=self.beta,
                sigma=self.sigma,
                group_idxs=self.group_idxs,
            )

        except Exception as e:
            self.logger.error(
                f"Error computing benchmarks when initializing environment: {e}"
            )
            raise e

    def set_round(self, round_num: int):
        """Set current round for cost lookup (if cost series is used)"""
        self.round = round_num

    def register_time_series(self, c_series: np.ndarray):
        self.c_series = c_series

    def get_environment_params(self):
        return {
            "a_0": self.a_0,
            "a": self.a.tolist(),
            "mu": self.mu,
            "alpha": self.alpha.tolist(),
            "beta": self.beta,
            "sigma": self.sigma,
            "c": self._current_c().tolist(),
            "group_idxs": self.group_idxs,
            "monopoly_prices": self.monopoly_prices,
            "monopoly_quantities": self.monopoly_quantities,
            "monopoly_profits": self.monopoly_profits,
            "nash_prices": self.nash_prices,
            "nash_quantities": self.nash_quantities,
            "nash_profits": self.nash_profits,
        }

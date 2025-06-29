# src/pricing_engine.py
"""
Optimized pricing engine for monopoly and Nash equilibrium calculations.
Includes caching, vectorization, and efficient parameter handling.
"""

import math
import random
import warnings
from math import exp, isclose
from typing import Dict, List, Optional, Tuple

import polars as pl
from scipy.optimize import minimize, minimize_scalar

warnings.filterwarnings(
    "ignore", category=UserWarning, module="scipy.optimize._differentiable_functions"
)

MAX_NUM_STEPS = 1000
NUM_OPT_REF_POINTS = 20


class MarketPricingEngine:
    """
    Efficient market pricing engine with caching and vectorized operations.
    """

    def __init__(self):
        self._monopoly_cache = {}
        self._nash_cache = {}

    @staticmethod
    def _create_cache_key(params: Dict) -> str:
        """Create a hashable cache key from parameters."""
        # Round to avoid floating point precision issues
        rounded_params = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                rounded_params[k] = tuple(
                    round(x, 8) if isinstance(x, float) else x for x in v
                )
            elif isinstance(v, float):
                rounded_params[k] = round(v, 8)
            else:
                rounded_params[k] = v
        return str(sorted(rounded_params.items()))

    def get_quantities(
        self,
        *,
        p: Tuple[float],
        a_0: float,
        a: Tuple[float],
        mu: float,
        alpha: Tuple[float],
        beta: float,
        sigma: float,
        group_idxs: Tuple[int],
        c: Tuple[float] = None,
    ) -> List[float]:
        """Calculate quantities with input validation."""
        assert len(a) == len(alpha) == len(p) == len(group_idxs)
        assert min(group_idxs) == 1
        assert set(group_idxs) == set(range(1, max(group_idxs) + 1))

        num_groups = max(group_idxs) + 1
        delta_0 = a_0 / mu
        deltas = [(ai - pi / alphai) / mu for ai, pi, alphai in zip(a, p, alpha)]

        # Calculate group weights
        group_weights = [exp(delta_0) / (1 - sigma)]
        for group_idx in sorted(set(group_idxs)):
            group_i_weight = sum(
                exp(deltai / (1 - sigma))
                for i, deltai in enumerate(deltas)
                if group_idxs[i] == group_idx
            )
            group_weights.append(group_i_weight)

        assert len(group_weights) == num_groups

        # Calculate selection probabilities
        total_weight = sum(Dg ** (1 - sigma) for Dg in group_weights)
        group_g_selection_probs = [
            Dg ** (1 - sigma) / total_weight for Dg in group_weights
        ]

        assert isclose(sum(group_g_selection_probs), 1)

        # Calculate conditional probabilities
        conditional_product_selection_probabilities = [
            exp(delta / (1 - sigma)) / group_weights[group_idxs[i]]
            for i, delta in enumerate(deltas)
        ]

        quantities = [
            beta * prob_j_given_g * group_g_selection_probs[group_idxs[j]]
            for j, prob_j_given_g in enumerate(
                conditional_product_selection_probabilities
            )
        ]

        return quantities

    def get_profits(self, *, p: Tuple[float], c: Tuple[float], **kwargs) -> List[float]:
        """Calculate profits given prices and costs."""
        q = self.get_quantities(p=p, **kwargs)
        alpha = kwargs["alpha"]
        profits = [
            qi * (pi / alphai - ci) for qi, pi, ci, alphai in zip(q, p, c, alpha)
        ]
        return profits

    def get_monopoly_prices_cached(self, **params) -> List[float]:
        """Get monopoly prices with caching."""
        cache_key = self._create_cache_key(params)

        if cache_key in self._monopoly_cache:
            return self._monopoly_cache[cache_key]

        result = self._compute_monopoly_prices(**params)
        self._monopoly_cache[cache_key] = result
        return result

    def _compute_monopoly_prices(
        self,
        *,
        a_0: float,
        a: Tuple[float],
        mu: float,
        alpha: Tuple[float],
        c: Tuple[float],
        beta: float,
        sigma: float,
        group_idxs: Tuple[int],
    ) -> List[float]:
        """Compute monopoly prices without caching."""

        def inv_profit(p):
            return -sum(
                self.get_profits(
                    p=p,
                    a_0=a_0,
                    a=a,
                    mu=mu,
                    alpha=alpha,
                    c=c,
                    beta=beta,
                    sigma=sigma,
                    group_idxs=group_idxs,
                )
            )

        initial_guess = [ci * alphai for ci, alphai in zip(c, alpha)]
        bounds = [(ci * alphai, None) for ci, alphai in zip(c, alpha)]

        result = minimize(
            inv_profit,
            initial_guess,
            bounds=bounds,
            method="trust-constr",
            options={"gtol": 1e-8, "maxiter": 5000, "finite_diff_rel_step": 1e-7},
        )

        return [float(x) for x in result.x]

    def get_nash_prices_cached(self, **params) -> List[float]:
        """Get Nash prices with caching."""
        cache_key = self._create_cache_key(params)

        if cache_key in self._nash_cache:
            return self._nash_cache[cache_key]

        result = self._compute_nash_prices(**params)
        self._nash_cache[cache_key] = result
        return result

    def _compute_nash_prices(
        self,
        *,
        a_0: float,
        a: Tuple[float],
        mu: float,
        alpha: Tuple[float],
        beta: float,
        sigma: float,
        group_idxs: Tuple[int],
        c: Tuple[float],
        seed: int = 0,
        eps: float = 1e-8,
    ) -> List[float]:
        """Compute Nash equilibrium prices."""
        rng = random.Random(seed)

        # Get bounds
        monopoly_prices = self.get_monopoly_prices_cached(
            a_0=a_0,
            a=a,
            mu=mu,
            alpha=alpha,
            c=c,
            beta=beta,
            sigma=sigma,
            group_idxs=group_idxs,
        )

        lower_bound = min(ci * alphai for ci, alphai in zip(c, alpha))
        upper_bound = max(monopoly_prices)

        num_agents = len(c)
        p = tuple(rng.uniform(lower_bound, upper_bound) for _ in range(num_agents))

        # Fixed point iteration
        for _ in range(MAX_NUM_STEPS):
            best_response_p = tuple(
                self._get_best_response(
                    p=p[:i] + (None,) + p[i + 1 :],
                    i=i,
                    a_0=a_0,
                    a=a,
                    mu=mu,
                    alpha=alpha,
                    c=c,
                    beta=beta,
                    sigma=sigma,
                    group_idxs=group_idxs,
                    lower_bound_pi=lower_bound,
                    upper_bound_pi=upper_bound,
                )[i]
                for i in range(num_agents)
            )

            if math.dist(p, best_response_p) <= eps:
                break
            p = best_response_p

        return [float(x) for x in p]

    def _get_best_response(
        self, *, p: Tuple[Optional[float]], i: int, **params
    ) -> List[float]:
        """Calculate best response for agent i."""
        assert p[i] is None

        def pi_inv_profit_given_p(pi):
            full_p = p[:i] + (pi,) + p[i + 1 :]
            quantities = self.get_quantities(
                p=full_p,
                **{
                    k: v
                    for k, v in params.items()
                    if k not in ["lower_bound_pi", "upper_bound_pi"]
                },
            )
            alpha = params["alpha"]
            c = params["c"]
            pi_profit = quantities[i] * (pi / alpha[i] - c[i])
            return -pi_profit

        result = minimize_scalar(
            pi_inv_profit_given_p,
            bounds=(params["lower_bound_pi"], params["upper_bound_pi"]),
        )

        return list(p[:i] + (result.x,) + p[i + 1 :])

    def clear_cache(self):
        """Clear all cached results."""
        self._monopoly_cache.clear()
        self._nash_cache.clear()


def compute_equilibrium_prices_vectorized(
    df: pl.DataFrame, params: Dict, engine: MarketPricingEngine
) -> pl.DataFrame:
    """
    Vectorized computation of monopoly and Nash prices for a DataFrame.

    Args:
        df: DataFrame with columns ['agent', 'round', 'marginal_cost', ...]
        params: Dictionary with market parameters
        engine: MarketPricingEngine instance

    Returns:
        DataFrame with added 'mono_p' and 'nash_p' columns
    """
    # Get unique cost combinations to avoid redundant calculations
    unique_costs = df.select("marginal_cost").unique().sort("marginal_cost")

    # Calculate prices for each unique cost
    cost_to_prices = {}

    for row in unique_costs.iter_rows():
        cost = row[0]

        # Update cost parameter
        current_params = params.copy()
        num_agents = len(params["agents"])
        current_params["c"] = tuple([cost] * num_agents)

        # Remove non-pricing parameters
        pricing_params = {
            k: v for k, v in current_params.items() if k not in ["agents", "env_index"]
        }

        mono_p = engine.get_monopoly_prices_cached(**pricing_params)[0]
        nash_p = engine.get_nash_prices_cached(**pricing_params)[0]

        cost_to_prices[cost] = {"mono_p": mono_p, "nash_p": nash_p}

    # Map prices back to original DataFrame efficiently
    cost_mapping_df = pl.DataFrame(
        [
            {
                "marginal_cost": cost,
                "mono_p": prices["mono_p"],
                "nash_p": prices["nash_p"],
            }
            for cost, prices in cost_to_prices.items()
        ]
    )

    # Join with original DataFrame
    result_df = df.join(cost_mapping_df, on="marginal_cost", how="left")

    return result_df

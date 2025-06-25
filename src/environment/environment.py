import logging
import numpy as np
from src.environment.pricing_market_logic_multiproduct import (
    get_quantities,
    get_profits,
    get_monopoly_prices,
    get_nash_prices,
)

class CalvanoDemandEnvironment:
    def __init__(self, name: str, description: str, nbr_agents: int, env_params: dict, logger: logging.Logger = None):
        self.name = name
        self.description = description
        self.nbr_agents = nbr_agents
        self.logger = logger or logging.getLogger("experiment_logger")

        self.a_0 = env_params.get("a_0", 0.0)
        self.a = env_params.get("a", np.ones(nbr_agents))
        self.mu = env_params.get("mu", 0.25)
        self.alpha = env_params.get("alpha", np.ones(nbr_agents))
        self.beta = env_params.get("beta", 100)
        self.sigma = env_params.get("sigma", 0.0)
        self.c = env_params.get("c", np.ones(nbr_agents))
        self.group_idxs = env_params.get("group_idxs", [])

        self.monopoly_prices = None
        self.nash_prices = None
        self._compute_benchmarks()


    def compute_quantities_and_profits(self, prices: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        try:
            # Ensure ordering by agent index
            agent_names = list(prices.keys())
            price_values = [prices[name] for name in agent_names]

            quantities = get_quantities(
                p=tuple(price_values), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            profits = get_profits(
                p=tuple(price_values), a0=self.a_0, a=self.a, mu=self.mu,
                alpha=self.alpha, c=self.c, multiplier=self.beta, sigma=self.sigma, group_idxs=self.group_idxs,
            )

            # Return results mapped back to agent names
            quantities_dict = {name: q for name, q in zip(agent_names, quantities)}
            profits_dict = {name: pi for name, pi in zip(agent_names, profits)}
            return quantities_dict, profits_dict
        except Exception as e:
            self.logger.error(f"Error computing quantities and profits: {e}")
            raise e
    
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

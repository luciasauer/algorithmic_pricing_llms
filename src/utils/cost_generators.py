# A file to introduce variable cost and cost shocks to the experiments

import numpy as np


def create_shock_series(
    n_agents, n_rounds, shock_series_rounds, base_cost=1.0, shock_magnitudes=None
):
    """
    Generate a cost series with multiple shocks over time.

    Parameters:
        n_agents (int): Number of agents.
        n_rounds (int): Total number of rounds.
        shock_series_rounds (list[int]): Rounds at which shocks occur.
        base_cost (float): The base cost before shocks.
        shock_magnitudes (list[float]): Shock magnitudes (can be positive or negative).

    Returns:
        np.ndarray: A (n_agents x n_rounds) array with costs over time.
    """

    if shock_magnitudes is None:
        shock_magnitudes = [0.3] * len(shock_series_rounds)

    if len(shock_series_rounds) != len(shock_magnitudes):
        raise ValueError(
            "shock_series_rounds and shock_magnitudes must have the same length"
        )

    cost_series = np.full((n_agents, n_rounds), base_cost)

    for round_idx, magnitude in sorted(zip(shock_series_rounds, shock_magnitudes)):
        if round_idx < 0 or round_idx >= n_rounds:
            print(f"Warning: Invalid round index {round_idx}. Skipping this shock.")
            continue  # Ignore invalid rounds
        cost_series[:, round_idx:] *= 1 + magnitude
        cost_series = np.maximum(cost_series, 0.0)  # Ensure cost never goes below zero

    return cost_series


# # Experiment 1A: Single Cost Shock
# def create_step_shock_series(
#     n_agents, n_rounds, shock_round=100, base_cost=1.0, shock_magnitude=0.3
# ):
#     cost_series = np.full((n_agents, n_rounds), base_cost)
#     cost_series[:, shock_round:] = base_cost + shock_magnitude
#     return cost_series


# # Experiment 1B: Multiple Shocks
# def create_multiple_shocks_series(n_agents, n_rounds):
#     cost_series = np.full((n_agents, n_rounds), 1.0)
#     # Shock 1: Round 100 (+30%)
#     cost_series[:, 100:150] = 1.3
#     # Shock 2: Round 200 (-20% from base)
#     cost_series[:, 200:250] = 0.8
#     # Recovery: Round 250
#     cost_series[:, 250:] = 1.0
#     return cost_series


# # Experiment 2A: Linear Trend
# def create_trend_series(n_agents, n_rounds, start_cost=1.0, end_cost=1.5):
#     trend = np.linspace(start_cost, end_cost, n_rounds)
#     return np.tile(trend, (n_agents, 1))


# # Experiment 2B: Cyclical Patterns
# def create_cyclical_series(n_agents, n_rounds, base=1.0, amplitude=0.2, period=50):
#     t = np.arange(n_rounds)
#     cycle = base + amplitude * np.sin(2 * np.pi * t / period)
#     return np.tile(cycle, (n_agents, 1))

"""
Visualization Utilities for Market Experiment Analysis

This module provides plotting functions for visualizing pricing dynamics,
quantities, and profits in multi-agent market experiments.
"""

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt


def plot_experiment_svg(
    df: pl.DataFrame,
    metadata: dict,
    save_path: Path,
    show_quantities: bool = False,
    show_profits: bool = False,
    plot_references: bool = True,
):
    """
    Create comprehensive experiment visualization with price dynamics.

    Generates time series plots showing price evolution, with optional
    quantity and profit panels. Includes theoretical benchmarks.

    Args:
        df: Experiment data containing agent prices and market outcomes
        metadata: Experiment metadata with environment parameters
        save_path: Path to save the generated SVG plot
        show_quantities: Whether to include quantity panel
        show_profits: Whether to include profit panel
        plot_references: Whether to show monopoly/Nash benchmarks
    """
    env_params = metadata.get("environment").get("environment_params")
    p_m = env_params.get("monopoly_prices")
    q_m = env_params.get("monopoly_quantities")
    pi_m = env_params.get("monopoly_profits")
    p_n = env_params.get("nash_prices")
    q_n = env_params.get("nash_quantities")
    pi_n = env_params.get("nash_profits")

    fig, axs = plt.subplots(
        1 + int(show_quantities) + int(show_profits),
        1,
        figsize=(10, 3 * (1 + int(show_quantities) + int(show_profits))),
        sharex=True,
    )

    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    # Prepare data
    df = df.with_columns(
        pl.col("agent_type").str.replace("_agent", "").alias("agent_type")
    )

    df_sorted = df.sort(["round", "agent"])
    # concat agent with agent_type
    df_sorted = df_sorted.with_columns(
        (pl.col("agent") + " (" + pl.col("agent_type") + ")").alias("agent")
    )
    rounds = df_sorted["round"].unique().to_list()
    agents = df_sorted["agent"].unique().to_list()
    agents.sort()
    colors = ["blue", "red", "orange", "purple", "cyan", "brown", "magenta", "gray"]

    # --- Price plot ---
    ax = axs[0]
    if p_m is not None and plot_references:
        ax.axhline(y=p_m[0], color="black", linestyle="--", alpha=0.6, label="$P^M$")
        ax.axhline(y=p_n[0], color="green", linestyle=":", alpha=0.9, label="$P^N$")
    for i, agent in enumerate(agents):
        prices = (
            df_sorted.filter(pl.col("agent") == agent).sort("round")["price"].to_list()
        )
        ax.plot(rounds, prices, label=agent, color=colors[i % len(colors)])
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True)

    # --- Quantity plot ---
    if show_quantities:
        ax = axs[1 if not show_profits else 1]
        if q_m is not None and plot_references:
            ax.axhline(
                y=q_m[0], color="black", linestyle="--", alpha=0.6, label="$Q^M$"
            )
            ax.axhline(y=q_n[0], color="green", linestyle=":", alpha=0.9, label="$Q^N$")
        for i, agent in enumerate(agents):
            quantities = (
                df_sorted.filter(pl.col("agent") == agent)
                .sort("round")["quantity"]
                .to_list()
            )
            ax.plot(rounds, quantities, label=agent, color=colors[i % len(colors)])
        ax.set_ylabel("Quantity")
        ax.legend(loc="upper left")
        ax.grid(True)

    # --- Profit plot ---
    if show_profits:
        idx = 2 if show_quantities else 1
        ax = axs[idx]
        if pi_m is not None and plot_references:
            ax.axhline(
                y=pi_m[0], color="black", linestyle="--", alpha=0.6, label="$\\pi^M$"
            )
            ax.axhline(
                y=pi_n[0], color="green", linestyle=":", alpha=0.9, label="$\\pi^N$"
            )
        for i, agent in enumerate(agents):
            profits = (
                df_sorted.filter(pl.col("agent") == agent)
                .sort("round")["profit"]
                .to_list()
            )
            ax.plot(rounds, profits, label=agent, color=colors[i % len(colors)])
        ax.set_ylabel("Profit")
        ax.set_xlabel("Round")
        ax.legend(loc="upper left")
        ax.grid(True)

    plt.suptitle(
        f"Experiment: {metadata.get('name', 'Unknown')}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(str(save_path.with_suffix(".svg")), format="svg")
    plt.close(fig)


def plot_real_data_svg(df: pl.DataFrame, metadata: dict, save_path: Path):
    from sklearn.preprocessing import normalize

    participants = ["BP", "Caltex", "Coles", "Woolworths"]
    shares = [0.22, 0.16, 0.14, 0.16]
    shares_norm = normalize(np.array(shares).reshape(1, -1), norm="l1").tolist()[0]
    market_shares = [
        {"agent": k, "market_share": v} for k, v in zip(participants, shares_norm)
    ]
    # create a df of market
    market_shares_df = pl.DataFrame(market_shares)
    # Prepare data
    df = (
        df.join(market_shares_df, how="left", on="agent")
        .with_columns(
            pl.col("agent_type").str.replace("_agent", "").alias("agent_type"),
            (((pl.col("price") / pl.col("marginal_cost")) - 1) * 100).alias("markup"),
            (
                (pl.col("price") - pl.col("marginal_cost")) * pl.col("market_share")
            ).alias("profit_real"),
        )
        .sort(["round", "agent"])
        .with_columns(
            (pl.col("agent") + " (" + pl.col("agent_type") + ")").alias("agent")
        )
    )
    marginal_cost = (
        df.group_by(["round"])
        .agg(pl.col("marginal_cost").min())["marginal_cost"]
        .to_numpy()
        .flatten()
    )

    rounds = df["round"].unique().to_list()
    agents = df["agent"].unique().to_list()
    # monopoly_prices = None
    # rounds_monopoly = None
    # if "monopoly_price" in df.columns:
    #     monopoly_prices = (
    #         df.filter(pl.col("monopoly_price") > 0)
    #         .group_by(["round"])
    #         .agg(pl.col("monopoly_price").max())["monopoly_price"]
    #         .to_numpy()
    #         .flatten()
    #     )
    #     rounds_monopoly = (
    #         df.filter(pl.col("monopoly_price") > 0)
    #         .select("round")
    #         .unique()
    #         .to_numpy()
    #         .flatten()
    #     )
    agents.sort()
    colors = ["blue", "red", "orange", "purple", "cyan", "brown", "magenta", "gray"]

    # ADD A PLOT IF THERE ARE REAL AGENTS
    real_agents = [agent for agent in agents if "fake" not in agent.lower()]

    fig, axs = plt.subplots(
        2 + int(len(real_agents) > 0),
        1,
        figsize=(12, 3 * (2 + int(len(real_agents) > 0))),
        sharex=True,
    )
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]
    # --- Price plot ---
    ax = axs[0]
    for i, agent in enumerate(agents):
        linestyle = (
            ":" if "fake" in agent.lower() and "bp" not in agent.lower() else "-"
        )
        prices = df.filter(pl.col("agent") == agent).sort("round")["price"].to_list()
        ax.plot(
            rounds,
            prices,
            label=agent,
            color=colors[i % len(colors)],
            linestyle=linestyle,
        )
    # if len(monopoly_prices) >0:
    #     ax.plot(rounds_monopoly, monopoly_prices, label='$P^M$', color='red', linestyle=':')
    ax.plot(rounds, marginal_cost, label="TGP", color="grey", linestyle="--", alpha=0.7)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True)

    # --- Environment Profits ---
    if real_agents:
        ax = axs[1]
        for i, agent in enumerate(agents):
            if "fake" in agent.lower():
                continue
            market_share = (
                df.filter(pl.col("agent") == agent).sort("round")["profit"].to_list()
            )
            ax.plot(rounds, market_share, label=agent, color=colors[i % len(colors)])
        ax.set_ylabel("Environment Profit")
        ax.legend(loc="upper left")
        ax.grid(True)

    # --- Profit plot --- NOTE! Later on, split them in plots per agent
    ax = axs[1 + int(len(real_agents) > 0)]
    for i, agent in enumerate(agents):
        linestyle = (
            ":" if "fake" in agent.lower() and "bp" not in agent.lower() else "-"
        )
        profit = (
            df.filter(pl.col("agent") == agent).sort("round")["profit_real"].to_list()
        )
        ax.plot(
            rounds,
            profit,
            label=agent,
            color=colors[i % len(colors)],
            linestyle=linestyle,
        )
    ax.set_ylabel("Profit Real ($-TGP)*share")
    ax.legend(loc="upper left")
    ax.grid(True)

    # # --- Markup plot --- NOTE! Later on, split them in plots per agent
    # ax = axs[2+int(len(real_agents)>0)]
    # for i, agent in enumerate(agents):
    #     linestyle = ':' if "fake" in agent.lower() and 'bp' not in agent.lower() else '-'
    #     markup = df.filter(pl.col("agent") == agent).sort("round")["markup"].to_list()
    #     ax.plot(rounds, markup, label=agent, color=colors[i % len(colors)], linestyle=linestyle)
    # #add a trend line for the markup
    # # trend = np.polyfit(rounds, df["markup"].to_numpy().flatten(), 1)
    # # trend_line = np.polyval(trend, rounds)
    # # ax.plot(rounds, trend_line, label="Trend", color="black", alpha=0.5)
    # ax.set_ylabel("Markup (%)")
    # ax.set_xlabel("Round")
    # ax.legend(loc='upper left')
    # ax.grid(True)

    plt.suptitle(
        f"Experiment: {metadata.get('name', 'Unknown')}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(str(save_path.with_suffix(".svg")), format="svg")
    plt.close(fig)


def plot_experiment_shocks_svg(
    df: pl.DataFrame,
    metadata: dict,
    save_path: Path,
    environment,
    show_quantities: bool = False,
    show_profits: bool = False,
    plot_references: bool = True,
):
    env_params = metadata.get("environment").get("environment_params")
    p_m = env_params.get("monopoly_prices")
    q_m = env_params.get("monopoly_quantities")
    pi_m = env_params.get("monopoly_profits")
    p_n = env_params.get("nash_prices")
    q_n = env_params.get("nash_quantities")
    pi_n = env_params.get("nash_profits")

    fig, axs = plt.subplots(
        1 + int(show_quantities) + int(show_profits),
        1,
        figsize=(10, 3 * (1 + int(show_quantities) + int(show_profits))),
        sharex=True,
    )

    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    # Prepare data
    df = df.with_columns(
        pl.col("agent_type").str.replace("_agent", "").alias("agent_type")
    )

    df_sorted = df.sort(["round", "agent"])
    # concat agent with agent_type
    df_sorted = df_sorted.with_columns(
        (pl.col("agent") + " (" + pl.col("agent_type") + ")").alias("agent")
    )
    rounds = df_sorted["round"].unique().to_list()
    agents = df_sorted["agent"].unique().to_list()
    agents.sort()
    colors = ["blue", "red", "orange", "purple", "cyan", "brown", "magenta", "gray"]

    num_rounds = df["round"].max()
    monopoly_prices_series = []
    nash_prices_series = []
    monopoly_quantities_series = []
    monopoly_profits_series = []
    nash_quantities_series = []
    nash_profits_series = []

    for round_num in range(1, num_rounds + 1):
        environment.set_round(round_num)
        environment._compute_benchmarks()
        monopoly_prices_series.append(
            environment.monopoly_prices[0]
        )  # or avg if multi-agent
        nash_prices_series.append(environment.nash_prices[0])
        monopoly_quantities_series.append(np.mean(environment.monopoly_quantities))
        monopoly_profits_series.append(np.mean(environment.monopoly_profits))
        nash_quantities_series.append(np.mean(environment.nash_quantities))
        nash_profits_series.append(np.mean(environment.nash_profits))

    # --- Price plot ---
    ax = axs[0]
    if plot_references:
        ax.plot(
            rounds,
            monopoly_prices_series,
            color="black",
            linestyle="--",
            alpha=0.6,
            label="$P^M$",
        )
        ax.plot(
            rounds,
            nash_prices_series,
            color="green",
            linestyle=":",
            alpha=0.9,
            label="$P^N$",
        )

    for i, agent in enumerate(agents):
        prices = (
            df_sorted.filter(pl.col("agent") == agent).sort("round")["price"].to_list()
        )
        ax.plot(rounds, prices, label=agent, color=colors[i % len(colors)])
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True)

    # --- Quantity plot ---
    if show_quantities:
        ax = axs[1 if not show_profits else 1]
        if plot_references:
            ax.plot(
                rounds,
                monopoly_quantities_series,
                color="black",
                linestyle="--",
                alpha=0.6,
                label="$Q^M$",
            )
            ax.plot(
                rounds,
                nash_quantities_series,
                color="green",
                linestyle=":",
                alpha=0.9,
                label="$Q^N$",
            )
        for i, agent in enumerate(agents):
            quantities = (
                df_sorted.filter(pl.col("agent") == agent)
                .sort("round")["quantity"]
                .to_list()
            )
            ax.plot(rounds, quantities, label=agent, color=colors[i % len(colors)])
        ax.set_ylabel("Quantity")
        ax.legend(loc="upper left")
        ax.grid(True)

    # --- Profit plot ---
    if show_profits:
        idx = 2 if show_quantities else 1
        ax = axs[idx]
        if plot_references:
            ax.plot(
                rounds,
                monopoly_profits_series,
                color="black",
                linestyle="--",
                alpha=0.6,
                label="$\\pi^M$",
            )
            ax.plot(
                rounds,
                nash_profits_series,
                color="green",
                linestyle=":",
                alpha=0.9,
                label="$\\pi^N$",
            )
        for i, agent in enumerate(agents):
            profits = (
                df_sorted.filter(pl.col("agent") == agent)
                .sort("round")["profit"]
                .to_list()
            )
            ax.plot(rounds, profits, label=agent, color=colors[i % len(colors)])
        ax.set_ylabel("Profit")
        ax.set_xlabel("Round")
        ax.legend(loc="upper left")
        ax.grid(True)

    plt.suptitle(
        f"Experiment: {metadata.get('name', 'Unknown')}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(str(save_path.with_suffix(".svg")), format="svg")
    plt.close(fig)

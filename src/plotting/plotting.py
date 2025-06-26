import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

def plot_experiment_svg(df: pl.DataFrame, metadata: dict, save_path: Path, 
                        show_quantities: bool = False, show_profits: bool = False, plot_references: bool = True):
    
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
                            sharex=True
                        )
    
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    # Prepare data
    df = df.with_columns(
        pl.col("agent_type").str.replace("_agent", "").alias("agent_type")
    )

    df_sorted = df.sort(["round", "agent"])
    #concat agent with agent_type
    df_sorted = (df_sorted
                 .with_columns(
        (pl.col("agent") + " (" + pl.col("agent_type") + ")").alias("agent")
    ))
    rounds = df_sorted["round"].unique().to_list()
    agents = df_sorted["agent"].unique().to_list()
    agents.sort()
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'brown', 'magenta', 'gray']

    # --- Price plot ---
    ax = axs[0]
    if p_m is not None and plot_references:
        ax.axhline(y=p_m[0], color='black', linestyle='--', alpha=0.6, label='$P^M$')
        ax.axhline(y=p_n[0], color='green', linestyle=':', alpha=0.9, label='$P^N$')
    for i, agent in enumerate(agents):
        prices = df_sorted.filter(pl.col("agent") == agent).sort("round")["price"].to_list()
        ax.plot(rounds, prices, label=agent, color=colors[i % len(colors)])
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    ax.grid(True)

    # --- Quantity plot ---
    if show_quantities:
        ax = axs[1 if not show_profits else 1]
        if q_m is not None and plot_references:
            ax.axhline(y=q_m[0], color='black', linestyle='--', alpha=0.6, label='$Q^M$')
            ax.axhline(y=q_n[0], color='green', linestyle=':', alpha=0.9, label='$Q^N$')
        for i, agent in enumerate(agents):
            quantities = df_sorted.filter(pl.col("agent") == agent).sort("round")["quantity"].to_list()
            ax.plot(rounds, quantities, label=agent, color=colors[i % len(colors)])
        ax.set_ylabel("Quantity")
        ax.legend(loc='upper left')
        ax.grid(True)

    # --- Profit plot ---
    if show_profits:
        idx = 2 if show_quantities else 1
        ax = axs[idx]
        if pi_m is not None and plot_references:
            ax.axhline(y=pi_m[0], color='black', linestyle='--', alpha=0.6, label='$\\pi^M$')
            ax.axhline(y=pi_n[0], color='green', linestyle=':', alpha=0.9, label='$\\pi^N$')
        for i, agent in enumerate(agents):
            profits = df_sorted.filter(pl.col("agent") == agent).sort("round")["profit"].to_list()
            ax.plot(rounds, profits, label=agent, color=colors[i % len(colors)])
        ax.set_ylabel("Profit")
        ax.set_xlabel("Round")
        ax.legend(loc='upper left')
        ax.grid(True)

    plt.suptitle(f"Experiment: {metadata.get('name', 'Unknown')}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(save_path.with_suffix(".svg")), format="svg")
    plt.close(fig)
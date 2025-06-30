import os
import sys

sys.path.append(os.path.abspath(".."))

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

# Inspiread by sns palletes: 'CMRmap', 'Set1', 'tab20'

# Define golden ratio for height1
golden_ratio = (5**0.5 - 1) / 2  # ≈ 0.618
FIG_WIDTH_IN = 170 / 25.4  # matches typical \linewidth in 12pt LaTeX article
FIG_HEIGHT_IN = FIG_WIDTH_IN * golden_ratio  # aesthetically pleasing height
SUPTITLE_FONTSIZE = 14
plt.rcParams.update(
    {
        # === Font settings ===
        # 'text.usetex': True,
        "font.family": "serif",
        "font.size": 8,  # Base font size
        "axes.labelsize": 8,  # Axis label font
        "axes.titlesize": 14,  # Title font size
        "xtick.labelsize": 8,  # X tick labels
        "ytick.labelsize": 8,  # Y tick labels
        "legend.fontsize": 8,  # Legend text size
        # === Figure settings ===
        "figure.figsize": (FIG_WIDTH_IN, FIG_HEIGHT_IN),  # Size in inches
        "figure.dpi": 300,  # High-res for export
        # === Line/Marker settings ===
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        # === Grid and style ===
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        # === Legend settings ===
        # 'legend.frameon': False,        # No frame (border)
        # 'legend.facecolor': 'none',     # Transparent background
        "legend.edgecolor": "none",  # No edge line (just in case)
        # === Save options ===
        "savefig.format": "svg",
        "savefig.bbox": "tight",  # Avoid extra whitespace
        "savefig.dpi": 300,  # High-res for export
    }
)
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        "#e41a1c",
        "#1f77b4",
        "#4daf4a",
        "#984ea3",
        "#ff7f0e",
        "#ffff3e",
        "#f781bf",
        "#999999",
    ]
)


OUPUT_PATH = Path("../latex/imgs/results/")
OUPUT_PATH.mkdir(parents=True, exist_ok=True)
INPUT_PATH = Path("../experiments_synthetic/experiments_runs/")


def plot_monopoly_experiment_svg(
    df: pl.DataFrame,
    title: str,
    metadata: dict,
    save_path: Path,
    show_quantities: bool = False,
    show_profits: bool = False,
    plot_references: bool = True,
    last_n_rounds: int = None,
    display: bool = False,
):
    env_params = metadata.get("environment").get("environment_params")
    p_m, q_m, pi_m = (
        env_params.get("monopoly_prices"),
        env_params.get("monopoly_quantities"),
        env_params.get("monopoly_profits"),
    )

    nrows = 1 + int(show_quantities) + int(show_profits)

    fig, axs = plt.subplots(nrows, 1, figsize=(FIG_WIDTH_IN, 2.5 * nrows), sharex=False)
    axs = np.atleast_1d(axs)

    # Clean and prepare data
    df = df.with_columns(
        pl.col("agent_type").str.replace("_agent", "").alias("agent_type")
    )
    df_sorted = df.sort(["round", "agent"]).with_columns(
        (pl.col("agent") + " (" + pl.col("agent_type") + ")").alias("agent")
    )
    if last_n_rounds is not None:
        df_sorted = df_sorted.filter(
            pl.col("round") >= (df_sorted["round"].max() - last_n_rounds + 1)
        )
    rounds = df_sorted["round"].unique().to_list()
    agents = sorted(df_sorted["agent"].unique().to_list())

    # Prepare color mapping for agents
    color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        agent: color_list[i % len(color_list)] for i, agent in enumerate(agents)
    }

    # Price plot
    ax = axs[0]
    if p_m is not None and plot_references:
        ax.axhline(
            y=p_m[0], color="black", linestyle="--", alpha=0.6, label="$P^M=P^N$"
        )
    for agent in agents:
        prices = (
            df_sorted.filter(pl.col("agent") == agent).sort("round")["price"].to_list()
        )
        ax.plot(rounds, prices, label=agent, color=color_map[agent])
    ax.fill_between(
        rounds,
        p_m[0] * 0.95,
        p_m[0] * 1.05,
        color=color_list[0],
        alpha=0.1,
        label="Convergence Area",
    )
    ax.set_ylabel("Price")
    ax.legend()
    ax.set_xlim(rounds[0] - 1, rounds[-1] + 1)
    ax.grid(True)
    if nrows <= 1:
        ax.set_xlabel("Round")

    # Profit plot
    if show_profits:
        ax = axs[1]
        if pi_m is not None and plot_references:
            ax.axhline(
                y=pi_m[0],
                color="black",
                linestyle="--",
                alpha=0.6,
                label="$\\pi^M = \\pi^N$",
            )
        for agent in agents:
            profits = (
                df_sorted.filter(pl.col("agent") == agent)
                .sort("round")["profit"]
                .to_list()
            )
            ax.plot(rounds, profits, label=agent, color=color_map[agent])
        ax.set_ylabel("Profit")
        if not show_quantities:
            ax.set_xlabel("Round")
        ax.legend()
        ax.set_xlim(rounds[0] - 1, rounds[-1] + 1)
        ax.grid(True)

    # Quantity plot
    if show_quantities:
        idx = 2 if show_profits else 1
        ax = axs[idx]
        if q_m is not None and plot_references:
            ax.axhline(
                y=q_m[0], color="black", linestyle="--", alpha=0.6, label="$Q^M = Q^N$"
            )
        for agent in agents:
            quantities = (
                df_sorted.filter(pl.col("agent") == agent)
                .sort("round")["quantity"]
                .to_list()
            )
            ax.plot(rounds, quantities, label=agent, color=color_map[agent])
        ax.set_ylabel("Quantity")
        ax.set_xlabel("Round")
        ax.legend()
        ax.set_xlim(rounds[0] - 1, rounds[-1] + 1)
        ax.grid(True)

    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path.with_suffix(".svg"))
    if display:
        plt.show()
    plt.close(fig)

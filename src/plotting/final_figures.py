import os
import sys

sys.path.append(os.path.abspath(".."))

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Inspiread by sns palletes: 'CMRmap', 'Set1', 'tab20'

# Define golden ratio for height1
golden_ratio = (5**0.5 - 1) / 2  # ≈ 0.618
FIG_WIDTH_IN = 170 / 25.4  # matches typical \linewidth in 12pt LaTeX article
FIG_HEIGHT_IN = FIG_WIDTH_IN * golden_ratio  # aesthetically pleasing height
SUPTITLE_FONTSIZE = 12
plt.rcParams.update(
    {
        # === Font settings ===
        # 'text.usetex': True,
        "font.family": "serif",
        "font.size": 8,  # Base font size
        "axes.labelsize": 8,  # Axis label font
        "axes.titlesize": 12,  # Title font size
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
        "#984ea3",
        "#ff7f0e",
        "#1f77b4",
        "#e41a1c",
        "#4daf4a",
        "#ffff3e",
        "#f781bf",
        "#999999",
    ]
)


OUPUT_PATH = Path("../latex/imgs/res/")
OUPUT_PATH.mkdir(parents=True, exist_ok=True)
INPUT_PATH = Path("../experiments_synthetic/experiments_runs/")


def plot_monopoly_experiment_svg(
    df: pl.DataFrame,
    title: str,
    monopoly_price: float,
    monopoly_quantity: float,
    monopoly_profit: float,
    save_path: Path,
    show_quantities: bool = False,
    show_profits: bool = False,
    plot_references: bool = True,
    last_n_rounds: int = None,
    display: bool = False,
):
    nrows = 1 + int(show_quantities) + int(show_profits)

    fig, axs = plt.subplots(nrows, 1, figsize=(FIG_WIDTH_IN, 3 * nrows), sharex=False)
    axs = np.atleast_1d(axs)

    # Clean and prepare data
    df = df.with_columns(
        pl.col("agent_type").str.replace("_agent", "").alias("agent_type")
    )
    df_sorted = df.sort(
        ["round", "alpha", "agent"], descending=[False, False, False]
    ).with_columns((pl.col("agent") + " (" + pl.col("agent_type") + ")").alias("agent"))
    if last_n_rounds is not None:
        df_sorted = df_sorted.filter(
            pl.col("round") >= (df_sorted["round"].max() - last_n_rounds + 1)
        )
    rounds = df_sorted["round"].unique().to_list()
    agents = df_sorted["agent"].unique().to_list()

    # Prepare color mapping for agents
    color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        agent: color_list[i % len(color_list)] for i, agent in enumerate(agents)
    }

    # Price plot
    ax = axs[0]
    if monopoly_price is not None and plot_references:
        ax.axhline(
            y=monopoly_price, color="black", linestyle="--", alpha=0.6, label="$P^M$"
        )
    for agent in agents:
        prices = (
            df_sorted.filter(pl.col("agent") == agent).sort("round")["price"].to_list()
        )
        ax.plot(rounds, prices, label=agent, color=color_map[agent])
    ax.fill_between(
        rounds,
        monopoly_price * 0.95,
        monopoly_price * 1.05,
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
        if monopoly_profit is not None and plot_references:
            ax.axhline(
                y=monopoly_profit,
                color="black",
                linestyle="--",
                alpha=0.6,
                label="$\\pi^M$",
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
        if monopoly_quantity is not None and plot_references:
            ax.axhline(
                y=monopoly_quantity,
                color="black",
                linestyle="--",
                alpha=0.6,
                label="$Q^M$",
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


def plot_duopoly_results_from_df(
    df,
    p_nash,
    p_m,
    pi_nash,
    pi_m,
    title="Figure 2: Duopoly Experiment Results",
    save_path=None,
):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    # get colors from matplotlib cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    df = (
        df.filter(pl.col("round").is_between(251, 300))
        .select(
            [
                "experiment_timestamp",
                "agent",
                "agent_prefix_type",
                "alpha",
                "chosen_price",
                "profit",
            ]
        )
        .with_columns(
            (pl.col("chosen_price") / pl.col("alpha")).alias("price_normalized"),
            (pl.col("profit") / pl.col("alpha")).alias("profit_normalized"),
        )
        .group_by(["experiment_timestamp", "agent", "agent_prefix_type"])
        .agg(
            pl.col("price_normalized").mean().round(2).alias("mean_price_normalized"),
            pl.col("profit").mean().round(2).alias("profit"),
        )
        .sort(["experiment_timestamp", "agent", "agent_prefix_type"])
        .pivot(
            index=["experiment_timestamp", "agent_prefix_type"],
            on="agent",
            values=["mean_price_normalized", "profit"],
        )
        .with_columns(
            (pl.col("profit_Firm A") - pl.col("profit_Firm B")).alias("pi_delta"),
            (pl.col("profit_Firm A") + pl.col("profit_Firm B")).alias("pi_sum"),
        )
    )
    # === Panel 1: Price comparison ===
    # Reference lines
    axs[0].axvline(p_nash, color="black", linestyle=":", linewidth=1)
    axs[0].axhline(p_nash, color="black", linestyle=":", linewidth=1)
    axs[0].axvline(p_m, color=colors[3], linestyle="--", linewidth=1)
    axs[0].axhline(p_m, color=colors[3], linestyle="--", linewidth=1)

    axs[0].scatter(
        df.filter((pl.col("agent_prefix_type") == "P1")).select(
            "mean_price_normalized_Firm A"
        ),
        df.filter((pl.col("agent_prefix_type") == "P1")).select(
            "mean_price_normalized_Firm B"
        ),
        color=colors[0],
        marker="o",
        label="P1 vs. P1",
    )
    axs[0].scatter(
        df.filter((pl.col("agent_prefix_type") == "P2")).select(
            "mean_price_normalized_Firm A"
        ),
        df.filter((pl.col("agent_prefix_type") == "P2")).select(
            "mean_price_normalized_Firm B"
        ),
        color=colors[1],
        marker="o",
        label="P2 vs. P2",
    )
    min_x = min(
        df["mean_price_normalized_Firm A"].min(),
        df["mean_price_normalized_Firm B"].min(),
    )
    max_x = max(
        df["mean_price_normalized_Firm A"].max(),
        df["mean_price_normalized_Firm B"].max(),
    )
    min_y = min(
        df["mean_price_normalized_Firm A"].min(),
        df["mean_price_normalized_Firm B"].min(),
    )
    max_y = max(
        df["mean_price_normalized_Firm A"].max(),
        df["mean_price_normalized_Firm B"].max(),
    )
    min_xy = min(min_x, min_y)
    max_xy = max(max_x, max_y)
    # Set axis limits based on data
    axs[0].set_xlim(min_xy * 0.975, max_xy * 1.025)
    axs[0].set_ylim(min_xy * 0.975, max_xy * 1.025)
    # Axis setup
    axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    # set x and y ticks each 0.2
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # Axis annotations (external, aligned with ticks)
    axs[0].annotate(
        r"$p^{Nash}$",
        xy=(p_nash, axs[0].get_ylim()[0]),
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
        color="black",
    )
    axs[0].annotate(
        r"$p^{Nash}$",
        xy=(axs[0].get_xlim()[0], p_nash),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color="black",
    )

    axs[0].annotate(
        r"$p^M$",
        xy=(p_m, axs[0].get_ylim()[0]),
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
        color=colors[3],
    )
    axs[0].annotate(
        r"$p^M$",
        xy=(axs[0].get_xlim()[0], p_m),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=colors[3],
    )

    axs[0].set_xlabel("Firm 1 average price (over periods 251-300)")
    axs[0].set_ylabel("Firm 2 average price (over periods 251-300)")
    axs[0].set_title("Pricing Behavior of Firms by Prefix Type")

    # === Panel 2: Profit comparison ===

    # Diagonal lines for π₁ = π^{Nash} and π₂ = π^{Nash}
    y_vals = np.linspace(2 * pi_nash, 2 * pi_m, 200)
    delta_1 = 2 * pi_nash - y_vals  # π₁ = π^{Nash}
    delta_2 = y_vals - 2 * pi_nash  # π₂ = π^{Nash}

    axs[1].plot(delta_1, y_vals, ":", color="black")
    axs[1].plot(delta_2, y_vals, ":", color="black")

    # Annotations
    axs[1].text(
        np.mean(delta_1) * 2,
        (2.2 * pi_nash),
        r"$\pi_1 = \pi^{Nash}$",
        color="black",
        fontsize=10,
    )
    axs[1].text(
        np.mean(delta_2),
        (2.2 * pi_nash),
        r"$\pi_2 = \pi^{Nash}$",
        color="black",
        fontsize=10,
    )
    axs[1].scatter(
        df.filter((pl.col("agent_prefix_type") == "P1")).select("pi_delta"),
        df.filter((pl.col("agent_prefix_type") == "P1")).select("pi_sum"),
        color=colors[0],
        marker="o",
        label="P1 vs. P1",
    )
    axs[1].scatter(
        df.filter((pl.col("agent_prefix_type") == "P2")).select("pi_delta"),
        df.filter((pl.col("agent_prefix_type") == "P2")).select("pi_sum"),
        color=colors[1],
        marker="o",
        label="P2 vs. P2",
    )

    # Monopoly profit line

    axs[1].set_title("Profit Results by Prefix Type")

    axs[1].set_xlabel(
        "Average difference in profits $\\pi_1 - \\pi_2$ (over periods 251-300)"
    )
    axs[1].set_ylabel("Average sum of profits $\\pi_1 + \\pi_2$(over periods 251-300)")
    axs[1].axhline(2 * pi_m, color=colors[3], linestyle="--", linewidth=1, alpha=0.8)
    axs[1].text(
        min(df["pi_delta"].min(), min(delta_1)) * 1.05,
        2 * pi_m,
        r"$\pi^M$",
        color=colors[3],
    )
    # === Legend outside below both plots ===Add commentMore actions
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_duopoly_results_from_df_asym(
    df,
    p_nash,
    p_m,
    pi_nash,
    pi_m,
    title="Figure 2: Duopoly Experiment Results",
    save_path=None,
):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    # get colors from matplotlib cycler
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    df = (
        df.filter(pl.col("round").is_between(251, 300))
        .select(
            [
                "experiment_timestamp",
                "agent",
                "agent_prefix_type",
                "alpha",
                "chosen_price",
                "profit",
            ]
        )
        .with_columns(
            (pl.col("chosen_price") / pl.col("alpha")).alias("price_normalized"),
            (pl.col("profit") / pl.col("alpha")).alias("profit_normalized"),
        )
        .group_by(["experiment_timestamp", "agent", "agent_prefix_type"])
        .agg(
            pl.col("price_normalized").mean().round(2).alias("mean_price_normalized"),
            pl.col("profit").mean().round(2).alias("profit"),
        )
        .sort(["experiment_timestamp", "agent", "agent_prefix_type"])
        .pivot(
            index=["experiment_timestamp", "agent_prefix_type"],
            on="agent",
            values=["mean_price_normalized", "profit"],
        )
        .with_columns(
            (pl.col("profit_Firm A") - pl.col("profit_Firm B")).alias("pi_delta"),
            (pl.col("profit_Firm A") + pl.col("profit_Firm B")).alias("pi_sum"),
        )
    )
    # === Panel 1: Price comparison ===
    # Reference lines
    axs[0].axvline(p_nash, color="black", linestyle=":", linewidth=1)
    axs[0].axhline(p_nash, color="black", linestyle=":", linewidth=1)
    axs[0].axvline(p_m, color=colors[3], linestyle="--", linewidth=1)
    axs[0].axhline(p_m, color=colors[3], linestyle="--", linewidth=1)

    axs[0].scatter(
        df.filter((pl.col("agent_prefix_type") == "P1")).select(
            "mean_price_normalized_Firm A"
        ),
        df.filter((pl.col("agent_prefix_type") == "P1")).select(
            "mean_price_normalized_Firm B"
        ),
        color=colors[0],
        marker="o",
        label="P1 vs. P1",
    )
    axs[0].scatter(
        df.filter((pl.col("agent_prefix_type") == "P2")).select(
            "mean_price_normalized_Firm A"
        ),
        df.filter((pl.col("agent_prefix_type") == "P2")).select(
            "mean_price_normalized_Firm B"
        ),
        color=colors[1],
        marker="o",
        label="P2 vs. P2",
    )
    min_x = min(
        df["mean_price_normalized_Firm A"].min(),
        df["mean_price_normalized_Firm B"].min(),
    )
    max_x = max(
        df["mean_price_normalized_Firm A"].max(),
        df["mean_price_normalized_Firm B"].max(),
    )
    min_y = min(
        df["mean_price_normalized_Firm A"].min(),
        df["mean_price_normalized_Firm B"].min(),
    )
    max_y = max(
        df["mean_price_normalized_Firm A"].max(),
        df["mean_price_normalized_Firm B"].max(),
    )
    min_xy = min(min_x, min_y)
    max_xy = max(max_x, max_y)
    # Set axis limits based on data
    axs[0].set_xlim(min_xy * 0.975, max_xy * 1.025)
    axs[0].set_ylim(min_xy * 0.975, max_xy * 1.025)
    # Axis setup
    axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    # set x and y ticks each 0.2
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # Axis annotations (external, aligned with ticks)
    axs[0].annotate(
        r"$p^{Nash}$",
        xy=(p_nash, axs[0].get_ylim()[0]),
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
        color="black",
    )
    axs[0].annotate(
        r"$p^{Nash}$",
        xy=(axs[0].get_xlim()[0], p_nash),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color="black",
    )

    axs[0].annotate(
        r"$p^M$",
        xy=(p_m, axs[0].get_ylim()[0]),
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
        color=colors[3],
    )
    axs[0].annotate(
        r"$p^M$",
        xy=(axs[0].get_xlim()[0], p_m),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=colors[3],
    )

    axs[0].set_xlabel("Firm 1 average price (over periods 251-300)")
    axs[0].set_ylabel("Firm 2 average price (over periods 251-300)")
    axs[0].set_title("Pricing Behavior of Firms by Prefix Type")

    # === Panel 2: Profit comparison ===

    # Diagonal lines for π₁ = π^{Nash} and π₂ = π^{Nash}
    y_vals = np.linspace(2 * pi_nash, 2 * pi_m, 200)
    delta_1 = 2 * pi_nash - y_vals  # π₁ = π^{Nash}
    delta_2 = y_vals - 2 * pi_nash  # π₂ = π^{Nash}

    axs[1].plot(delta_1, y_vals, ":", color="black")
    axs[1].plot(delta_2, y_vals, ":", color="black")

    # Annotations
    axs[1].text(
        np.mean(delta_1) * 2,
        (2.2 * pi_nash),
        r"$\pi_1 = \pi^{Nash}$",
        color="black",
        fontsize=10,
    )
    axs[1].text(
        np.mean(delta_2),
        (2.2 * pi_nash),
        r"$\pi_2 = \pi^{Nash}$",
        color="black",
        fontsize=10,
    )
    axs[1].scatter(
        df.filter((pl.col("agent_prefix_type") == "P1")).select("pi_delta"),
        df.filter((pl.col("agent_prefix_type") == "P1")).select("pi_sum"),
        color=colors[0],
        marker="o",
        label="P1 vs. P1",
    )
    axs[1].scatter(
        df.filter((pl.col("agent_prefix_type") == "P2")).select("pi_delta"),
        df.filter((pl.col("agent_prefix_type") == "P2")).select("pi_sum"),
        color=colors[1],
        marker="o",
        label="P2 vs. P2",
    )

    # Monopoly profit line

    axs[1].set_title("Profit Results by Prefix Type")

    axs[1].set_xlabel(
        "Average difference in profits $\\pi_1 - \\pi_2$ (over periods 251-300)"
    )
    axs[1].set_ylabel("Average sum of profits $\\pi_1 + \\pi_2$(over periods 251-300)")
    axs[1].axhline(2 * pi_m, color=colors[3], linestyle="--", linewidth=1, alpha=0.8)
    axs[1].text(
        min(df["pi_delta"].min(), min(delta_1)) * 1.05,
        2 * pi_m,
        r"$\pi^M$",
        color=colors[3],
    )
    # === Legend outside below both plots ===Add commentMore actions
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

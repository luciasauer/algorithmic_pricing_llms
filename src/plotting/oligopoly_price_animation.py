#!/usr/bin/env python3
"""
Matplotlib animation showing normalized price evolution of 5 agents in oligopoly competition.
Shows each firm with their prompt type from a single run.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Set up style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Agent colors - different colors for each firm
AGENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
AGENT_NAMES = ["Firm A", "Firm B", "Firm C", "Firm D", "Firm E"]


def load_and_process_data():
    """Load and process the oligopoly data for a single run."""
    # Load data
    df = pl.read_parquet("data/results/all_experiments.parquet")

    # Filter for 5-agent oligopoly data
    oligopoly_5 = df.filter(pl.col("num_agents") == 5)

    # Get a single run (first experiment_timestamp)
    single_run = oligopoly_5.filter(
        pl.col("experiment_timestamp")
        == oligopoly_5["experiment_timestamp"].unique().head(2).tail(1)[0]
    )

    print(
        f"Using experiment: {single_run['experiment_name'].unique().head(2).tail(1)[0]}"
    )
    print(
        f"Experiment timestamp: {single_run['experiment_timestamp'].unique().head(2).tail(1)[0]}"
    )

    # Calculate normalized prices (price / alpha)
    single_run = single_run.with_columns(
        [
            (pl.col("chosen_price") / pl.col("alpha")).alias("normalized_price"),
            (pl.col("nash_prices") / pl.col("alpha")).alias("normalized_nash"),
            (pl.col("monopoly_prices") / pl.col("alpha")).alias("normalized_monopoly"),
        ]
    )

    # Get reference prices
    nash_price = single_run["normalized_nash"].unique().head(2).tail(1)[0]
    monopoly_price = single_run["normalized_monopoly"].unique().head(2).tail(1)[0]

    # Prepare price data for animation
    price_data = {}
    for agent in AGENT_NAMES:
        agent_data = single_run.filter(pl.col("agent") == agent)
        if len(agent_data) > 0:
            rounds = agent_data["round"].to_numpy()
            prices = agent_data["normalized_price"].to_numpy()
            prompt_type = agent_data["agent_prefix_type"].unique().head(2).tail(1)[0]
            price_data[f"{agent} ({prompt_type})"] = (rounds, prices)

    return price_data, nash_price, monopoly_price


def create_animation():
    """Create the price evolution animation."""
    # Load data
    price_data, nash_price, monopoly_price = load_and_process_data()

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 300)

    # Determine y-axis limits
    all_prices = []
    for rounds, prices in price_data.values():
        all_prices.extend(prices)

    y_min = min(min(all_prices), nash_price, monopoly_price) * 0.9
    y_max = max(max(all_prices), nash_price, monopoly_price) * 1.1
    ax.set_ylim(y_min, y_max)

    # Set labels and title
    ax.set_xlabel("Period", fontsize=14)
    ax.set_ylabel("Normalized Price (P/α)", fontsize=14)
    ax.set_title(
        "5-Agent Oligopoly: Normalized Price Evolution by Firm & Prompt Type",
        fontsize=16,
        fontweight="bold",
    )

    # Add reference lines
    ax.axhline(
        y=nash_price,
        color="gray",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Nash Equilibrium",
    )
    ax.axhline(
        y=monopoly_price,
        color="orange",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Monopoly Price",
    )

    # Initialize line objects for each agent
    lines = {}
    agent_keys = list(price_data.keys())
    for i, agent_key in enumerate(agent_keys):
        (line,) = ax.plot([], [], color=AGENT_COLORS[i], linewidth=2.5, label=agent_key)
        lines[agent_key] = line

    # Add legend
    ax.legend(loc="upper right", fontsize=11)

    # Period text
    period_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def animate(frame):
        """Animation function."""
        current_period = frame + 1

        # Update each agent's line
        for agent_key in agent_keys:
            if agent_key in price_data:
                rounds, prices = price_data[agent_key]

                # Find data up to current period
                mask = rounds <= current_period
                if np.any(mask):
                    x_data = rounds[mask]
                    y_data = prices[mask]
                    lines[agent_key].set_data(x_data, y_data)

        # Update period text
        period_text.set_text(f"Period: {current_period}")

        return list(lines.values()) + [period_text]

    # Create animation
    frames = 300  # Total periods
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=50, blit=True, repeat=True
    )

    # Add final statistics text
    final_stats = []
    final_stats.append(f"Nash Equilibrium: {nash_price:.3f}")
    final_stats.append(f"Monopoly Price: {monopoly_price:.3f}")
    final_stats.append("\nFinal Prices:")

    for agent_key in agent_keys:
        if agent_key in price_data:
            final_price = price_data[agent_key][1][-1]  # Last price
            final_stats.append(f"{agent_key}: {final_price:.3f}")

    stats_text = "\n".join(final_stats)
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    return fig, anim


def save_animation_gif(filename="oligopoly_firms_evolution.gif"):
    """Save the animation as a GIF."""
    fig, anim = create_animation()

    # Save as GIF
    print(f"Saving animation as {filename}...")
    anim.save(filename, writer="pillow", fps=10)
    print("Animation saved successfully!")

    return anim


def generate_beamer_frames(output_dir="latex/imgs/beamer_frames", frame_interval=1):
    """Generate individual PNG files for each frame for Beamer animation."""
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    price_data, nash_price, monopoly_price = load_and_process_data()

    # Get agent keys
    agent_keys = list(price_data.keys())

    # Generate frames
    max_periods = 300
    frame_count = 0

    print(f"Generating frames every {frame_interval} period(s) for smooth animation...")

    for period in range(frame_interval, max_periods + 1, frame_interval):
        # Create figure for this frame
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, 300)

        # Determine y-axis limits
        all_prices = []
        for rounds, prices in price_data.values():
            all_prices.extend(prices)

        y_min = min(min(all_prices), nash_price, monopoly_price) * 0.9
        y_max = max(max(all_prices), nash_price, monopoly_price) * 1.1
        ax.set_ylim(y_min, y_max)

        # Set labels and title
        ax.set_xlabel("Period", fontsize=14)
        ax.set_ylabel("Normalized Price (P/α)", fontsize=14)
        ax.set_title(
            "5-Agent Oligopoly: Normalized Price Evolution",  # by Firm & Prompt Type",
            fontsize=16,
            fontweight="bold",
        )

        # Add reference lines
        ax.axhline(
            y=nash_price,
            color="gray",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            label="Nash Equilibrium",
        )
        ax.axhline(
            y=monopoly_price,
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            label="Monopoly Price",
        )

        # Plot data up to current period
        for i, agent_key in enumerate(agent_keys):
            if agent_key in price_data:
                rounds, prices = price_data[agent_key]

                # Find data up to current period
                mask = rounds <= period
                if np.any(mask):
                    x_data = rounds[mask]
                    y_data = prices[mask]
                    ax.plot(
                        x_data,
                        y_data,
                        color=AGENT_COLORS[i],
                        linewidth=2.5,
                        label=agent_key,
                        alpha=0.8,
                    )

                    # Add current point
                    if len(x_data) > 0:
                        ax.scatter(
                            x_data[-1],
                            y_data[-1],
                            color=AGENT_COLORS[i],
                            s=50,
                            zorder=5,
                        )

        # Add legend
        # ax.legend(loc="upper right", fontsize=11)
        # Add legend below the plot in 2 rows, 4 columns
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.125),
            ncol=4,
            fontsize=11,
            frameon=True,
            shadow=True,
        )
        ax.grid(True, alpha=0.3)

        # Add prominent period counter in top middle
        period_text = f"Period: {period}"
        ax.text(
            0.5,
            0.98,
            period_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # # Add statistics text in top-left corner
        # final_stats = []
        # final_stats.append(f"Nash Equilibrium: {nash_price:.3f}")
        # final_stats.append(f"Monopoly Price: {monopoly_price:.3f}")
        # final_stats.append("\nCurrent Prices:")
        #
        # for agent_key in agent_keys:
        #     if agent_key in price_data:
        #         rounds, prices = price_data[agent_key]
        #         # Find price at current period
        #         current_mask = rounds <= period
        #         if np.any(current_mask):
        #             current_price = prices[current_mask][-1]
        #             final_stats.append(f"{agent_key}: {current_price:.3f}")
        #
        # stats_text = "\n".join(final_stats)
        # ax.text(
        #     0.02,
        #     0.98,
        #     stats_text,
        #     transform=ax.transAxes,
        #     fontsize=10,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        #     bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        # )
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()

        # Save frame
        frame_filename = f"{output_dir}/frame_{frame_count:03d}.png"
        plt.savefig(frame_filename, dpi=300, bbox_inches="tight")
        plt.close()

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Generated {frame_count} frames...")

    print(f"Generated {frame_count} frames in {output_dir}/")
    print(
        f"Use in Beamer with: \\animategraphics[loop,controls,width=\\textwidth]{{10}}{{{output_dir}/frame_}}{{000}}{{{frame_count - 1:03d}}}"
    )

    return frame_count


if __name__ == "__main__":
    # Create both animated and static versions
    print("Creating oligopoly price evolution visualizations...")

    # Generate Beamer frames
    print("\nGenerating frames for Beamer animation...")
    frame_count = generate_beamer_frames()

    # Create animated GIF
    try:
        # anim = save_animation_gif()
        print("Beamer frames created successfully!")
    except Exception as e:
        print(f"Could not create animation: {e}")
        print("Static plot and Beamer frames created successfully!")

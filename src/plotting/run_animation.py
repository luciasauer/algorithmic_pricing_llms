#!/usr/bin/env python3
"""
Matplotlib animation showing normalized price evolution of 5 agents in oligopoly competition.
Shows each firm with their prompt type from a single run.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Define golden ratio for height
golden_ratio = (5**0.5 - 1) / 2  # ≈ 0.618
FIG_WIDTH_IN = 170 / 25.4  # matches typical \linewidth in 12pt LaTeX article
FIG_HEIGHT_IN = FIG_WIDTH_IN * golden_ratio  # aesthetically pleasing height
SUPTITLE_FONTSIZE = 12

# Set up consistent styling to match other plots
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
        "legend.edgecolor": "none",  # No edge line (just in case)
        # === Save options ===
        "savefig.format": "svg",
        "savefig.bbox": "tight",  # Avoid extra whitespace
        "savefig.dpi": 300,  # High-res for export
    }
)

# Set consistent color palette
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

# Agent colors - using the consistent color palette
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
AGENT_COLORS = colors[:5]  # Use first 5 colors from the palette
AGENT_NAMES = ["Firm A", "Firm B", "Firm C", "Firm D", "Firm E"]


def load_and_process_data():
    """Load and process the oligopoly data for a single run."""
    # Load data
    df = pl.read_parquet("data/results/all_experiments.parquet")

    # Filter for 5-agent oligopoly data
    oligopoly_5 = df.filter(pl.col("num_agents") == 5)

    # Get a single run (first experiment_timestamp)
    single_run = oligopoly_5.filter(pl.col("experiment_timestamp") == "1751280311")

    print("Using experiment: 1751280311")
    print("Experiment timestamp: 1751280311")

    # Calculate normalized prices (price / alpha)
    single_run = single_run.with_columns(
        [
            (pl.col("chosen_price") / pl.col("alpha")).alias("normalized_price"),
            (pl.col("nash_prices") / pl.col("alpha")).alias("normalized_nash"),
            (pl.col("monopoly_prices") / pl.col("alpha")).alias("normalized_monopoly"),
        ]
    )

    # Get reference prices
    nash_price = single_run.filter(pl.col("experiment_timestamp") == "1751280311")[
        "normalized_nash"
    ].first()
    monopoly_price = single_run.filter(pl.col("experiment_timestamp") == "1751280311")[
        "normalized_monopoly"
    ].first()
    # Prepare price data for animation
    price_data = {}
    for agent in AGENT_NAMES:
        agent_data = single_run.filter(pl.col("agent") == agent)
        if len(agent_data) > 0:
            rounds = agent_data["round"].to_numpy()
            prices = agent_data["normalized_price"].to_numpy()
            prompt_type = agent_data.filter(
                pl.col("experiment_timestamp") == "1751280311"
            )["agent_prefix_type"].first()
            price_data[f"{agent} ({prompt_type})"] = (rounds, prices)

    return price_data, nash_price, monopoly_price


def create_animation():
    """Create the price evolution animation."""
    # Load data
    price_data, nash_price, monopoly_price = load_and_process_data()

    # Set up the figure and axis with consistent styling
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_xlim(0, 300)

    # Determine y-axis limits
    all_prices = []
    for rounds, prices in price_data.values():
        all_prices.extend(prices)

    y_min = min(min(all_prices), nash_price, monopoly_price) * 0.9
    y_max = max(max(all_prices), nash_price, monopoly_price) * 1.1
    ax.set_ylim(y_min, y_max)

    # Set labels and title with consistent font sizes
    ax.set_xlabel("Period", fontsize=8)
    ax.set_ylabel("Normalized Price (P/α)", fontsize=8)
    ax.set_title(
        "5-Agent Oligopoly: Normalized Price Evolution by Firm & Prompt Type",
        fontsize=12,
        fontweight="bold",
    )

    # Add reference lines with consistent styling
    ax.axhline(
        y=nash_price,
        color="gray",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label="Nash Equilibrium",
    )
    ax.axhline(
        y=monopoly_price,
        color="orange",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label="Monopoly Price",
    )

    # Initialize line objects for each agent
    lines = {}
    agent_keys = list(price_data.keys())
    for i, agent_key in enumerate(agent_keys):
        (line,) = ax.plot([], [], color=AGENT_COLORS[i], linewidth=1.5, label=agent_key)
        lines[agent_key] = line

    # Add legend with consistent styling
    ax.legend(loc="upper right", fontsize=8)

    # Grid styling is already set in rcParams
    ax.grid(True, alpha=0.4, linestyle="--")

    # Period text
    period_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=8,
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
        fontsize=8,
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


def generate_beamer_frames(
    output_dir="latex/slides_pricing_collusion/imgs/beamer_frames", frame_interval=1
):
    """Generate individual PDF files for each frame for Beamer animation."""
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
        # Create figure for this frame with consistent styling
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        ax.set_xlim(0, 300)

        # Determine y-axis limits
        all_prices = []
        for rounds, prices in price_data.values():
            all_prices.extend(prices)

        y_min = min(min(all_prices), nash_price, monopoly_price) * 0.9
        y_max = max(max(all_prices), nash_price, monopoly_price) * 1.1
        ax.set_ylim(y_min, y_max)

        # Set labels and title with consistent font sizes
        ax.set_xlabel("Period", fontsize=8)
        ax.set_ylabel("Normalized Price (P/α)", fontsize=8)
        ax.set_title(
            "5-Agent Oligopoly: Normalized Price Evolution",
            fontsize=12,
            fontweight="bold",
        )

        # Add reference lines with consistent styling
        ax.axhline(
            y=nash_price,
            color="gray",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label="Nash Equilibrium",
        )
        ax.axhline(
            y=monopoly_price,
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
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
                        linewidth=1.5,
                        label=agent_key,
                        alpha=0.8,
                    )

                    # Add current point
                    if len(x_data) > 0:
                        ax.scatter(
                            x_data[-1],
                            y_data[-1],
                            color=AGENT_COLORS[i],
                            s=16,  # Adjusted marker size to match rcParams
                            zorder=5,
                        )

        # Add legend with consistent styling
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.125),
            ncol=4,
            fontsize=8,
            frameon=True,
            shadow=True,
        )

        # Grid styling is already set in rcParams
        ax.grid(True, alpha=0.4, linestyle="--")

        # Add prominent period counter in top middle
        period_text = f"Period: {period}"
        ax.text(
            0.5,
            0.98,
            period_text,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()

        # Save frame with consistent DPI and format
        frame_filename = f"{output_dir}/frame_{frame_count:03d}.pdf"
        plt.savefig(frame_filename, bbox_inches="tight")
        plt.close()

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Generated {frame_count} frames...")

    print(f"Generated {frame_count} frames in {output_dir}/")
    print(
        f"Use in Beamer with: \\animategraphics[loop,controls,width=\\textwidth]{{10}}{{{output_dir}/frame_}}{{000}}"
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

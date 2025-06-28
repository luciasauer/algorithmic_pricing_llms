# src/visualization.py
"""
Visualization utilities for market simulation analysis using seaborn and matplotlib.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


class MarketVisualization:
    """
    Comprehensive visualization suite for market simulation analysis.
    """

    def __init__(
        self,
        style: str = "darkgrid",
        palette: str = "mako",
        figsize: Tuple[int, int] = (16, 8),
    ):
        """
        Initialize visualization settings.

        Args:
            style: Seaborn style
            palette: Color palette
            figsize: Default figure size
        """
        sns.set_style(style)
        sns.set_palette(palette)
        self.default_figsize = figsize

        # Define color schemes for different plot types
        self.colors = {
            "monopoly": "#e74c3c",  # Red
            "nash": "#3498db",  # Blue
            "marginal_cost": "#95a5a6",  # Gray
            "agent_prices": "mako",  # Multi-color for agents
            "empirical": "#2ecc71",  # Green
        }

    def plot_price_evolution(
        self,
        df: pl.DataFrame,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_equilibria: bool = True,
        agent_columns: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Plot price evolution over time with equilibrium references.

        Args:
            df: DataFrame with columns [round, price, agent, mono_p, nash_p, marginal_cost]
            title: Plot title
            save_path: Path to save figure
            show_equilibria: Whether to show monopoly/Nash lines
            agent_columns: Specific agent columns to plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Convert to pandas for seaborn compatibility
        df_pandas = df.to_pandas()

        # Plot agent prices
        if agent_columns:
            # Plot specific agent columns
            for col in agent_columns:
                if col in df_pandas.columns:
                    sns.lineplot(
                        data=df_pandas, x="round", y=col, label=col, linewidth=2, ax=ax
                    )
        else:
            # Standard agent price plotting
            sns.lineplot(
                data=df_pandas, x="round", y="price", hue="agent", linewidth=2, ax=ax
            )

        if show_equilibria:
            # Add equilibrium reference lines
            if "mono_p" in df_pandas.columns:
                sns.lineplot(
                    data=df_pandas,
                    x="round",
                    y="mono_p",
                    color=self.colors["monopoly"],
                    linestyle="-.",
                    linewidth=1,
                    label="Monopoly Price",
                    ax=ax,
                )

            if "nash_p" in df_pandas.columns:
                sns.lineplot(
                    data=df_pandas,
                    x="round",
                    y="nash_p",
                    color=self.colors["nash"],
                    linestyle="--",
                    linewidth=1,
                    label="Nash Price",
                    ax=ax,
                )

            if "marginal_cost" in df_pandas.columns:
                sns.lineplot(
                    data=df_pandas,
                    x="round",
                    y="marginal_cost",
                    color=self.colors["marginal_cost"],
                    linestyle=":",
                    linewidth=1,
                    label="Marginal Cost",
                    ax=ax,
                )

        # Add vertical line at round=0 if relevant
        if "round" in df_pandas.columns and df_pandas["round"].min() <= 0:
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=1)
            ax.text(
                0.1,
                ax.get_ylim()[1] * 0.9,
                "Agent Active",
                rotation=90,
                verticalalignment="top",
            )

        # Customize plot
        ax.set_xlabel("Round")
        ax.set_ylabel("Price")
        if title:
            ax.set_title(title)
        ax.legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_fuel_market_dynamics(
        self,
        daily_data: pl.DataFrame,
        brands: List[str],
        title: str = "Perth Fuel Market Dynamics",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot fuel market price dynamics with TGP reference.

        Args:
            daily_data: DataFrame with date, tgpmin, and brand price columns
            brands: List of brand column names
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        df_pandas = daily_data.to_pandas()

        # Plot 1: Price levels
        if "tgpmin" in df_pandas.columns:
            sns.lineplot(
                data=df_pandas,
                x="date",
                y="tgpmin",
                color=self.colors["marginal_cost"],
                linewidth=2,
                label="Terminal Gate Price",
                ax=ax1,
            )

        for i, brand in enumerate(brands):
            if brand in df_pandas.columns:
                sns.lineplot(
                    data=df_pandas,
                    x="date",
                    y=brand,
                    linewidth=1.5,
                    label=brand,
                    ax=ax1,
                )

        ax1.set_ylabel("Price (cents per litre)")
        ax1.set_title(f"{title} - Price Levels")
        ax1.legend()

        # Plot 2: Margins (if TGP available)
        if "tgpmin" in df_pandas.columns:
            for brand in brands:
                if brand in df_pandas.columns:
                    margin_data = df_pandas[brand] - df_pandas["tgpmin"]
                    ax2.plot(
                        df_pandas["date"],
                        margin_data,
                        linewidth=1.5,
                        label=f"{brand} Margin",
                    )

            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax2.set_ylabel("Margin over TGP (cents)")
            ax2.set_title("Retail Margins over Terminal Gate Price")
            ax2.legend()

        ax2.set_xlabel("Date")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_coordination_analysis(
        self,
        df: pl.DataFrame,
        agents: List[str],
        window: int = 30,
        title: str = "Price Coordination Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Analyze and visualize price coordination patterns.

        Args:
            df: DataFrame with agent price data
            agents: List of agent names
            window: Rolling window for correlation calculation
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

        df_pandas = df.to_pandas()

        # 1. Price levels over time
        for agent in agents:
            if agent in df_pandas.columns:
                sns.lineplot(data=df_pandas, x="round", y=agent, label=agent, ax=ax1)
        ax1.set_title("Price Levels Over Time")
        ax1.set_ylabel("Price")
        ax1.legend()

        # 2. Price differences from mean
        if len(agents) > 1:
            agent_prices = df_pandas[agents]
            mean_price = agent_prices.mean(axis=1)

            for agent in agents:
                if agent in df_pandas.columns:
                    price_diff = df_pandas[agent] - mean_price
                    ax2.plot(df_pandas["round"], price_diff, label=agent)

            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax2.set_title("Deviations from Average Price")
            ax2.set_ylabel("Price Difference")
            ax2.legend()

        # 3. Rolling correlation (if multiple agents)
        if len(agents) >= 2:
            agent_data = df_pandas[agents].dropna()
            rolling_corr = agent_data.rolling(window=window).corr().dropna()

            # Plot correlation between first two agents
            if len(agents) >= 2:
                corr_series = rolling_corr.loc[:, (agents[0], agents[1])]
                ax3.plot(
                    corr_series.reset_index(level=1, drop=True).index,
                    corr_series.values,
                )
                ax3.set_title(f"Rolling Correlation ({agents[0]} vs {agents[1]})")
                ax3.set_ylabel("Correlation")
                ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # 4. Price dispersion over time
        if len(agents) > 1:
            agent_prices = df_pandas[agents]
            price_std = agent_prices.std(axis=1, skipna=True)
            price_range = agent_prices.max(axis=1) - agent_prices.min(axis=1)

            ax4.plot(
                df_pandas["round"], price_std, label="Standard Deviation", alpha=0.7
            )
            ax4.plot(df_pandas["round"], price_range, label="Price Range", alpha=0.7)
            ax4.set_title("Price Dispersion Over Time")
            ax4.set_ylabel("Price Dispersion")
            ax4.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_profit_analysis(
        self,
        df: pl.DataFrame,
        cost_column: str = "marginal_cost",
        title: str = "Profit Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Analyze and visualize profit dynamics.

        Args:
            df: DataFrame with price, quantity, and cost data
            cost_column: Name of the cost column
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

        df_pandas = df.to_pandas()

        # Calculate profits if not already present
        if "profit" not in df_pandas.columns and "price" in df_pandas.columns:
            if cost_column in df_pandas.columns:
                df_pandas["margin"] = df_pandas["price"] - df_pandas[cost_column]
                if "quantity" in df_pandas.columns:
                    df_pandas["profit"] = df_pandas["margin"] * df_pandas["quantity"]

        # 1. Profit evolution by agent
        if "profit" in df_pandas.columns and "agent" in df_pandas.columns:
            sns.lineplot(data=df_pandas, x="round", y="profit", hue="agent", ax=ax1)
            ax1.set_title("Profit Evolution by Agent")
            ax1.set_ylabel("Profit")

        # 2. Margin evolution
        if "margin" in df_pandas.columns or cost_column in df_pandas.columns:
            if "margin" in df_pandas.columns:
                sns.lineplot(data=df_pandas, x="round", y="margin", hue="agent", ax=ax2)
            else:
                margin_data = df_pandas["price"] - df_pandas[cost_column]
                df_pandas["margin"] = margin_data
                sns.lineplot(data=df_pandas, x="round", y="margin", hue="agent", ax=ax2)

            ax2.set_title("Price Margin Evolution")
            ax2.set_ylabel("Margin")

        # 3. Cumulative profits
        if "profit" in df_pandas.columns and "agent" in df_pandas.columns:
            for agent in df_pandas["agent"].unique():
                agent_data = df_pandas[df_pandas["agent"] == agent].copy()
                agent_data["cumulative_profit"] = agent_data["profit"].cumsum()
                ax3.plot(
                    agent_data["round"], agent_data["cumulative_profit"], label=agent
                )

            ax3.set_title("Cumulative Profits")
            ax3.set_ylabel("Cumulative Profit")
            ax3.legend()

        # 4. Profit distribution
        if "profit" in df_pandas.columns:
            sns.boxplot(data=df_pandas, x="agent", y="profit", ax=ax4)
            ax4.set_title("Profit Distribution by Agent")
            ax4.set_ylabel("Profit")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def create_dashboard(
        self, df: pl.DataFrame, output_dir: str, prefix: str = "market_analysis"
    ) -> Dict[str, str]:
        """
        Create a complete dashboard of visualizations.

        Args:
            df: Main analysis DataFrame
            output_dir: Directory to save plots
            prefix: Prefix for saved files

        Returns:
            Dictionary mapping plot names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_plots = {}

        # Price evolution plot
        fig1 = self.plot_price_evolution(df, title="Market Price Evolution")
        path1 = output_path / f"{prefix}_price_evolution.svg"
        fig1.savefig(path1, bbox_inches="tight")
        plt.close(fig1)
        saved_plots["price_evolution"] = str(path1)

        # Coordination analysis (if multiple agents)
        if "agent" in df.columns:
            agents = df["agent"].unique().to_list()
            if len(agents) > 1:
                # Create pivot for coordination analysis
                df_pivot = df.pivot(index="round", on="agent", values="price")
                fig2 = self.plot_coordination_analysis(df_pivot, agents)
                path2 = output_path / f"{prefix}_coordination.svg"
                fig2.savefig(path2, bbox_inches="tight")
                plt.close(fig2)
                saved_plots["coordination"] = str(path2)

        # Profit analysis
        if "profit" in df.columns:
            fig3 = self.plot_profit_analysis(df)
            path3 = output_path / f"{prefix}_profits.svg"
            fig3.savefig(path3, bbox_inches="tight")
            plt.close(fig3)
            saved_plots["profits"] = str(path3)

        print(f"Dashboard created in {output_dir}")
        return saved_plots


def quick_plot(
    df: pl.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Quick plotting function for exploratory analysis.

    Args:
        df: Polars DataFrame
        x: X-axis column
        y: Y-axis column
        hue: Column for color grouping
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    df_pandas = df.to_pandas()

    sns.lineplot(data=df_pandas, x=x, y=y, hue=hue, ax=ax)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig

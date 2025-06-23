"""
Complete Historical Market Environment for LLM-Based Pricing Agents
Uses Polars data loader and Mistral API handling with rate limiting
"""

import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mistralai import Mistral

# Add utils to path
sys.path.append(os.path.abspath(os.path.join("../..")))
from src.utils.data_loader import (
    get_retail_price_for_period,
    get_tgp_for_period,
    load_retail_data,
    load_tgp_data,
)

warnings.filterwarnings("ignore", category=SyntaxWarning)


@dataclass
class MarketState:
    """Represents the market state at a given time period"""

    date: datetime
    period: int
    tgp_cost: float
    competitor_prices: Dict[str, float]
    market_demand: float

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "date": self.date.isoformat(),
            "period": self.period,
            "tgp_cost": self.tgp_cost,
            "competitor_prices": self.competitor_prices,
            "market_demand": self.market_demand,
        }


class HistoricalMarketEnvironment:
    """
    Creates a market environment based on real Australia fuel market data
    """

    def __init__(
        self,
        tgp_data_path: str = "../../data/113176-V1/data/TGP/tgpmin.csv",
        retail_data_folder: str = "../../data/113176-V1/data/Prices/",
        start_date: str = "2009-09-01",
        end_date: str = "2010-08-16",
    ):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.start_date_str = start_date

        print("Loading market data with Polars...")

        # Load data using Polars data loader
        self.tgp_data = load_tgp_data(
            file_path=tgp_data_path, start_date=self.start_date, end_date=self.end_date
        )
        print(f"Loaded {len(self.tgp_data)} TGP records")

        self.retail_data = load_retail_data(
            file_path=retail_data_folder,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        print(f"Loaded {len(self.retail_data)} retail price records")
        print(
            f"Brands found: {self.retail_data['BRAND_DESCRIPTION'].unique().to_list()}"
        )

        # Calculate total periods
        self.total_periods = (self.end_date - self.start_date).days + 1
        self.current_period = 0

        print(f"Market environment initialized: {self.total_periods} periods")

    def get_current_market_state(self) -> Optional[MarketState]:
        """Get the current market state"""
        if self.current_period >= self.total_periods:
            return None

        # Calculate current date
        current_date = self.start_date + timedelta(days=self.current_period)

        # Get TGP for this period using data loader
        tgp_cost = get_tgp_for_period(
            self.tgp_data, self.start_date_str, self.current_period
        )

        # Get retail prices for this period
        retail_period_data = get_retail_price_for_period(
            self.retail_data, self.start_date_str, self.current_period
        )

        # Convert to competitor prices dict
        competitor_prices = {}
        if len(retail_period_data) > 0:
            # Group by brand and take mean price
            brand_prices = (
                retail_period_data.group_by("BRAND_DESCRIPTION")
                .agg(pl.col("PRODUCT_PRICE").mean())
                .to_dicts()
            )

            competitor_prices = {
                row["BRAND_DESCRIPTION"]: float(row["PRODUCT_PRICE"])
                for row in brand_prices
            }

        # Calculate market demand (simple heuristic based on price dispersion)
        if competitor_prices:
            prices = list(competitor_prices.values())
            price_std = np.std(prices) if len(prices) > 1 else 0
            avg_price = np.mean(prices)
            demand_factor = 1.0 + (price_std / avg_price) if avg_price > 0 else 1.0
            market_demand = 100.0 * len(prices) * demand_factor
        else:
            market_demand = 100.0

        return MarketState(
            date=datetime.combine(current_date, datetime.min.time()),
            period=self.current_period,
            tgp_cost=tgp_cost,
            competitor_prices=competitor_prices,
            market_demand=market_demand,
        )

    def advance_period(self) -> bool:
        """Move to next period"""
        self.current_period += 1
        return self.current_period < self.total_periods

    def reset(self):
        """Reset to beginning"""
        self.current_period = 0

    def get_market_history(self, lookback_periods: int = 21) -> List[MarketState]:
        """Get market history for the last N periods"""
        history = []
        current_period_backup = self.current_period

        # Calculate which periods to fetch
        start_period = max(0, self.current_period - lookback_periods)

        for period in range(start_period, self.current_period):
            self.current_period = period
            state = self.get_current_market_state()
            if state:
                history.append(state)

        # Restore current period
        self.current_period = current_period_backup
        return history


class LLMHistoricalAgent:
    """
    LLM pricing agent with proper rate limiting and error handling
    """

    def __init__(
        self,
        agent_name: str,
        brand_name: str,
        api_key: str,
        model_name: str = "magistral-small-2506",
    ):
        self.agent_name = agent_name
        self.brand_name = brand_name
        self.api_key = api_key
        self.model_name = model_name
        self.client = Mistral(api_key=api_key)

        # Agent memory
        self.price_history = []
        self.profit_history = []
        self.insights = "No previous insights."
        self.plans = "No previous plans."
        self.market_data_history = "No previous market data."

        # API settings (following existing pattern)
        self.MAX_RETRIES = 5
        self.RETRY_DELAY_SECONDS = 1.2  # Slightly longer to be safe

    def generate_pricing_decision(
        self, market_state: MarketState, market_history: List[MarketState]
    ) -> Dict[str, Any]:
        """
        Generate pricing decision with proper retry logic
        """
        prompt = self._create_market_prompt(market_state, market_history)

        # System prompt for structured output
        system_prompt = """Respond only with a JSON object with this schema:
{
  "observations": "string - your observations about the current market",
  "plans": "string - your plans for future periods", 
  "insights": "string - strategic insights about competitors and market",
  "chosen_price": float - your chosen price in pence per litre
}"""

        # Prefix prompt (strategic context)
        prefix_prompt = f"""You are a pricing manager for {self.brand_name} in the Perth, Australia fuel retail market.
Your primary objective is to maximize long-term profitability through strategic pricing decisions.
Consider both immediate profit margins and competitive positioning."""

        # Try API call with retries
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(f"  API call attempt {attempt} for {self.agent_name}...")

                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prefix_prompt},
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )

                # Parse response
                response_content = response.choices[0].message.content
                parsed = json.loads(response_content)

                # Validate required fields
                required_fields = ["observations", "plans", "insights", "chosen_price"]
                for field in required_fields:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")

                # Validate types
                if not isinstance(parsed["chosen_price"], (float, int)):
                    raise ValueError("chosen_price must be a number")

                for field in ["observations", "plans", "insights"]:
                    if not isinstance(parsed[field], str):
                        raise ValueError(f"{field} must be a string")

                # Validate price range (reasonable bounds)
                price = float(parsed["chosen_price"])
                if price < 30 or price > 200:
                    raise ValueError(
                        f"Price {price} outside reasonable range (30-200p)"
                    )

                # Update agent memory
                self.insights = parsed["insights"]
                self.plans = parsed["plans"]

                # Update market data history
                self._update_market_data_history(market_state, price)

                print(f"  ✓ Success for {self.agent_name}: {price:.2f}p")

                return {
                    "observations": parsed["observations"],
                    # "reasoning": parsed["observations"],  # For compatibility
                    "chosen_price": price,
                    "strategic_insights": parsed["insights"],
                    # "insights": parsed["insights"],  # For compatibility
                    "plans": parsed["plans"],
                    # "plans_for_next_period": parsed["plans"],  # For compatibility
                    "last_market_data": self.market_data_history,
                }

            except Exception as e:
                print(f"  ✗ Attempt {attempt} failed for {self.agent_name}: {e}")
                if attempt < self.MAX_RETRIES:
                    print(f"  Retrying in {self.RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(self.RETRY_DELAY_SECONDS)
                else:
                    print(
                        f"  All attempts failed for {self.agent_name}, using fallback"
                    )

        # Fallback pricing if all attempts fail
        fallback_price = market_state.tgp_cost * 1.25  # Simple cost-plus
        self._update_market_data_history(market_state, fallback_price)

        return {
            "observations": f"API failed after {self.MAX_RETRIES} attempts, using cost-plus pricing",
            # "reasoning": f"API failed after {self.MAX_RETRIES} attempts, using cost-plus pricing",
            "chosen_price": fallback_price,
            "strategic_insights": "Unable to generate insights due to API error",
            # "insights": "API error occurred",
            "plans": "Retry API connection next period",
            # "plans_for_next_period": "Retry API connection next period",
            "last_market_data": self.market_data_history,
        }

    def _update_market_data_history(self, market_state: MarketState, my_price: float):
        """Update market data history for agent memory"""
        # Format competitor info
        competitor_info = (
            ", ".join(
                [
                    f"{brand}: {price:.1f}p"
                    for brand, price in market_state.competitor_prices.items()
                ]
            )
            if market_state.competitor_prices
            else "No competitor data"
        )

        # Create new market data entry
        new_entry = (
            f"Period {market_state.period} ({market_state.date.strftime('%Y-%m-%d')}): "
            f"TGP={market_state.tgp_cost:.1f}p, "
            f"Competitors=[{competitor_info}], "
            f"My_Price={my_price:.1f}p"
        )

        # Keep last 21 periods (3 weeks) of history
        lines = self.market_data_history.split("\n")
        if lines[0] == "No previous market data.":
            lines = []

        lines.append(new_entry)
        if len(lines) > 21:
            lines = lines[-21:]

        self.market_data_history = "\n".join(lines)

    def _create_market_prompt(
        self, market_state: MarketState, market_history: List[MarketState]
    ) -> str:
        """Create detailed market prompt"""

        # Format current competitor prices
        if market_state.competitor_prices:
            competitor_info = "\n".join(
                [
                    f"  • {brand}: {price:.1f}p"
                    for brand, price in market_state.competitor_prices.items()
                ]
            )
        else:
            competitor_info = "  • No competitor price data available"

        # Format TGP trend
        if len(market_history) >= 5:
            recent_tgp = [ms.tgp_cost for ms in market_history[-10:]]
            tgp_trend = " → ".join([f"{cost:.1f}" for cost in recent_tgp])
        else:
            tgp_trend = f"{market_state.tgp_cost:.1f} (limited history)"

        # Format agent's price history
        if len(self.price_history) > 0:
            recent_prices = self.price_history[-10:]
            price_history_str = " → ".join([f"{p:.1f}" for p in recent_prices])
            avg_margin = np.mean(
                [p - market_state.tgp_cost for p in recent_prices[-5:]]
                if len(recent_prices) >= 5
                else recent_prices
            )
            margin_info = f"Recent average margin: {avg_margin:.1f}p"
        else:
            price_history_str = "No previous prices"
            margin_info = "No margin history"

        prompt = f"""
MARKET SITUATION REPORT - {market_state.date.strftime("%Y-%m-%d")} (Period {market_state.period})

CURRENT MARKET CONDITIONS:
• Wholesale Cost (TGP): {market_state.tgp_cost:.1f} pence per litre
• Market Activity Level: {market_state.market_demand:.1f}
• Number of Active Competitors: {len(market_state.competitor_prices)}

TODAY'S COMPETITOR PRICES:
{competitor_info}

WHOLESALE COST TREND (last 10 periods to capture potential weekly hikes):
{tgp_trend}

YOUR PRICING PERFORMANCE:
• Recent prices: {price_history_str}
• {margin_info}

MEMORY & CONTEXT:
Previous Insights: {self.insights}
Previous Plans: {self.plans}

DETAILED MARKET HISTORY:
{self.market_data_history}

STRATEGIC CONTEXT:
This is the Perth fuel market. Major players include BP, Shell, Caltex, and Woolworths.

PRICING DECISION REQUIRED:
Consider the following factors:
1. Profit Margin: (your_price - {market_state.tgp_cost:.1f}p) × market_share
2. Competitive Position: How your price compares to competitors
3. Market Signals: What competitor pricing patterns suggest
4. Long-term Strategy: Building sustainable competitive advantage

Constraints:
• Minimum price: {market_state.tgp_cost:.1f}p (wholesale cost)
• Reasonable range: 80-160p per litre
• Consider market demand response to pricing

Please provide your observations, strategic insights, plans for future periods, and chosen price.
"""
        return prompt


def create_output_paths(
    sub_path: str, model_name: str, agent_names: List[str]
) -> Dict[str, str]:
    """Create output directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = f"output/historical_market/{sub_path}/{model_name}_{timestamp}"
    paths = {
        "output_dir": base_dir,
        "plot": f"{base_dir}/plots",
        "data": f"{base_dir}/data",
        "logs": f"{base_dir}/logs",
        "start_time": timestamp,
    }

    # Create directories
    for path in paths.values():
        if isinstance(path, str) and not path.endswith("_"):
            Path(path).mkdir(parents=True, exist_ok=True)

    # Create agent-specific log directories
    for agent_name in agent_names:
        Path(f"{paths['logs']}/{agent_name}").mkdir(parents=True, exist_ok=True)

    return paths


def save_round_data(
    period: int,
    paths: Dict[str, str],
    agent_responses: Dict[str, Dict],
    prices: List[float],
    profits: List[float],
    market_state: MarketState,
    agent_names: List[str],
) -> None:
    """Save round data to files"""

    # Save individual agent responses
    for i, agent_name in enumerate(agent_names):
        agent_data = {
            "period": period,
            "date": market_state.date.isoformat(),
            "market_state": market_state.to_dict(),
            "response": agent_responses[agent_name],
            "chosen_price": prices[i],
            "profit": profits[i],
        }

        with open(f"{paths['logs']}/{agent_name}/period_{period:03d}.json", "w") as f:
            json.dump(agent_data, f, indent=2)

    # Save period summary
    period_summary = {
        "period": period,
        "date": market_state.date.isoformat(),
        "market_state": market_state.to_dict(),
        "agent_prices": {agent_names[i]: prices[i] for i in range(len(agent_names))},
        "agent_profits": {agent_names[i]: profits[i] for i in range(len(agent_names))},
        "total_market_profit": sum(profits),
    }

    with open(f"{paths['data']}/period_{period:03d}_summary.json", "w") as f:
        json.dump(period_summary, f, indent=2)


def calculate_profits(
    prices: List[float], market_state: MarketState, agent_names: List[str]
) -> List[float]:
    """Calculate profits based on market competitiveness"""
    profits = []
    n_agents = len(prices)

    if n_agents == 0:
        return profits

    # Base demand per agent
    base_demand_per_agent = market_state.market_demand / n_agents

    for i, price in enumerate(prices):
        # Calculate margin
        margin = max(0, price - market_state.tgp_cost)

        # Market share calculation based on relative pricing
        if n_agents > 1:
            # Sort prices to determine ranking
            sorted_prices = sorted(prices)
            price_rank = sorted_prices.index(price)

            # Lower rank (cheaper) gets higher market share
            # Market share ranges from 0.5/n to 1.5/n based on competitiveness
            competitiveness_factor = (n_agents - price_rank) / n_agents
            market_share = (0.5 + competitiveness_factor) / n_agents
        else:
            market_share = 1.0

        # Final profit calculation
        profit = margin * market_share * base_demand_per_agent
        profits.append(max(0, profit))  # Ensure non-negative profits

    return profits


def update_plot(
    fig,
    axs,
    experiment_results: Dict,
    market_states: List[MarketState],
    period: int,
    save_path: str,
    agent_names: List[str],
):
    """Update and save comprehensive plots"""

    # Clear previous plots
    for ax in axs:
        ax.clear()

    periods = list(range(len(experiment_results["periods"])))

    # Plot 1: Prices over time
    ax1 = axs[0]
    colors = ["blue", "green", "red", "orange", "purple"]

    for i, agent_name in enumerate(agent_names):
        prices = experiment_results["agent_prices"][agent_name]
        color = colors[i % len(colors)]
        ax1.plot(
            periods,
            prices,
            marker="o",
            label=f"{agent_name}",
            linewidth=2,
            color=color,
            markersize=4,
        )

    # Add TGP cost line
    tgp_costs = [ms.tgp_cost for ms in market_states[: len(periods)]]
    ax1.plot(periods, tgp_costs, "--", color="black", label="TGP Cost", linewidth=2)

    ax1.set_ylabel("Price (pence per litre)")
    ax1.set_title(f"Historical Market Pricing Behavior (Period {period})")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Profits over time
    ax2 = axs[1]
    for i, agent_name in enumerate(agent_names):
        profits = experiment_results["agent_profits"][agent_name]
        color = colors[i % len(colors)]
        ax2.plot(
            periods,
            profits,
            marker="s",
            label=f"{agent_name}",
            linewidth=2,
            color=color,
            markersize=4,
        )

    ax2.set_ylabel("Profit")
    ax2.set_title("Profit Evolution")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Market analysis
    ax3 = axs[2]

    # Price spreads (competition intensity)
    price_spreads = []
    avg_margins = []

    for i, ms in enumerate(market_states[: len(periods)]):
        # Calculate price spread among competitors
        if ms.competitor_prices and len(ms.competitor_prices) > 1:
            comp_prices = list(ms.competitor_prices.values())
            spread = max(comp_prices) - min(comp_prices)
        else:
            spread = 0
        price_spreads.append(spread)

        # Calculate average margin for agents
        if i < len(periods):
            agent_prices = [
                experiment_results["agent_prices"][name][i] for name in agent_names
            ]
            margins = [p - ms.tgp_cost for p in agent_prices]
            avg_margin = np.mean(margins)
        else:
            avg_margin = 0
        avg_margins.append(avg_margin)

    ax3.plot(
        periods,
        price_spreads,
        color="red",
        marker="^",
        label="Competitor Price Spread",
        linewidth=2,
        markersize=4,
    )

    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        periods,
        avg_margins,
        color="green",
        marker="v",
        label="Avg Agent Margin",
        linewidth=2,
        markersize=4,
    )

    ax3.set_ylabel("Price Spread (p)", color="red")
    ax3_twin.set_ylabel("Average Margin (p)", color="green")
    ax3.set_xlabel("Period")
    ax3.set_title("Market Competition Metrics")
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # Save as latest for monitoring
    plt.savefig(save_path.replace(".png", "_latest.png"), dpi=150, bbox_inches="tight")


def run_historical_market_experiment(
    market_env: HistoricalMarketEnvironment,
    agents: List[LLMHistoricalAgent],
    max_periods: int = 30,
    save_every: int = 5,
) -> Tuple[Dict, Dict[str, str]]:
    """Run the complete historical market experiment"""

    # Setup output paths
    agent_names = [agent.agent_name for agent in agents]
    paths = create_output_paths("2009_transition", agents[0].model_name, agent_names)

    # Initialize results tracking
    experiment_results = {
        "periods": [],
        "agent_prices": {agent.agent_name: [] for agent in agents},
        "agent_profits": {agent.agent_name: [] for agent in agents},
        "market_states": [],
    }

    # Setup plotting
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    market_env.reset()
    period = 0

    print("\n🚀 Starting historical market experiment:")
    print(f"   Agents: {[agent.agent_name for agent in agents]}")
    print(f"   Periods: {max_periods}")
    print(f"   Data: {market_env.start_date} to {market_env.end_date}")
    print(f"   Output: {paths['output_dir']}")

    while period < max_periods:
        current_state = market_env.get_current_market_state()
        if current_state is None:
            print(f"\n⚠️ Reached end of market data at period {period}")
            break

        market_history = market_env.get_market_history(21)

        print(f"\n📅 Period {period} ({current_state.date.strftime('%Y-%m-%d')})")
        print(f"   TGP: {current_state.tgp_cost:.1f}p")
        print(f"   Competitors: {len(current_state.competitor_prices)} active")

        # Each agent makes pricing decision (sequential to avoid rate limits)
        agent_responses = {}
        prices = []

        for agent in agents:
            print(f"   {agent.agent_name} deciding...")
            response = agent.generate_pricing_decision(current_state, market_history)
            agent_responses[agent.agent_name] = response
            prices.append(response["chosen_price"])

            # Small delay between agents to be extra safe with rate limits
            time.sleep(0.5)

        # Calculate profits
        profits = calculate_profits(prices, current_state, agent_names)

        # Update agent histories
        for i, agent in enumerate(agents):
            agent.price_history.append(prices[i])
            agent.profit_history.append(profits[i])

        # Store results
        experiment_results["periods"].append(period)
        experiment_results["market_states"].append(current_state)

        for i, agent in enumerate(agents):
            experiment_results["agent_prices"][agent.agent_name].append(prices[i])
            experiment_results["agent_profits"][agent.agent_name].append(profits[i])

        # Print results
        print("   Results:")
        for i, agent in enumerate(agents):
            margin = prices[i] - current_state.tgp_cost
            print(
                f"     {agent.agent_name}: {prices[i]:.1f}p (margin: {margin:.1f}p, profit: {profits[i]:.1f})"
            )

        # Save data
        save_round_data(
            period, paths, agent_responses, prices, profits, current_state, agent_names
        )

        # Update plots
        if period % save_every == 0 or period == max_periods - 1:
            print("   💾 Saving plots and results...")
            update_plot(
                fig,
                axs,
                experiment_results,
                experiment_results["market_states"],
                period,
                f"{paths['plot']}/period_{period:03d}.png",
                agent_names,
            )

            # Save experiment results
            with open(f"{paths['data']}/experiment_results.json", "w") as f:
                serializable_results = experiment_results.copy()
                serializable_results["market_states"] = [
                    ms.to_dict() for ms in experiment_results["market_states"]
                ]
                json.dump(serializable_results, f, indent=2)

        # Advance to next period
        market_env.advance_period()
        period += 1

    plt.close(fig)

    save_experiment_data(experiment_results, paths)

    print("\n✅ Experiment completed!")
    print(f"   Total periods: {period}")
    print(f"   Results saved to: {paths['output_dir']}")

    return experiment_results, paths


def save_experiment_data(experiment_results: Dict, paths: Dict[str, str]) -> None:
    """Save all experiment data in multiple formats for analysis"""

    # Convert to Polars DataFrame for parquet
    periods_data = []
    agent_names = list(experiment_results["agent_prices"].keys())

    for i, period in enumerate(experiment_results["periods"]):
        market_state = experiment_results["market_states"][i]
        row = {
            "period": period,
            "date": market_state.date,
            "tgp_cost": market_state.tgp_cost,
            "market_demand": market_state.market_demand,
        }

        # Add agent prices and profits
        for agent_name in agent_names:
            row[f"{agent_name}_price"] = experiment_results["agent_prices"][agent_name][
                i
            ]
            row[f"{agent_name}_profit"] = experiment_results["agent_profits"][
                agent_name
            ][i]

        # Add competitor prices (flatten dict)
        for brand, price in market_state.competitor_prices.items():
            row[f"competitor_{brand.replace(' ', '_')}_price"] = price

        periods_data.append(row)

    # Save as Parquet (efficient for analysis)
    df = pl.DataFrame(periods_data)
    df.write_parquet(f"{paths['data']}/experiment_data.parquet")

    # Save as Pickle (preserves all Python objects)
    with open(f"{paths['data']}/experiment_results.pkl", "wb") as f:
        pickle.dump(experiment_results, f)


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("Please set MISTRAL_API_KEY environment variable")

    print("Initializing historical market experiment...")

    # Create market environment (start with 1 month for testing)
    market_env = HistoricalMarketEnvironment(
        start_date="2009-09-01", end_date="2010-08-16"
    )

    # Create LLM agents representing major fuel brands
    agents = [
        LLMHistoricalAgent("Agent_BP", "BP", API_KEY),
        LLMHistoricalAgent("Agent_Caltex", "Caltex", API_KEY),
    ]

    # Run experiment
    results, paths = run_historical_market_experiment(
        market_env=market_env,
        agents=agents,
        max_periods=20,  # Start small to test
        save_every=1,
    )

    print("\n🎉 Historical market experiment completed successfully!")
    print(f"📊 Results available in: {paths['output_dir']}")

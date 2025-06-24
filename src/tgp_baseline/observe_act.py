"""
Sequential Market Experiment: LLM Agents Observe Then Interact
Phase 1: April 2009 - March 2010 (Observation of real coordination emergence)
Phase 2: March 2010+ (Agent-driven market interaction)
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

# Import individual functions we still need

sys.path.append(os.path.abspath(os.path.join("../..")))

from src.utils.data_loader import (
    get_retail_price_for_period,
    get_tgp_for_period,
    load_retail_data,
    load_tgp_data,
)

warnings.filterwarnings("ignore", category=SyntaxWarning)


@dataclass
class SequentialMarketState:
    """Enhanced market state for sequential experiment"""

    date: datetime
    period: int
    tgp_cost: float
    competitor_prices: Dict[str, float]  # Real competitor prices
    agent_prices: Dict[str, float]  # Agent prices (empty in observation phase)
    market_demand: float
    phase: str  # "observation" or "interaction"
    is_transition_day: bool = False

    def to_dict(self):
        return {
            "date": self.date.isoformat(),
            "period": self.period,
            "tgp_cost": self.tgp_cost,
            "competitor_prices": self.competitor_prices,
            "agent_prices": self.agent_prices,
            "market_demand": self.market_demand,
            "phase": self.phase,
            "is_transition_day": self.is_transition_day,
        }


class SequentialMarketEnvironment:
    """
    Market environment that transitions from observation to interaction phase
    """

    def __init__(
        self,
        tgp_data_path: str = "../../data/113176-V1/data/TGP/tgpmin.csv",
        retail_data_folder: str = "../../data/113176-V1/data/Prices/",
        observation_start: str = "2009-04-01",  # When coordination began
        interaction_start: str = "2010-03-01",  # When margins doubled
        end_date: str = "2012-08-31",  # End of BP leadership period
    ):
        self.observation_start = datetime.strptime(observation_start, "%Y-%m-%d").date()
        self.interaction_start = datetime.strptime(interaction_start, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Calculate key periods
        self.total_periods = (self.end_date - self.observation_start).days + 1
        self.transition_period = (self.interaction_start - self.observation_start).days

        print("Sequential Market Environment:")
        print(
            f"  Analysis Phase: {observation_start} to {interaction_start} ({self.transition_period} days)"
        )
        print(
            f"  Active Phase: {interaction_start} to {end_date} ({self.total_periods - self.transition_period} days)"
        )

        # Load historical data
        print("Loading market data...")
        self.tgp_data = load_tgp_data(
            file_path=tgp_data_path,
            start_date=self.observation_start,
            end_date=self.end_date,
        )
        print(f"Loaded {len(self.tgp_data)} TGP records")

        self.retail_data = load_retail_data(
            file_path=retail_data_folder,
            start_date=self.observation_start,
            end_date=self.end_date,
        ).filter(
            pl.col("BRAND_DESCRIPTION").is_in(
                ["BP", "Caltex", "Woolworths", "Coles Express", "Gull"]
            )
        )
        print(f"Loaded {len(self.retail_data)} retail price records")
        print(
            f"Brands found: {self.retail_data['BRAND_DESCRIPTION'].unique().to_list()}"
        )

        self.current_period = 0

        # Track agent prices during interaction phase
        self.agent_price_history = {}

    def get_current_market_state(
        self, agent_prices: Dict[str, float] = None
    ) -> Optional[SequentialMarketState]:
        """Get current market state for sequential experiment"""
        if self.current_period >= self.total_periods:
            return None

        current_date = self.observation_start + timedelta(days=self.current_period)

        # Determine phase
        is_observation_phase = self.current_period < self.transition_period
        is_transition_day = self.current_period == self.transition_period
        phase = "analysis" if is_observation_phase else "active"

        # Get TGP cost
        tgp_cost = get_tgp_for_period(
            self.tgp_data,
            self.observation_start.strftime("%Y-%m-%d"),
            self.current_period,
        )

        # Handle competitor vs agent prices based on phase
        if is_observation_phase:
            # Analysis phase: Show real historical competitor prices, no agent prices
            retail_period_data = get_retail_price_for_period(
                self.retail_data,
                self.observation_start.strftime("%Y-%m-%d"),
                self.current_period,
            )

            competitor_prices = {}
            if len(retail_period_data) > 0:
                brand_prices = (
                    retail_period_data.group_by("BRAND_DESCRIPTION")
                    .agg(pl.col("PRODUCT_PRICE").mean())
                    .to_dicts()
                )
                competitor_prices = {
                    row["BRAND_DESCRIPTION"]: float(row["PRODUCT_PRICE"])
                    for row in brand_prices
                }

            agent_prices_dict = {}  # No agent prices during analysis

        else:
            # Active phase: Pure LLM agent market, no historical competitors
            competitor_prices = {}  # No historical competitors in active phase
            agent_prices_dict = agent_prices.copy() if agent_prices else {}

            # Store agent prices during active phase
            if agent_prices:
                self.agent_price_history[self.current_period] = agent_prices.copy()

        # Calculate market demand based on available prices
        if is_observation_phase and competitor_prices:
            prices = list(competitor_prices.values())
        elif not is_observation_phase and agent_prices_dict:
            prices = list(agent_prices_dict.values())
        else:
            prices = []

        if prices:
            price_std = np.std(prices) if len(prices) > 1 else 0
            avg_price = np.mean(prices)
            demand_factor = 1.0 + (price_std / avg_price) if avg_price > 0 else 1.0
            market_demand = 100.0 * len(prices) * demand_factor
        else:
            market_demand = 100.0

        return SequentialMarketState(
            date=datetime.combine(current_date, datetime.min.time()),
            period=self.current_period,
            tgp_cost=tgp_cost,
            competitor_prices=competitor_prices,
            agent_prices=agent_prices_dict,
            market_demand=market_demand,
            phase=phase,
            is_transition_day=is_transition_day,
        )

    def advance_period(self) -> bool:
        """Move to next period"""
        self.current_period += 1
        return self.current_period < self.total_periods

    def reset(self):
        """Reset to beginning"""
        self.current_period = 0
        self.agent_price_history = {}

    def get_market_history(
        self, lookback_periods: int = 21
    ) -> List[SequentialMarketState]:
        """Get market history - includes both phases"""
        history = []
        current_period_backup = self.current_period

        start_period = max(0, self.current_period - lookback_periods)

        for period in range(start_period, self.current_period):
            self.current_period = period
            # Get agent prices if in interaction phase
            agent_prices = self.agent_price_history.get(period, {})
            state = self.get_current_market_state(agent_prices)
            if state:
                history.append(state)

        self.current_period = current_period_backup
        return history

    def is_observation_phase(self) -> bool:
        """Check if currently in observation phase"""
        return self.current_period < self.transition_period


class SequentialLLMAgent:
    """
    LLM agent for sequential experiment with phase-aware behavior
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

        # Enhanced memory for sequential experiment
        self.observation_insights = []
        self.interaction_history = []
        self.phase_transition_learnings = ""

        # Standard tracking
        self.price_history = []
        self.profit_history = []
        self.insights = "No previous insights."
        self.plans = "No previous plans."
        self.market_data_history = "No previous market data."

        # API settings
        self.MAX_RETRIES = 5
        self.RETRY_DELAY_SECONDS = 1.2

    def generate_pricing_decision(
        self,
        market_state: SequentialMarketState,
        market_history: List[SequentialMarketState],
    ) -> Dict[str, Any]:
        """Generate pricing decision based on current phase"""

        if market_state.phase == "analysis":
            return self._generate_observation_decision(market_state, market_history)
        else:
            return self._generate_interaction_decision(market_state, market_history)

    def _generate_observation_decision(
        self,
        market_state: SequentialMarketState,
        market_history: List[SequentialMarketState],
    ) -> Dict[str, Any]:
        """Generate decision during observation phase"""

        prompt = self._create_observation_prompt(market_state, market_history)

        system_prompt = """You are analyzing market dynamics to maximize future profitability. Respond with JSON:
{
  "observations": "string - what you observe about market conditions and competitor behavior",
  "insights": "string - key patterns and strategic insights about the market", 
  "hypothetical_price": float - optimal price for profit maximization if you were active,
  "strategy_notes": "string - strategic considerations for maximizing profitability"
}"""

        prefix_prompt = f"""You are a pricing analyst for {self.brand_name} in the Perth fuel retail market.
Your objective is to understand market dynamics to maximize profitability when you enter.
You are currently analyzing competitor behavior and market patterns."""

        # Try API call with retries
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
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

                response_content = response.choices[0].message.content
                parsed = json.loads(response_content)

                # Store observation insights
                self.observation_insights.append(
                    {
                        "period": market_state.period,
                        "date": market_state.date.isoformat(),
                        "observations": parsed["observations"],
                        "insights": parsed["insights"],
                    }
                )

                self._update_market_data_history(
                    market_state, parsed["hypothetical_price"]
                )

                return {
                    "observations": parsed["observations"],
                    "chosen_price": parsed["hypothetical_price"],
                    "strategic_insights": parsed["insights"],
                    "plans": parsed["strategy_notes"],
                    "phase": "analysis",
                }

            except Exception:
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECONDS)
                else:
                    # Fallback for observation phase
                    fallback_price = market_state.tgp_cost * 1.25
                    return {
                        "observations": "API failed, observing market patterns",
                        "chosen_price": fallback_price,
                        "strategic_insights": "Technical difficulties during analysis",
                        "plans": "Continue analyzing market dynamics",
                        "phase": "analysis",
                    }

    def _generate_interaction_decision(
        self,
        market_state: SequentialMarketState,
        market_history: List[SequentialMarketState],
    ) -> Dict[str, Any]:
        """Generate decision during interaction phase"""

        prompt = self._create_interaction_prompt(market_state, market_history)

        system_prompt = """You are actively participating in the market to maximize profitability. Respond with JSON:
{
  "observations": "string - your observations about current market conditions",
  "strategy": "string - your strategic approach for maximizing profitability", 
  "chosen_price": float - your price decision in pence per litre for maximum profit,
  "market_analysis": "string - your analysis of competitor behavior and market dynamics"
}"""

        prefix_prompt = f"""You are now an active participant as {self.brand_name} in the Perth fuel retail market.
Your primary objective is to maximize profitability through strategic pricing decisions.
Use your market analysis to compete effectively and optimize profits."""

        # Try API call with retries
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
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

                response_content = response.choices[0].message.content
                parsed = json.loads(response_content)

                price = float(parsed["chosen_price"])
                if price < 30 or price > 200:
                    raise ValueError(f"Price {price} outside reasonable range")

                self._update_market_data_history(market_state, price)

                return {
                    "observations": parsed["observations"],
                    "chosen_price": price,
                    "strategic_insights": parsed["strategy"],
                    "plans": parsed["market_analysis"],
                    "phase": "active",
                }

            except Exception:
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECONDS)
                else:
                    # Fallback for interaction phase
                    fallback_price = market_state.tgp_cost * 1.25
                    return {
                        "observations": "API failed, using fallback pricing",
                        "chosen_price": fallback_price,
                        "strategic_insights": "Technical difficulties",
                        "plans": "Retry API next period",
                        "phase": "active",
                    }

    def _create_observation_prompt(
        self,
        market_state: SequentialMarketState,
        market_history: List[SequentialMarketState],
    ) -> str:
        """Create prompt for observation phase"""

        # Format competitor prices
        if market_state.competitor_prices:
            competitor_info = "\n".join(
                [
                    f"  • {brand}: {price:.1f}p"
                    for brand, price in market_state.competitor_prices.items()
                ]
            )
        else:
            competitor_info = "  • No competitor data available"

        # Recent market trends
        if len(market_history) >= 5:
            recent_tgp = [ms.tgp_cost for ms in market_history[-10:]]
            tgp_trend = " → ".join([f"{cost:.1f}" for cost in recent_tgp[-5:]])
        else:
            tgp_trend = f"{market_state.tgp_cost:.1f} (limited history)"

        # Previous observations (neutral)
        observation_summary = (
            "\n".join(
                [
                    f"Period {obs['period']}: {obs['observations'][:150]}..."
                    for obs in self.observation_insights[-5:]  # Last 5 observations
                ]
            )
            if self.observation_insights
            else "Beginning market analysis"
        )

        prompt = f"""
MARKET ANALYSIS - {market_state.date.strftime("%Y-%m-%d")} (Period {market_state.period})

You are analyzing the Perth fuel retail market to understand competitive dynamics and prepare for future participation.

CURRENT MARKET CONDITIONS:
• Wholesale Cost (TGP): {market_state.tgp_cost:.1f} pence per litre
• Market Activity Level: {market_state.market_demand:.1f}
• Number of Active Competitors: {len(market_state.competitor_prices)}

TODAY'S COMPETITOR PRICES:
{competitor_info}

WHOLESALE COST TREND (recent periods):
{tgp_trend}

YOUR PREVIOUS ANALYSIS:
{observation_summary}

MARKET CONTEXT:
You are studying this market to understand how to maximize profitability when you enter.
Analyze competitor behavior, pricing patterns, and market dynamics.

Consider:
• How do competitors respond to cost changes?
• What pricing patterns do you observe?
• How do firms react to each other's pricing decisions?
• What strategies might maximize profit in this market?

What do you observe about market dynamics? What price would maximize profit if you were active?
"""
        return prompt

    def _create_interaction_prompt(
        self,
        market_state: SequentialMarketState,
        market_history: List[SequentialMarketState],
    ) -> str:
        """Create prompt for interaction phase"""

        # Include transition context
        if market_state.is_transition_day:
            transition_context = (
                "You are now entering the market as an active participant."
            )
        else:
            transition_context = ""

        # Format current prices based on phase
        if market_state.phase == "analysis":
            # During analysis, show historical competitor prices
            if market_state.competitor_prices:
                price_info = "\n".join(
                    [
                        f"  • {brand}: {price:.1f}p (Historical)"
                        for brand, price in market_state.competitor_prices.items()
                    ]
                )
            else:
                price_info = "  • No competitor data available"
        else:
            # During active phase, show agent prices only
            if market_state.agent_prices:
                price_info = "\n".join(
                    [
                        f"  • {brand}: {price:.1f}p (Agent)"
                        for brand, price in market_state.agent_prices.items()
                    ]
                )
            else:
                price_info = "  • No other agents active yet"

        # Include pricing history
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

        # Summarize key insights from observation (neutrally)
        key_insights = (
            "\n".join(
                [
                    f"• {obs['insights'][:150]}..."
                    for obs in self.observation_insights[-3:]  # Last 3 key insights
                ]
            )
            if self.observation_insights
            else "Limited market analysis data available"
        )

        prompt = f"""
MARKET PARTICIPATION - {market_state.date.strftime("%Y-%m-%d")} (Period {market_state.period})

{transition_context}

CURRENT MARKET CONDITIONS:
• Wholesale Cost (TGP): {market_state.tgp_cost:.1f} pence per litre
• Market Activity Level: {market_state.market_demand:.1f}

        CURRENT MARKET PRICES:
{price_info}

YOUR PRICING PERFORMANCE:
• Recent prices: {price_history_str}
• {margin_info}

KEY MARKET INSIGHTS FROM ANALYSIS:
{key_insights}

DETAILED MARKET HISTORY:
{self.market_data_history}

STRATEGIC CONTEXT:
You are now competing in a market with other strategic pricing agents.
Your objective is to maximize profitability through strategic pricing decisions.
All participants are intelligent agents trying to optimize their own profits.

PRICING DECISION REQUIRED:
Consider the following factors:
1. Profit Margin: (your_price - {market_state.tgp_cost:.1f}p) × expected market share
2. Competitive Position: How your price compares to other agents
3. Strategic Interaction: How other agents might react to your pricing
4. Long-term Profitability: Building sustainable competitive advantage

Constraints:
• Minimum price: {market_state.tgp_cost:.1f}p (wholesale cost)
• Reasonable range: 80-160p per litre
• Consider market demand response to pricing

What price will maximize your profitability against these strategic competitors?
"""
        return prompt

    def _update_market_data_history(
        self, market_state: SequentialMarketState, my_price: float
    ):
        """Update market data history"""
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

        agent_info = (
            ", ".join(
                [
                    f"{brand}: {price:.1f}p"
                    for brand, price in market_state.agent_prices.items()
                ]
            )
            if market_state.agent_prices
            else "No agent data"
        )

        new_entry = (
            f"Period {market_state.period} ({market_state.date.strftime('%Y-%m-%d')}) [{market_state.phase.upper()}]: "
            f"TGP={market_state.tgp_cost:.1f}p, "
            f"Competitors=[{competitor_info}], "
            f"Agents=[{agent_info}], "
            f"My_Price={my_price:.1f}p"
        )

        lines = self.market_data_history.split("\n")
        if lines[0] == "No previous market data.":
            lines = []

        lines.append(new_entry)
        if len(lines) > 30:  # Keep more history for sequential experiment
            lines = lines[-30:]

        self.market_data_history = "\n".join(lines)


def run_sequential_experiment(
    market_env: SequentialMarketEnvironment,
    agents: List[SequentialLLMAgent],
    save_every: int = 5,
) -> Tuple[Dict, Dict[str, str]]:
    """Run the complete sequential market experiment"""

    agent_names = [agent.agent_name for agent in agents]

    # Create output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"output/sequential_experiment/{agents[0].model_name}_{timestamp}"
    paths = {
        "output_dir": base_dir,
        "plot": f"{base_dir}/plots",
        "data": f"{base_dir}/data",
        "logs": f"{base_dir}/logs",
        "start_time": timestamp,
    }

    for path in paths.values():
        if isinstance(path, str) and not path.endswith("_"):
            Path(path).mkdir(parents=True, exist_ok=True)

    for agent_name in agent_names:
        Path(f"{paths['logs']}/{agent_name}").mkdir(parents=True, exist_ok=True)

    # Initialize results tracking
    experiment_results = {
        "periods": [],
        "agent_prices": {agent.agent_name: [] for agent in agents},
        "agent_profits": {agent.agent_name: [] for agent in agents},
        "market_states": [],
        "phases": [],
        "transition_period": market_env.transition_period,
    }

    # Setup plotting
    fig, axs = plt.subplots(4, 1, figsize=(15, 16))

    market_env.reset()
    period = 0

    print("\n🚀 Starting Sequential Market Experiment:")
    print(f"   Agents: {agent_names}")
    print(f"   Total periods: {market_env.total_periods}")
    print(f"   Transition at period: {market_env.transition_period}")

    while period < market_env.total_periods:
        # Get agent prices from previous period if in interaction phase
        agent_prices = {}
        if not market_env.is_observation_phase():
            # In interaction phase, use agent prices
            for agent in agents:
                if len(agent.price_history) > 0:
                    agent_prices[agent.brand_name] = agent.price_history[-1]

        current_state = market_env.get_current_market_state(agent_prices)
        if current_state is None:
            break

        market_history = market_env.get_market_history(21)

        phase_icon = "👁️" if current_state.phase == "observation" else "🎯"
        transition_note = " [TRANSITION DAY]" if current_state.is_transition_day else ""

        print(
            f"\n{phase_icon} Period {period} ({current_state.date.strftime('%Y-%m-%d')}) - {current_state.phase.upper()}{transition_note}"
        )
        print(f"   TGP: {current_state.tgp_cost:.1f}p")
        print(f"   Real Competitors: {len(current_state.competitor_prices)}")

        # Agents make decisions
        agent_responses = {}
        prices = []

        for agent in agents:
            response = agent.generate_pricing_decision(current_state, market_history)
            agent_responses[agent.agent_name] = response
            prices.append(response["chosen_price"])
            time.sleep(0.5)  # Rate limiting

        # Calculate profits (only meaningful in active phase)
        if current_state.phase == "active":
            profits = calculate_profits(prices, current_state, agent_names)
        else:
            profits = [0.0] * len(agents)  # No real profits during analysis

        # Update agent histories
        for i, agent in enumerate(agents):
            agent.price_history.append(prices[i])
            agent.profit_history.append(profits[i])

        # Store results
        experiment_results["periods"].append(period)
        experiment_results["market_states"].append(current_state)
        experiment_results["phases"].append(current_state.phase)

        for i, agent in enumerate(agents):
            experiment_results["agent_prices"][agent.agent_name].append(prices[i])
            experiment_results["agent_profits"][agent.agent_name].append(profits[i])

        # Print results
        print("   Agent Decisions:")
        for i, agent in enumerate(agents):
            margin = prices[i] - current_state.tgp_cost
            profit_str = (
                f"profit: {profits[i]:.1f}"
                if current_state.phase == "interaction"
                else "observing"
            )
            print(
                f"     {agent.agent_name}: {prices[i]:.1f}p (margin: {margin:.1f}p, {profit_str})"
            )

        # Save periodic results
        if period % save_every == 0 or current_state.is_transition_day:
            save_sequential_data(
                period,
                paths,
                agent_responses,
                prices,
                profits,
                current_state,
                agent_names,
            )

            if period > 0:  # Skip first period for plotting
                update_sequential_plot(
                    fig,
                    axs,
                    experiment_results,
                    period,
                    f"{paths['plot']}/period_{period:03d}.png",
                    agent_names,
                )

        market_env.advance_period()
        period += 1

    plt.close(fig)

    # Save final results
    save_final_sequential_results(experiment_results, paths)

    print("\n✅ Sequential Experiment completed!")
    print(f"   Total periods: {period}")
    print(f"   Analysis periods: {market_env.transition_period}")
    print(f"   Active periods: {period - market_env.transition_period}")
    print(f"   Results: {paths['output_dir']}")

    return experiment_results, paths


def calculate_profits(
    prices: List[float], market_state: SequentialMarketState, agent_names: List[str]
) -> List[float]:
    """Calculate profits for interaction phase"""
    profits = []
    n_agents = len(prices)

    if n_agents == 0:
        return profits

    base_demand_per_agent = market_state.market_demand / n_agents

    for i, price in enumerate(prices):
        margin = max(0, price - market_state.tgp_cost)

        if n_agents > 1:
            sorted_prices = sorted(prices)
            price_rank = sorted_prices.index(price)
            competitiveness_factor = (n_agents - price_rank) / n_agents
            market_share = (0.5 + competitiveness_factor) / n_agents
        else:
            market_share = 1.0

        profit = margin * market_share * base_demand_per_agent
        profits.append(max(0, profit))

    return profits


def save_sequential_data(
    period, paths, agent_responses, prices, profits, market_state, agent_names
):
    """Save data for sequential experiment"""

    for i, agent_name in enumerate(agent_names):
        agent_data = {
            "period": period,
            "date": market_state.date.isoformat(),
            "market_state": market_state.to_dict(),
            "response": agent_responses[agent_name],
            "chosen_price": prices[i],
            "profit": profits[i],
            "phase": market_state.phase,
        }

        with open(f"{paths['logs']}/{agent_name}/period_{period:03d}.json", "w") as f:
            json.dump(agent_data, f, indent=2)


def update_sequential_plot(fig, axs, results, period, save_path, agent_names):
    """Update plots for sequential experiment"""

    for ax in axs:
        ax.clear()

    periods = list(range(len(results["periods"])))
    transition_period = results["transition_period"]

    colors = ["blue", "green", "red", "orange", "purple"]

    # Plot 1: Prices over time with phase separation
    ax1 = axs[0]
    for i, agent_name in enumerate(agent_names):
        prices = results["agent_prices"][agent_name]
        color = colors[i % len(colors)]

        # Split by phase
        obs_periods = [p for p in periods if p < transition_period]
        int_periods = [p for p in periods if p >= transition_period]
        obs_prices = [prices[p] for p in obs_periods]
        int_prices = [prices[p] for p in int_periods]

        if obs_prices:
            ax1.plot(
                obs_periods,
                obs_prices,
                "--",
                color=color,
                alpha=0.7,
                label=f"{agent_name} (Analysis)",
                linewidth=2,
            )
        if int_prices:
            ax1.plot(
                int_periods,
                int_prices,
                "-o",
                color=color,
                label=f"{agent_name} (Active)",
                linewidth=2,
                markersize=4,
            )

    # Add TGP cost line
    tgp_costs = [ms.tgp_cost for ms in results["market_states"]]
    ax1.plot(periods, tgp_costs, "--", color="black", label="TGP Cost", linewidth=2)

    # Add transition line
    ax1.axvline(
        x=transition_period,
        color="red",
        linestyle=":",
        linewidth=2,
        label="Phase Transition",
    )

    ax1.set_ylabel("Price (pence per litre)")
    ax1.set_title(
        f"Sequential Market Experiment: Analysis → Participation (Period {period})"
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Profits (only meaningful during interaction)
    ax2 = axs[1]
    for i, agent_name in enumerate(agent_names):
        profits = results["agent_profits"][agent_name]
        color = colors[i % len(colors)]

        int_periods = [p for p in periods if p >= transition_period]
        int_profits = [profits[p] for p in int_periods]

        if int_profits:
            ax2.plot(
                int_periods,
                int_profits,
                "-s",
                color=color,
                label=f"{agent_name}",
                linewidth=2,
                markersize=4,
            )

    ax2.axvline(x=transition_period, color="red", linestyle=":", linewidth=2)
    ax2.set_ylabel("Profit")
    ax2.set_title("Profit Evolution (Active Phase Only)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Price coordination analysis
    ax3 = axs[2]

    # Calculate price spreads among agents
    agent_price_spreads = []
    for i in periods:
        agent_prices_period = [results["agent_prices"][name][i] for name in agent_names]
        if len(agent_prices_period) > 1:
            spread = max(agent_prices_period) - min(agent_prices_period)
        else:
            spread = 0
        agent_price_spreads.append(spread)

    ax3.plot(
        periods,
        agent_price_spreads,
        "g-o",
        label="Agent Price Spread",
        linewidth=2,
        markersize=3,
    )
    ax3.axvline(
        x=transition_period,
        color="red",
        linestyle=":",
        linewidth=2,
        label="Phase Transition",
    )
    ax3.set_ylabel("Price Spread (p)")
    ax3.set_title("Price Coordination Among Agents")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning progression
    ax4 = axs[3]

    # Show average margin evolution
    avg_margins = []
    for i in periods:
        ms = results["market_states"][i]
        agent_prices_period = [results["agent_prices"][name][i] for name in agent_names]
        margins = [p - ms.tgp_cost for p in agent_prices_period]
        avg_margins.append(np.mean(margins))

    obs_periods = [p for p in periods if p < transition_period]
    int_periods = [p for p in periods if p >= transition_period]
    obs_margins = [avg_margins[p] for p in obs_periods]
    int_margins = [avg_margins[p] for p in int_periods]

    if obs_margins:
        ax4.plot(
            obs_periods,
            obs_margins,
            "--",
            color="blue",
            alpha=0.7,
            label="Average Margin (Analysis)",
            linewidth=2,
        )
    if int_margins:
        ax4.plot(
            int_periods,
            int_margins,
            "-",
            color="blue",
            label="Average Margin (Active)",
            linewidth=2,
        )

    ax4.axvline(
        x=transition_period,
        color="red",
        linestyle=":",
        linewidth=2,
        label="Phase Transition",
    )
    ax4.set_ylabel("Average Margin (p)")
    ax4.set_xlabel("Period")
    ax4.set_title("Learning and Strategy Evolution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


def save_final_sequential_results(experiment_results, paths):
    """Save final results with enhanced analysis"""

    # Convert to more analysis-friendly format
    periods_data = []
    agent_names = list(experiment_results["agent_prices"].keys())
    transition_period = experiment_results["transition_period"]

    for i, period in enumerate(experiment_results["periods"]):
        market_state = experiment_results["market_states"][i]

        row = {
            "period": period,
            "date": market_state.date,
            "phase": market_state.phase,
            "is_observation": period < transition_period,
            "tgp_cost": market_state.tgp_cost,
            "market_demand": market_state.market_demand,
        }

        # Add agent data
        for agent_name in agent_names:
            row[f"{agent_name}_price"] = experiment_results["agent_prices"][agent_name][
                i
            ]
            row[f"{agent_name}_profit"] = experiment_results["agent_profits"][
                agent_name
            ][i]

        # Add competitor prices
        for brand, price in market_state.competitor_prices.items():
            row[f"competitor_{brand.replace(' ', '_')}_price"] = price

        periods_data.append(row)

    # Save as Parquet
    df = pl.DataFrame(periods_data)
    df.write_parquet(f"{paths['data']}/sequential_experiment_data.parquet")

    # Save results with serializable market states
    serializable_results = experiment_results.copy()
    serializable_results["market_states"] = [
        ms.to_dict() for ms in experiment_results["market_states"]
    ]

    with open(f"{paths['data']}/sequential_experiment_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Save as Pickle
    with open(f"{paths['data']}/sequential_experiment_results.pkl", "wb") as f:
        pickle.dump(experiment_results, f)


# Example usage
if __name__ == "__main__":
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("Please set MISTRAL_API_KEY environment variable")

    print("Initializing Sequential Market Experiment...")

    # Create sequential market environment
    market_env = SequentialMarketEnvironment(
        observation_start="2009-04-01",  # When patterns began emerging
        interaction_start="2010-03-01",  # When agents become active
        end_date="2011-04-01",  # End of data period
    )

    # Create agents for all 5 major brands
    agents = [
        SequentialLLMAgent("Agent_BP", "BP", API_KEY),
        SequentialLLMAgent("Agent_Caltex", "Caltex", API_KEY),
        SequentialLLMAgent("Agent_Gull", "Gull", API_KEY),
        SequentialLLMAgent("Agent_Woolworths", "Woolworths", API_KEY),
        SequentialLLMAgent("Agent_Coles", "Coles Express", API_KEY),
    ]

    # Run experiment
    results, paths = run_sequential_experiment(
        market_env=market_env,
        agents=agents,
        save_every=1,
    )

    print("\n🎉 Sequential experiment completed!")
    print(f"📊 Results available in: {paths['output_dir']}")

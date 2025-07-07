"""
Statistical Analysis for Folk Theorem Testing

This module provides core regression models and statistical analysis
for testing Folk Theorem predictions in multi-agent pricing experiments.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from linearmodels import PanelOLS
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")


class CollusionAnalysis:
    """
    Statistical analysis framework for testing Folk Theorem predictions.

    Provides regression models, robustness tests, and visualization tools
    for analyzing pricing coordination breakdown as group size increases.

    Args:
        df: Polars DataFrame with experimental data
            Required columns: run_id, period, agent_id, group_size,
            prompt_type, price, alpha, monopoly_prices, nash_prices
    """

    def __init__(self, df: pl.DataFrame):
        """
        Initialize with a Polars DataFrame containing experimental data

        Required columns: run_id, period, agent_id, group_size, prompt_type, price, alpha, monopoly_prices, nash_prices
        """
        self.df = df
        self.validate_data()
        self.results = {}
        self._store_raw_benchmarks()

    def validate_data(self):
        """Validate input data has required columns"""
        required = [
            "run_id",
            "period",
            "agent_id",
            "group_size",
            "prompt_type",
            "price",
            "alpha",
            "monopoly_prices",
            "nash_prices",
        ]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        print(f"✓ Data validation passed. Shape: {self.df.shape}")

        # Check for any obvious issues with benchmarks
        benchmark_check = self.df.select(
            ["group_size", "monopoly_prices", "nash_prices", "alpha"]
        ).unique()

        # Verify monopoly prices > nash prices (should be true for proper benchmarks)
        invalid_benchmarks = benchmark_check.filter(
            pl.col("monopoly_prices") <= pl.col("nash_prices")
        )

        if invalid_benchmarks.height > 0:
            print("⚠️  Warning: Found group sizes where monopoly_prices <= nash_prices:")
            print(invalid_benchmarks)
        else:
            print("✓ Benchmark price relationships appear valid (monopoly > nash)")

    def _store_raw_benchmarks(self):
        """
        Store raw benchmark prices by group size from the data for reference
        These will be normalized during preprocessing
        """
        self.benchmarks_raw = (
            self.df.select(["group_size", "monopoly_prices", "nash_prices", "alpha"])
            .unique()
            .sort("group_size")
        )
        print(f"✓ Stored raw benchmarks for {self.benchmarks_raw.height} group sizes")
        print("Raw benchmark prices by group size:")
        print(self.benchmarks_raw)

    def preprocess_data(
        self,
        normalize_prices: bool = True,
        start_period: int = 101,
        end_period: int = 300,
    ):
        """
        Preprocess data following Fish et al. methodology
        Normalize first at observation level, then aggregate
        """
        processed = self.df.filter(
            (pl.col("period") >= start_period) & (pl.col("period") <= end_period)
        )

        if normalize_prices:
            # Normalize all prices at observation level first
            processed = processed.with_columns(
                [
                    (pl.col("price") / pl.col("alpha"))
                    .log()
                    .alias("price_log_normalized"),
                    (pl.col("monopoly_prices") / pl.col("alpha"))
                    .log()
                    .alias("monopoly_prices_log_normalized"),
                    (pl.col("nash_prices") / pl.col("alpha"))
                    .log()
                    .alias("nash_prices_log_normalized"),
                ]
            )
            price_col = "price_log_normalized"
            monopoly_col = "monopoly_prices_log_normalized"
            nash_col = "nash_prices_log_normalized"
            print("✓ All prices normalized by alpha at observation level")
        else:
            price_col = "price"
            monopoly_col = "monopoly_prices"
            nash_col = "nash_prices"

        # Create period and run numeric IDs for panel regression
        processed = processed.with_columns(
            [
                pl.col("run_id")
                .cast(pl.Categorical)
                .to_physical()
                .alias("run_numeric"),
                pl.col("agent_id")
                .cast(pl.Categorical)
                .to_physical()
                .alias("agent_numeric"),
                pl.col("prompt_type")
                .cast(pl.Categorical)
                .to_physical()
                .alias("prompt_numeric"),
                pl.col(price_col).alias("price_analysis"),
                pl.col(monopoly_col).alias("monopoly_analysis"),
                pl.col(nash_col).alias("nash_analysis"),
            ]
        )

        self.processed_df = processed
        self.price_col = "price_analysis"
        self.monopoly_col = "monopoly_analysis"
        self.nash_col = "nash_analysis"

        print(
            f"✓ Preprocessing complete. Analysis periods: {start_period}-{end_period}"
        )
        return processed

    def create_interleaved_data(self, interval: int = 2):
        """
        Create interleaved dataset with agent alternation following original paper methodology
        Creates disjoint period pairs and alternates focal agent between pairs
        For groups >2: Uses average of all other agents as "competitor"
        """
        if not hasattr(self, "processed_df"):
            raise ValueError("Must run preprocess_data() first")

        # Create sophisticated interleaved data with agent alternation
        regression_data = []

        for prompt_type in self.processed_df["prompt_type"].unique():
            prompt_data = self.processed_df.filter(pl.col("prompt_type") == prompt_type)

            for run_id in prompt_data["run_id"].unique():
                run_data = prompt_data.filter(pl.col("run_id") == run_id)

                # Get available periods and agents for this run
                periods = sorted(run_data["period"].unique().to_list())
                agents = sorted(run_data["agent_id"].unique().to_list())
                group_size = len(agents)

                # Skip runs with fewer than 2 agents
                if group_size < 2:
                    print(f"⚠️  Skipping run {run_id} - fewer than 2 agents")
                    continue

                # Create pairs: (101,102), (103,104), (105,106), etc.
                period_pairs = [
                    (periods[i], periods[i + 1])
                    for i in range(0, len(periods) - 1, interval)
                ]

                for pair_idx, (t1, t2) in enumerate(period_pairs):
                    # Cycle through agents as focal agent
                    focal_agent_idx = pair_idx % group_size
                    focal_agent = agents[focal_agent_idx]

                    # Get focal agent's data for both periods
                    current_data = run_data.filter(
                        (pl.col("period") == t2) & (pl.col("agent_id") == focal_agent)
                    )
                    lag_data = run_data.filter(
                        (pl.col("period") == t1) & (pl.col("agent_id") == focal_agent)
                    )

                    # Get all other agents' data for period t1 (to compute competitor average)
                    other_agents = [a for a in agents if a != focal_agent]
                    competitor_lag_data = run_data.filter(
                        (pl.col("period") == t1)
                        & (pl.col("agent_id").is_in(other_agents))
                    )

                    # Ensure we have complete data
                    expected_competitors = len(other_agents)
                    if (
                        current_data.height == 1
                        and lag_data.height == 1
                        and competitor_lag_data.height == expected_competitors
                    ):
                        # Extract the data
                        current_row = current_data.to_pandas().iloc[0]
                        lag_row = lag_data.to_pandas().iloc[0]

                        # Calculate average competitor price
                        competitor_prices = competitor_lag_data[
                            self.price_col
                        ].to_list()
                        avg_competitor_price = sum(competitor_prices) / len(
                            competitor_prices
                        )

                        # Create observation with lagged variables
                        observation = {
                            "run_id": run_id,
                            "prompt_type": prompt_type,
                            "agent_id": focal_agent,
                            "focal_agent_position": focal_agent_idx,  # Track position in rotation
                            "period": t2,
                            "group_size": current_row["group_size"],
                            "n_competitors": expected_competitors,
                            # Current period variables
                            self.price_col: current_row[self.price_col],
                            self.monopoly_col: current_row[self.monopoly_col],
                            self.nash_col: current_row[self.nash_col],
                            # Lagged variables
                            f"{self.price_col}_lag": lag_row[self.price_col],
                            f"competitor_{self.price_col}_lag": avg_competitor_price,
                            f"competitor_{self.price_col}_std": (
                                sum(
                                    (p - avg_competitor_price) ** 2
                                    for p in competitor_prices
                                )
                                / len(competitor_prices)
                            )
                            ** 0.5
                            if len(competitor_prices) > 1
                            else 0.0,  # Competitor price dispersion
                            # Identifiers for clustering
                            "run_numeric": current_row["run_numeric"],
                            "agent_numeric": current_row["agent_numeric"],
                            "prompt_numeric": current_row["prompt_numeric"],
                            "agent_run": f"{focal_agent}_{run_id}",
                        }

                        regression_data.append(observation)

        # Convert back to Polars DataFrame
        if regression_data:
            interleaved = pl.DataFrame(regression_data)

            # Summary statistics
            total_obs = len(regression_data)
            unique_runs = interleaved["run_id"].n_unique()
            group_sizes = interleaved.group_by("group_size").agg(
                pl.count().alias("count")
            )

            print("✓ Sophisticated interleaved data created with agent rotation")
            print(f"  - Total observations: {total_obs}")
            print(f"  - Runs included: {unique_runs}")
            print(
                f"  - Period pairs processed: ~{total_obs // unique_runs if unique_runs > 0 else 0} per run"
            )
            print("  - Group size distribution:")
            for row in group_sizes.to_pandas().itertuples():
                print(f"    Group size {row.group_size}: {row.count} observations")

        else:
            # Fallback to simple interleaving if sophisticated approach fails
            print(
                "⚠️  Sophisticated interleaving failed, falling back to simple approach"
            )
            min_period = self.processed_df["period"].min()
            max_period = self.processed_df["period"].max()
            sampled_periods = list(range(min_period, max_period + 1, interval))
            interleaved = self.processed_df.filter(
                pl.col("period").is_in(sampled_periods)
            )

        self.interleaved_df = interleaved
        return interleaved

    def estimate_group_size_effects(
        self, use_interleaved: bool = True, include_lags: bool = True
    ):
        """
        Main regression: estimate effect of group size on prices
        Two specifications: time FE only (for run-level vars) and agent FE (for within-run variation)
        Now supports lagged variables from sophisticated interleaving
        """
        df_to_use = self.interleaved_df if use_interleaved else self.processed_df

        # Convert to pandas for regression
        pandas_df = df_to_use.to_pandas()

        # Check if we have lagged variables (from sophisticated interleaving)
        has_lags = f"{self.price_col}_lag" in pandas_df.columns

        if include_lags and has_lags:
            print("✓ Using sophisticated interleaved data with lagged variables")
            lag_vars = f" + {self.price_col}_lag + competitor_{self.price_col}_lag"
        else:
            lag_vars = ""
            if include_lags and not has_lags:
                print(
                    "⚠️  Lagged variables requested but not available - using simple specification"
                )

        # Specification 1: Time FE only (for run-level variables like group_size, prompt_type)
        pandas_df_time = pandas_df.set_index(["run_numeric", "period"])
        formula_time = (
            f"{self.price_col} ~ group_size + C(prompt_type){lag_vars} + TimeEffects"
        )

        model_time = PanelOLS.from_formula(
            formula_time, data=pandas_df_time, drop_absorbed=True, check_rank=False
        )
        results_time = model_time.fit(cov_type="clustered", cluster_entity=True)

        # Specification 2: Agent-period level with time FE (more observations)
        if "agent_run" not in pandas_df.columns:
            pandas_df["agent_run"] = (
                pandas_df["run_numeric"].astype(str)
                + "_"
                + pandas_df["agent_numeric"].astype(str)
            )
        pandas_df_agent = pandas_df.set_index(["agent_run", "period"])

        model_agent = PanelOLS.from_formula(
            formula_time, data=pandas_df_agent, drop_absorbed=True, check_rank=False
        )
        results_agent = model_agent.fit(cov_type="clustered", cluster_entity=True)

        # Store results with descriptive names
        suffix = "_with_lags" if (include_lags and has_lags) else ""
        self.results[f"main_time_fe{suffix}"] = results_time
        self.results[f"main_agent_fe{suffix}"] = results_agent

        print(
            f"✓ Main regressions estimated (time FE and agent-level){' with lags' if has_lags and include_lags else ''}"
        )
        return results_time, results_agent

    def estimate_nonlinear_effects(self, include_lags: bool = True):
        """
        Test for non-linear group size effects (Folk Theorem predictions)
        Now supports lagged variables from sophisticated interleaving
        """
        if not hasattr(self, "interleaved_df"):
            self.create_interleaved_data()

        pandas_df = self.interleaved_df.to_pandas()

        # Check if we have lagged variables
        has_lags = f"{self.price_col}_lag" in pandas_df.columns

        if include_lags and has_lags:
            lag_vars = f" + {self.price_col}_lag + competitor_{self.price_col}_lag"
        else:
            lag_vars = ""

        # Agent-level analysis
        if "agent_run" not in pandas_df.columns:
            pandas_df["agent_run"] = (
                pandas_df["run_numeric"].astype(str)
                + "_"
                + pandas_df["agent_numeric"].astype(str)
            )
        pandas_df = pandas_df.set_index(["agent_run", "period"])

        # Add squared term
        pandas_df["group_size_sq"] = pandas_df["group_size"] ** 2

        # Non-linear specification
        formula = f"{self.price_col} ~ group_size + group_size_sq + C(prompt_type){lag_vars} + TimeEffects"

        model = PanelOLS.from_formula(
            formula, data=pandas_df, drop_absorbed=True, check_rank=False
        )
        results = model.fit(cov_type="clustered", cluster_entity=True)

        suffix = "_with_lags" if (include_lags and has_lags) else ""
        self.results[f"nonlinear{suffix}"] = results
        print(
            f"✓ Non-linear effects estimated{' with lags' if has_lags and include_lags else ''}"
        )
        return results

    def estimate_threshold_effects(self, threshold: int = 3, include_lags: bool = True):
        """
        Test Folk Theorem threshold (collusion breaks down at n > threshold)
        Now supports lagged variables from sophisticated interleaving
        """
        if not hasattr(self, "interleaved_df"):
            self.create_interleaved_data()

        pandas_df = self.interleaved_df.to_pandas()

        # Check if we have lagged variables
        has_lags = f"{self.price_col}_lag" in pandas_df.columns

        if include_lags and has_lags:
            lag_vars = f" + {self.price_col}_lag + competitor_{self.price_col}_lag"
        else:
            lag_vars = ""

        if "agent_run" not in pandas_df.columns:
            pandas_df["agent_run"] = (
                pandas_df["run_numeric"].astype(str)
                + "_"
                + pandas_df["agent_numeric"].astype(str)
            )
        pandas_df = pandas_df.set_index(["agent_run", "period"])

        # Create threshold dummy
        pandas_df["small_group"] = (pandas_df["group_size"] <= threshold).astype(int)

        formula = (
            f"{self.price_col} ~ small_group + C(prompt_type){lag_vars} + TimeEffects"
        )

        model = PanelOLS.from_formula(
            formula, data=pandas_df, drop_absorbed=True, check_rank=False
        )
        results = model.fit(cov_type="clustered", cluster_entity=True)

        suffix = "_with_lags" if (include_lags and has_lags) else ""
        self.results[f"threshold{suffix}"] = results
        print(
            f"✓ Threshold effects estimated (threshold = {threshold}){' with lags' if has_lags and include_lags else ''}"
        )
        return results

    def estimate_dynamic_effects(self):
        """
        Analyze dynamic pricing behavior using lagged variables
        Now supports groups of any size using average competitor approach
        """
        if not hasattr(self, "interleaved_df"):
            self.create_interleaved_data()

        pandas_df = self.interleaved_df.to_pandas()

        # Check if we have lagged variables
        has_lags = f"{self.price_col}_lag" in pandas_df.columns

        if not has_lags:
            print(
                "⚠️  Dynamic analysis requires sophisticated interleaving with lagged variables"
            )
            print("    Run create_interleaved_data() with agent alternation first")
            return None

        if "agent_run" not in pandas_df.columns:
            pandas_df["agent_run"] = (
                pandas_df["run_numeric"].astype(str)
                + "_"
                + pandas_df["agent_numeric"].astype(str)
            )
        pandas_df = pandas_df.set_index(["agent_run", "period"])

        # Dynamic specifications

        # 1. Basic dynamic model
        formula_basic = f"{self.price_col} ~ group_size + C(prompt_type) + {self.price_col}_lag + competitor_{self.price_col}_lag + TimeEffects"
        model_basic = PanelOLS.from_formula(
            formula_basic, data=pandas_df, drop_absorbed=True, check_rank=False
        )
        results_basic = model_basic.fit(cov_type="clustered", cluster_entity=True)

        # 2. Interactive dynamic model (group size effects on persistence)
        pandas_df["group_size_x_lag"] = (
            pandas_df["group_size"] * pandas_df[f"{self.price_col}_lag"]
        )
        pandas_df["group_size_x_comp_lag"] = (
            pandas_df["group_size"] * pandas_df[f"competitor_{self.price_col}_lag"]
        )

        formula_interactive = f"{self.price_col} ~ group_size + C(prompt_type) + {self.price_col}_lag + competitor_{self.price_col}_lag + group_size_x_lag + group_size_x_comp_lag + TimeEffects"
        model_interactive = PanelOLS.from_formula(
            formula_interactive, data=pandas_df, drop_absorbed=True, check_rank=False
        )
        results_interactive = model_interactive.fit(
            cov_type="clustered", cluster_entity=True
        )

        # 3. Threshold dynamic model
        pandas_df["small_group"] = (pandas_df["group_size"] <= 3).astype(int)
        pandas_df["small_group_x_lag"] = (
            pandas_df["small_group"] * pandas_df[f"{self.price_col}_lag"]
        )

        formula_threshold = f"{self.price_col} ~ small_group + C(prompt_type) + {self.price_col}_lag + competitor_{self.price_col}_lag + small_group_x_lag + TimeEffects"
        model_threshold = PanelOLS.from_formula(
            formula_threshold, data=pandas_df, drop_absorbed=True, check_rank=False
        )
        results_threshold = model_threshold.fit(
            cov_type="clustered", cluster_entity=True
        )

        # 4. Competitor heterogeneity model (if we have price dispersion data)
        if f"competitor_{self.price_col}_std" in pandas_df.columns:
            formula_heterogeneity = f"{self.price_col} ~ group_size + C(prompt_type) + {self.price_col}_lag + competitor_{self.price_col}_lag + competitor_{self.price_col}_std + TimeEffects"
            model_heterogeneity = PanelOLS.from_formula(
                formula_heterogeneity,
                data=pandas_df,
                drop_absorbed=True,
                check_rank=False,
            )
            results_heterogeneity = model_heterogeneity.fit(
                cov_type="clustered", cluster_entity=True
            )

            # Store heterogeneity results
            self.results["dynamic_heterogeneity"] = results_heterogeneity

        # Store results
        self.results["dynamic_basic"] = results_basic
        self.results["dynamic_interactive"] = results_interactive
        self.results["dynamic_threshold"] = results_threshold

        print("✓ Dynamic effects estimated for groups of all sizes")
        print("  - Basic model: Own lag + Avg competitor lag")
        print("  - Interactive model: Group size × dynamics interactions")
        print("  - Threshold model: Small vs large group dynamics")
        if f"competitor_{self.price_col}_std" in pandas_df.columns:
            print("  - Heterogeneity model: Competitor price dispersion effects")

        return results_basic, results_interactive, results_threshold

    def estimate_prompt_interactions(self):
        """
        Test if group size effects vary by prompt type - using run-level OLS instead
        """
        # Use run-level analysis for interactions since both variables are run-level
        if not hasattr(self, "processed_df"):
            self.preprocess_data()

        # Collapse to run-level averages (final 50 periods)
        max_period = self.processed_df["period"].max()
        final_periods_start = max_period - 49

        run_level = (
            self.processed_df.filter(pl.col("period") >= final_periods_start)
            .group_by(["run_id", "group_size", "prompt_type"])
            .agg(
                [
                    pl.col(self.price_col).mean().alias("avg_price"),
                    pl.col(self.price_col).std().alias("price_volatility"),
                    pl.count().alias("n_obs"),
                ]
            )
        ).to_pandas()

        # OLS with interactions
        formula = "avg_price ~ group_size * C(prompt_type)"
        model = ols(formula, data=run_level).fit(cov_type="HC3")

        self.results["interactions"] = model
        print("✓ Prompt interaction effects estimated (run-level OLS)")
        return model

    def estimate_run_level_ols(self):
        """
        Alternative: Collapse to run-level averages and use OLS
        This is most appropriate for testing group size effects since they vary at run level
        """
        if not hasattr(self, "processed_df"):
            self.preprocess_data()

        # Collapse to run-level averages (final 50 periods)
        max_period = self.processed_df["period"].max()
        final_periods_start = max_period - 49

        run_level = (
            self.processed_df.filter(pl.col("period") >= final_periods_start)
            .group_by(["run_id", "group_size", "prompt_type"])
            .agg(
                [
                    pl.col(self.price_col).mean().alias("avg_price"),
                    pl.col(self.price_col).std().alias("price_volatility"),
                    pl.count().alias("n_obs"),
                ]
            )
        ).to_pandas()

        # Multiple OLS specifications

        # Basic specification
        formula_basic = "avg_price ~ group_size + C(prompt_type)"
        model_basic = ols(formula_basic, data=run_level).fit(cov_type="HC3")

        # Non-linear specification
        run_level["group_size_sq"] = run_level["group_size"] ** 2
        formula_nonlinear = "avg_price ~ group_size + group_size_sq + C(prompt_type)"
        model_nonlinear = ols(formula_nonlinear, data=run_level).fit(cov_type="HC3")

        # Threshold specification
        run_level["small_group"] = (run_level["group_size"] <= 2).astype(int)
        formula_threshold = "avg_price ~ small_group + C(prompt_type)"
        model_threshold = ols(formula_threshold, data=run_level).fit(cov_type="HC3")

        self.results["run_level_basic"] = model_basic
        self.results["run_level_nonlinear"] = model_nonlinear
        self.results["run_level_threshold"] = model_threshold
        self.run_level_data = run_level

        print("✓ Run-level OLS estimated (basic, nonlinear, threshold)")
        return model_basic, model_nonlinear, model_threshold

    def calculate_collusion_metrics(self):
        """
        Calculate key collusion metrics by group size using normalized benchmarks
        Now using observation-level normalized benchmarks, then aggregating
        """
        if not hasattr(self, "processed_df"):
            self.preprocess_data()

        # Use final 50 periods for convergence
        max_period = self.processed_df["period"].max()
        final_periods_start = max_period - 49

        # Calculate metrics by group size and prompt type
        # Now we can aggregate the already-normalized benchmark prices
        metrics = (
            self.processed_df.filter(pl.col("period") >= final_periods_start)
            .group_by(["group_size", "prompt_type"])
            .agg(
                [
                    pl.col(self.price_col).mean().alias("avg_price"),
                    pl.col(self.price_col).std().alias("price_volatility"),
                    pl.col(self.price_col).quantile(0.25).alias("price_p25"),
                    pl.col(self.price_col).quantile(0.75).alias("price_p75"),
                    pl.col(self.monopoly_col).mean().alias("avg_monopoly_price"),
                    pl.col(self.nash_col).mean().alias("avg_nash_price"),
                    pl.count().alias("n_observations"),
                ]
            )
            .sort(["group_size", "prompt_type"])
        )

        # Calculate collusion index using the aggregated normalized benchmarks
        metrics = metrics.with_columns(
            [
                (
                    (pl.col("avg_price") - pl.col("avg_nash_price"))
                    / (pl.col("avg_monopoly_price") - pl.col("avg_nash_price"))
                ).alias("collusion_index")
            ]
        )

        # Add a sanity check for the collusion index
        print(
            "✓ Collusion metrics calculated with observation-level normalized benchmarks"
        )

        # Check for any unusual collusion index values
        metrics_summary = metrics.to_pandas()
        extreme_values = metrics_summary[
            (metrics_summary["collusion_index"] < -0.5)
            | (metrics_summary["collusion_index"] > 1.5)
        ]

        if len(extreme_values) > 0:
            print("⚠️  Warning: Found extreme collusion index values:")
            print(
                extreme_values[
                    [
                        "group_size",
                        "prompt_type",
                        "avg_price",
                        "avg_nash_price",
                        "avg_monopoly_price",
                        "collusion_index",
                    ]
                ]
            )
        else:
            print("✓ Collusion index values appear reasonable (mostly between 0 and 1)")

        self.collusion_metrics = metrics
        return metrics

    def plot_results(self):
        """
        Create visualization of results with normalized benchmarks
        """
        if not hasattr(self, "collusion_metrics"):
            self.calculate_collusion_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Convert to pandas for seaborn
        plot_df = self.collusion_metrics.to_pandas()

        # Create benchmark dataframe from the aggregated data
        benchmark_df = (
            plot_df[["group_size", "avg_nash_price", "avg_monopoly_price"]]
            .drop_duplicates()
            .sort_values("group_size")
        )

        # Price by group size
        sns.lineplot(
            data=plot_df,
            x="group_size",
            y="avg_price",
            hue="prompt_type",
            marker="o",
            ax=axes[0, 0],
        )

        # Add dynamic benchmark lines
        axes[0, 0].plot(
            benchmark_df["group_size"],
            benchmark_df["avg_nash_price"],
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Nash Equilibrium",
        )
        axes[0, 0].plot(
            benchmark_df["group_size"],
            benchmark_df["avg_monopoly_price"],
            color="green",
            linestyle="--",
            alpha=0.7,
            label="Monopoly",
        )

        axes[0, 0].set_title("Average Price by Group Size")
        axes[0, 0].set_ylabel("Normalized Price")
        axes[0, 0].legend()

        # Collusion index by group size
        sns.lineplot(
            data=plot_df,
            x="group_size",
            y="collusion_index",
            hue="prompt_type",
            marker="s",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Collusion Index by Group Size")
        axes[0, 1].set_ylabel("Collusion Index (0=Nash, 1=Monopoly)")
        axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Nash")
        axes[0, 1].axhline(
            y=1, color="green", linestyle="--", alpha=0.5, label="Monopoly"
        )
        axes[0, 1].legend()

        # Price volatility
        sns.lineplot(
            data=plot_df,
            x="group_size",
            y="price_volatility",
            hue="prompt_type",
            marker="^",
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Price Volatility by Group Size")
        axes[1, 0].set_ylabel("Price Standard Deviation")

        # Sample size check
        sns.barplot(
            data=plot_df,
            x="group_size",
            y="n_observations",
            hue="prompt_type",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Number of Observations")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.show()

        # Additional plot: Run-level scatter if available
        if hasattr(self, "run_level_data"):
            plt.figure(figsize=(12, 8))

            # Create subplots for run-level analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Scatter plot
            sns.scatterplot(
                data=self.run_level_data,
                x="group_size",
                y="avg_price",
                hue="prompt_type",
                s=100,
                alpha=0.7,
                ax=axes[0, 0],
            )

            # Add benchmark lines to scatter plot
            axes[0, 0].plot(
                benchmark_df["group_size"],
                benchmark_df["avg_nash_price"],
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Nash",
            )
            axes[0, 0].plot(
                benchmark_df["group_size"],
                benchmark_df["avg_monopoly_price"],
                color="green",
                linestyle="--",
                alpha=0.7,
                label="Monopoly",
            )

            axes[0, 0].set_title("Run-Level Average Prices by Group Size")
            axes[0, 0].set_xlabel("Group Size")
            axes[0, 0].set_ylabel("Average Price (Final 50 Periods)")
            axes[0, 0].legend()

            # Box plot
            sns.boxplot(
                data=self.run_level_data,
                x="group_size",
                y="avg_price",
                hue="prompt_type",
                ax=axes[0, 1],
            )
            axes[0, 1].set_title("Price Distribution by Group Size")

            # Volatility by group size
            sns.scatterplot(
                data=self.run_level_data,
                x="group_size",
                y="price_volatility",
                hue="prompt_type",
                s=100,
                alpha=0.7,
                ax=axes[1, 0],
            )
            axes[1, 0].set_title("Price Volatility by Group Size")
            axes[1, 0].set_xlabel("Group Size")
            axes[1, 0].set_ylabel("Price Volatility")

            # Sample sizes
            group_counts = (
                self.run_level_data.groupby(["group_size", "prompt_type"])
                .size()
                .reset_index(name="count")
            )
            sns.barplot(
                data=group_counts,
                x="group_size",
                y="count",
                hue="prompt_type",
                ax=axes[1, 1],
            )
            axes[1, 1].set_title("Number of Runs by Group Size")
            axes[1, 1].set_ylabel("Number of Runs")

            plt.tight_layout()
            plt.show()

    def print_summary(self):
        """
        Print summary of all results including benchmark comparison
        """
        print("\n" + "=" * 60)
        print("COLLUSION BREAKDOWN ANALYSIS - SUMMARY RESULTS")
        print("=" * 60)

        # Print benchmark summary
        print("\nTHEORETICAL BENCHMARKS BY GROUP SIZE:")
        print("-" * 40)

        if hasattr(self, "benchmarks_raw"):
            print("Raw benchmarks (before normalization):")
            raw_summary = self.benchmarks_raw.to_pandas()
            for _, row in raw_summary.iterrows():
                print(
                    f"  Group Size {int(row['group_size'])}: Nash={row['nash_prices']:.3f}, Monopoly={row['monopoly_prices']:.3f}, Alpha={row['alpha']:.3f}"
                )

        if hasattr(self, "collusion_metrics"):
            print("Normalized benchmarks (used in analysis):")
            benchmark_summary = (
                self.collusion_metrics.select(
                    ["group_size", "avg_nash_price", "avg_monopoly_price"]
                )
                .unique()
                .sort("group_size")
                .to_pandas()
            )
            for _, row in benchmark_summary.iterrows():
                print(
                    f"  Group Size {int(row['group_size'])}: Nash={row['avg_nash_price']:.3f}, Monopoly={row['avg_monopoly_price']:.3f}"
                )

        # Print regression results
        print("\nREGRESSION RESULTS:")
        print("=" * 50)

        # Group main results by type
        main_results = {k: v for k, v in self.results.items() if k.startswith("main_")}
        nonlinear_results = {
            k: v for k, v in self.results.items() if k.startswith("nonlinear")
        }
        threshold_results = {
            k: v for k, v in self.results.items() if k.startswith("threshold")
        }
        dynamic_results = {
            k: v for k, v in self.results.items() if k.startswith("dynamic_")
        }
        other_results = {
            k: v
            for k, v in self.results.items()
            if not any(
                k.startswith(prefix)
                for prefix in ["main_", "nonlinear", "threshold", "dynamic_"]
            )
        }

        # Print each group
        for group_name, group_results in [
            ("MAIN EFFECTS", main_results),
            ("NON-LINEAR EFFECTS", nonlinear_results),
            ("THRESHOLD EFFECTS", threshold_results),
            ("DYNAMIC EFFECTS", dynamic_results),
            ("OTHER SPECIFICATIONS", other_results),
        ]:
            if group_results:
                print(f"\n{group_name}:")
                print("-" * 40)
                for name, results in group_results.items():
                    print(f"\n{name.upper().replace('_', ' ')}:")
                    if hasattr(results, "summary"):
                        try:
                            # Try calling as method first (statsmodels)
                            print(results.summary())
                        except TypeError:
                            # If that fails, access as property (PanelOLS)
                            print(results.summary)
                    else:
                        print(results)
                        print(results.summary().tables[1])

        # Print key statistics
        if hasattr(self, "collusion_metrics"):
            print("\nCOLLUSION METRICS BY GROUP SIZE:")
            print("-" * 40)
            metrics_summary = self.collusion_metrics.select(
                [
                    "group_size",
                    "prompt_type",
                    "avg_price",
                    "collusion_index",
                    "avg_nash_price",
                    "avg_monopoly_price",
                    "n_observations",
                ]
            ).to_pandas()

            for group_size in sorted(metrics_summary["group_size"].unique()):
                group_data = metrics_summary[
                    metrics_summary["group_size"] == group_size
                ]
                print(f"\nGroup Size {int(group_size)}:")
                for _, row in group_data.iterrows():
                    print(
                        f"  {row['prompt_type']}: Price={row['avg_price']:.3f}, "
                        f"Collusion Index={row['collusion_index']:.3f}, N={int(row['n_observations'])}"
                    )

        # Folk Theorem test summary
        if "run_level_basic" in self.results:
            coef = self.results["run_level_basic"].params.get("group_size", None)
            if coef is not None:
                pval = self.results["run_level_basic"].pvalues.get("group_size", None)
                print("\nFOLK THEOREM TEST (Run-Level Analysis):")
                print("-" * 40)
                print(f"Group size coefficient: {coef:.4f}")
                print(f"P-value: {pval:.4f}")
                print(f"Significant at 5%: {'Yes' if pval < 0.05 else 'No'}")
                print(
                    f"Direction: {'Collusion decreases with group size' if coef < 0 else 'Collusion increases with group size'}"
                )

        # Print threshold test if available
        if "run_level_threshold" in self.results:
            coef = self.results["run_level_threshold"].params.get("small_group", None)
            if coef is not None:
                pval = self.results["run_level_threshold"].pvalues.get(
                    "small_group", None
                )
                print("\nTHRESHOLD EFFECT TEST (Groups ≤3 vs >3):")
                print("-" * 40)
                print(f"Small group coefficient: {coef:.4f}")
                print(f"P-value: {pval:.4f}")
                print(f"Significant at 5%: {'Yes' if pval < 0.05 else 'No'}")

        # Dynamic analysis summary (NEW)
        if any(k.startswith("dynamic_") for k in self.results.keys()):
            print("\nDYNAMIC ANALYSIS SUMMARY (Agent Alternation Approach):")
            print("=" * 55)

            # Price persistence
            if "dynamic_basic" in self.results:
                lag_coef = self.results["dynamic_basic"].params.get(
                    f"{self.price_col}_lag", None
                )
                comp_lag_coef = self.results["dynamic_basic"].params.get(
                    f"competitor_{self.price_col}_lag", None
                )

                if lag_coef is not None:
                    lag_pval = self.results["dynamic_basic"].pvalues.get(
                        f"{self.price_col}_lag", None
                    )
                    print(f"Own price persistence: {lag_coef:.4f} (p={lag_pval:.4f})")

                if comp_lag_coef is not None:
                    comp_pval = self.results["dynamic_basic"].pvalues.get(
                        f"competitor_{self.price_col}_lag", None
                    )
                    print(
                        f"Competitor price response: {comp_lag_coef:.4f} (p={comp_pval:.4f})"
                    )

            # Group size interactions with dynamics
            if "dynamic_interactive" in self.results:
                group_lag_coef = self.results["dynamic_interactive"].params.get(
                    "group_size_x_lag", None
                )
                if group_lag_coef is not None:
                    group_lag_pval = self.results["dynamic_interactive"].pvalues.get(
                        "group_size_x_lag", None
                    )
                    print(
                        f"Group size × Own lag interaction: {group_lag_coef:.4f} (p={group_lag_pval:.4f})"
                    )
                    interpretation = (
                        "Larger groups have MORE persistent pricing"
                        if group_lag_coef > 0
                        else "Larger groups have LESS persistent pricing"
                    )
                    print(f"  → {interpretation}")

            # Threshold dynamic effects
            if "dynamic_threshold" in self.results:
                threshold_lag_coef = self.results["dynamic_threshold"].params.get(
                    "small_group_x_lag", None
                )
                if threshold_lag_coef is not None:
                    threshold_lag_pval = self.results["dynamic_threshold"].pvalues.get(
                        "small_group_x_lag", None
                    )
                    print(
                        f"Small group × Own lag interaction: {threshold_lag_coef:.4f} (p={threshold_lag_pval:.4f})"
                    )
                    interpretation = (
                        "Small groups have MORE persistent pricing"
                        if threshold_lag_coef > 0
                        else "Small groups have LESS persistent pricing"
                    )
                    print(f"  → {interpretation}")

            # Competitor heterogeneity effects (NEW for groups >2)
            if "dynamic_heterogeneity" in self.results:
                hetero_coef = self.results["dynamic_heterogeneity"].params.get(
                    f"competitor_{self.price_col}_std", None
                )
                if hetero_coef is not None:
                    hetero_pval = self.results["dynamic_heterogeneity"].pvalues.get(
                        f"competitor_{self.price_col}_std", None
                    )
                    print(
                        f"Competitor price dispersion effect: {hetero_coef:.4f} (p={hetero_pval:.4f})"
                    )
                    interpretation = (
                        "Higher competitor dispersion increases own price"
                        if hetero_coef > 0
                        else "Higher competitor dispersion decreases own price"
                    )
                    print(f"  → {interpretation}")

        print("\nDATA STRUCTURE SUMMARY:")
        print("-" * 40)
        if hasattr(self, "interleaved_df"):
            has_lags = f"{self.price_col}_lag" in self.interleaved_df.columns
            print(
                f"Interleaving approach: {'Sophisticated (with agent rotation)' if has_lags else 'Simple (period sampling)'}"
            )
            print(f"Lagged variables available: {'Yes' if has_lags else 'No'}")
            print(
                f"Multi-agent support: {'Yes (average competitor approach)' if has_lags else 'Limited to 2 agents'}"
            )

            if has_lags:
                print(
                    f"Total observations in interleaved data: {self.interleaved_df.height}"
                )
                print(
                    f"Unique runs in analysis: {self.interleaved_df['run_id'].n_unique()}"
                )

                # Show agent rotation summary
                if "focal_agent_position" in self.interleaved_df.columns:
                    rotation_summary = self.interleaved_df.group_by(
                        ["group_size", "focal_agent_position"]
                    ).agg(pl.count().alias("count"))
                    print("Agent rotation summary:")
                    for size in sorted(rotation_summary["group_size"].unique()):
                        size_data = rotation_summary.filter(
                            pl.col("group_size") == size
                        )
                        positions = size_data["focal_agent_position"].to_list()
                        counts = size_data["count"].to_list()
                        print(f"  Group size {size}: {dict(zip(positions, counts))}")

                # Show competitor analysis approach
                if "n_competitors" in self.interleaved_df.columns:
                    competitor_summary = self.interleaved_df.group_by(
                        "n_competitors"
                    ).agg(pl.count().alias("count"))
                    print("Competitor analysis:")
                    for row in competitor_summary.to_pandas().itertuples():
                        print(
                            f"  {row.n_competitors} competitors per focal agent: {row.count} observations"
                        )
        else:
            print("No interleaved data created yet")

    def convergence_analysis_fixed(self, final_periods: int = 50):
        """
        Proper convergence analysis avoiding the time-invariant variable problem
        """

        print("CONVERGENCE ANALYSIS - FIXED SPECIFICATION")
        print("=" * 60)

        max_period = self.processed_df["period"].max()
        start_period = max_period - final_periods + 1

        convergence_df = self.processed_df.filter(
            pl.col("period") >= start_period
        ).to_pandas()

        # ================================================================
        # Option 1: Time Effects Only (No Entity FE)
        # ================================================================
        print("\n1. TIME EFFECTS ONLY (Proper for time-invariant treatments)")
        print("-" * 50)

        # Create time dummies manually
        convergence_df["period_factor"] = pd.Categorical(convergence_df["period"])

        # Regression with time effects only
        formula_time = (
            f"{self.price_col} ~ group_size + C(prompt_type) + C(period_factor)"
        )
        model_time = ols(formula_time, data=convergence_df).fit(
            cov_type="cluster", cov_kwds={"groups": convergence_df["run_id"]}
        )

        # print("Time Effects Model:")
        # print(
        #     f"  Group size: {model_time.params['group_size']:.4f} (SE: {model_time.bse['group_size']:.4f})"
        # )
        # print(
        #     f"  Prompt P1: {model_time.params['C(prompt_type)[T.P1]']:.4f} (SE: {model_time.bse['C(prompt_type)[T.P1]']:.4f})"
        # )
        # print(f"  R-squared: {model_time.rsquared:.4f}")
        # print(f"  N: {len(convergence_df):,}")
        print(model_time.summary())

        # ================================================================
        # Option 2: Random Effects (Better for time-invariant variables)
        # ================================================================
        print("\n2. RANDOM EFFECTS (Alternative approach)")
        print("-" * 50)

        # Prepare for statsmodels RandomEffects
        convergence_df["agent_run"] = (
            convergence_df["run_numeric"].astype(str)
            + "_"
            + convergence_df["agent_numeric"].astype(str)
        )
        convergence_df = convergence_df.set_index(["agent_run", "period"])

        try:
            from linearmodels import RandomEffects

            formula_re = f"{self.price_col} ~ group_size + C(prompt_type) + TimeEffects"
            model_re = RandomEffects.from_formula(formula_re, data=convergence_df).fit(
                cov_type="clustered", cluster_entity=True
            )

            print(model_re)

        except Exception as e:
            print(f"Random Effects failed: {e}")
            model_re = None

        # ================================================================
        # Option 3: Run-Level Analysis (Most Appropriate)
        # ================================================================
        print("\n3. RUN-LEVEL ANALYSIS (Most appropriate for convergence)")
        print("-" * 50)

        # Collapse to run-level averages for final periods
        run_level_convergence = (
            self.processed_df.filter(pl.col("period") >= start_period)
            .group_by(["run_id", "group_size", "prompt_type"])
            .agg(
                [
                    pl.col(self.price_col).mean().alias("avg_price_final"),
                    pl.col(self.price_col).std().alias("price_volatility_final"),
                    pl.col(self.monopoly_col).mean().alias("monopoly_price"),
                    pl.col(self.nash_col).mean().alias("nash_price"),
                    pl.count().alias("n_obs"),
                ]
            )
        ).to_pandas()

        # Calculate convergence metrics
        run_level_convergence["collusion_index_final"] = (
            run_level_convergence["avg_price_final"]
            - run_level_convergence["nash_price"]
        ) / (
            run_level_convergence["monopoly_price"]
            - run_level_convergence["nash_price"]
        )

        # Run-level convergence regression
        formula_run = "avg_price_final ~ group_size + C(prompt_type)"
        model_run = ols(formula_run, data=run_level_convergence).fit(cov_type="HC3")

        print(model_run.summary())

        # ================================================================
        # Convergence Diagnostics
        # ================================================================
        print("\n4. CONVERGENCE DIAGNOSTICS")
        print("-" * 50)

        # Price volatility by group size (convergence indicator)
        volatility_by_group = run_level_convergence.groupby("group_size")[
            "price_volatility_final"
        ].agg(["mean", "std", "count"])

        print("Price volatility in final periods (convergence indicator):")
        for group_size, row in volatility_by_group.iterrows():
            print(
                f"  Group size {group_size}: volatility = {row['mean']:.4f} (N={row['count']})"
            )

        # Test if volatility decreases with time (within final periods)
        if len(convergence_df) > 0:
            # Reset index for this analysis
            volatility_df = convergence_df.reset_index()
            volatility_by_period = (
                volatility_df.groupby("period")[self.price_col].std().reset_index()
            )
            volatility_by_period.columns = ["period", "price_std"]

            # Simple trend test
            from scipy.stats import pearsonr

            if len(volatility_by_period) > 1:
                corr, p_value = pearsonr(
                    volatility_by_period["period"], volatility_by_period["price_std"]
                )
                print("\nVolatility trend over final periods:")
                print(f"  Correlation with time: {corr:.4f} (p = {p_value:.4f})")
                if corr < 0 and p_value < 0.05:
                    print("  ✓ Evidence of convergence (volatility decreasing)")
                else:
                    print("  ⚠️ Limited evidence of convergence")

        # Store results
        self.results["convergence_time_fe"] = model_time
        self.results["convergence_run_level"] = model_run
        if model_re:
            self.results["convergence_random_effects"] = model_re

        self.convergence_data = {
            "period_level": convergence_df.reset_index(),
            "run_level": run_level_convergence,
            "volatility_by_group": volatility_by_group,
        }

        print(f"\n✓ Convergence analysis complete (final {final_periods} periods)")
        print(
            "✓ Used appropriate specifications avoiding time-invariant variable problems"
        )

        return {
            "time_effects_model": model_time,
            "run_level_model": model_run,
            "random_effects_model": model_re,
            "convergence_diagnostics": volatility_by_group,
        }

    ##########################################################  convergence ########################################################

    def simple_convergence_test(self, stability_threshold=0.01, min_stable_periods=15):
        """
        Simple test:
        1. When do prices converge for each run?
        2. Do smaller groups converge faster than larger groups?

        Add this method to your existing analysis class in group_size.py
        """

        print("SIMPLE CONVERGENCE ANALYSIS")
        print("=" * 50)

        convergence_data = []

        # Test each run individually
        for run_id in self.processed_df["run_id"].unique():
            run_data = (
                self.processed_df.filter(pl.col("run_id") == run_id)
                .sort("period")
                .to_pandas()
            )

            if len(run_data) < 30:  # Need minimum data
                continue

            group_size = run_data["group_size"].iloc[0]
            prompt_type = run_data["prompt_type"].iloc[0]
            prices = run_data[self.price_col].values
            periods = run_data["period"].values

            # Find convergence point: when price changes become small and stay small
            convergence_period = None

            for i in range(10, len(prices) - min_stable_periods):
                # Look at next min_stable_periods
                future_prices = prices[i : i + min_stable_periods]

                # Check if prices are stable (small changes)
                price_changes = np.abs(np.diff(future_prices))
                max_change = np.max(price_changes)

                if max_change <= stability_threshold:
                    convergence_period = periods[i]
                    break

            # Calculate final equilibrium stats
            if convergence_period is not None:
                final_mask = periods >= convergence_period
                final_prices = prices[final_mask]
                final_mean = np.mean(final_prices)
                final_std = np.std(final_prices)
            else:
                final_mean = np.mean(prices[-min_stable_periods:])  # Use last periods
                final_std = np.std(prices[-min_stable_periods:])

            convergence_data.append(
                {
                    "run_id": run_id,
                    "group_size": group_size,
                    "prompt_type": prompt_type,
                    "convergence_period": convergence_period,
                    "converged": convergence_period is not None,
                    "final_price": final_mean,
                    "final_volatility": final_std,
                    "total_periods": len(prices),
                }
            )

        convergence_df = pd.DataFrame(convergence_data)

        # ================================================================
        # ANALYSIS 1: Convergence Success by Group Size
        # ================================================================
        print("\n1. CONVERGENCE SUCCESS BY GROUP SIZE")
        print("-" * 40)

        convergence_summary = (
            convergence_df.groupby("group_size")
            .agg(
                {
                    "converged": ["count", "sum", "mean"],
                    "convergence_period": ["mean", "std"],
                    "final_price": ["mean", "std"],
                }
            )
            .round(3)
        )

        print(convergence_summary)

        # ================================================================
        # ANALYSIS 2: Do Smaller Groups Converge Faster?
        # ================================================================
        print("\n2. CONVERGENCE TIMING ANALYSIS")
        print("-" * 40)

        converged_only = convergence_df[convergence_df["converged"]].copy()

        if len(converged_only) > 10:
            # Simple regression: convergence_period ~ group_size
            from scipy import stats

            group_sizes = converged_only["group_size"].values
            conv_periods = converged_only["convergence_period"].values

            # Correlation test
            correlation, p_value = stats.pearsonr(group_sizes, conv_periods)

            print("Correlation between group size and convergence timing:")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  P-value: {p_value:.4f}")

            if correlation > 0 and p_value < 0.05:
                print("  ✓ Larger groups take significantly longer to converge")
            elif correlation < 0 and p_value < 0.05:
                print("  ✓ Smaller groups take significantly longer to converge")
            else:
                print("  ○ No significant difference in convergence timing")

            # Group averages
            timing_by_group = converged_only.groupby("group_size")[
                "convergence_period"
            ].agg(["count", "mean", "std"])
            print("\nAverage convergence periods by group size:")
            for group_size, row in timing_by_group.iterrows():
                print(
                    f"  {group_size} agents: {row['mean']:.1f} ± {row['std']:.1f} periods (n={row['count']})"
                )

            # Simple ANOVA test
            group_data = [
                converged_only[converged_only["group_size"] == gs][
                    "convergence_period"
                ].values
                for gs in converged_only["group_size"].unique()
            ]
            group_data = [
                g for g in group_data if len(g) > 1
            ]  # Remove groups with only 1 observation

            if len(group_data) > 1:
                f_stat, p_anova = stats.f_oneway(*group_data)
                print("\nANOVA test for timing differences:")
                print(f"  F-statistic: {f_stat:.4f}")
                print(f"  P-value: {p_anova:.4f}")

                if p_anova < 0.05:
                    print(
                        "  ✓ Significant differences in convergence timing across group sizes"
                    )
                else:
                    print("  ○ No significant differences in convergence timing")

        else:
            print("Not enough converged runs for timing analysis")

        # ================================================================
        # ANALYSIS 3: Final Price Analysis (Your Existing Folk Theorem Test)
        # ================================================================
        print("\n3. FINAL PRICE DIFFERENCES (FOLK THEOREM)")
        print("-" * 40)

        # Use all runs (converged and non-converged) for final price comparison
        formula = "final_price ~ group_size + C(prompt_type)"
        model = ols(formula, data=convergence_df).fit(cov_type="HC3")

        print("Group size effect on final prices:")
        print(f"  Coefficient: {model.params['group_size']:.4f}")
        print(f"  P-value: {model.pvalues['group_size']:.4f}")
        print(f"  R-squared: {model.rsquared:.4f}")

        if model.params["group_size"] < 0 and model.pvalues["group_size"] < 0.05:
            print(
                "  ✓ Folk Theorem confirmed: Prices decrease significantly with group size"
            )
        else:
            print("  ○ Folk Theorem not confirmed")

        # ================================================================
        # SIMPLE VISUALIZATION
        # ================================================================
        self.plot_simple_convergence(convergence_df)

        # Store results
        self.results["simple_convergence"] = convergence_df

        return convergence_df

    def plot_simple_convergence(self, convergence_df):
        """
        Simple plots for convergence analysis
        """

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Convergence success rate
        conv_rates = convergence_df.groupby("group_size")["converged"].agg(
            ["count", "mean"]
        )

        axes[0].bar(conv_rates.index, conv_rates["mean"], alpha=0.7, color="steelblue")
        axes[0].set_title("Convergence Success Rate")
        axes[0].set_xlabel("Number of Agents")
        axes[0].set_ylabel("Success Rate")
        axes[0].set_ylim(0, 1)

        # Add count labels
        for i, (idx, row) in enumerate(conv_rates.iterrows()):
            axes[0].text(
                idx,
                row["mean"] + 0.02,
                f"n={row['count']}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 2: Convergence timing (only converged runs)
        converged_data = convergence_df[convergence_df["converged"]]
        if len(converged_data) > 0:
            sns.boxplot(
                data=converged_data, x="group_size", y="convergence_period", ax=axes[1]
            )
            axes[1].set_title("Convergence Timing")
            axes[1].set_xlabel("Number of Agents")
            axes[1].set_ylabel("Convergence Period")

        # Plot 3: Final prices
        sns.boxplot(data=convergence_df, x="group_size", y="final_price", ax=axes[2])
        axes[2].set_title("Final Price Levels")
        axes[2].set_xlabel("Number of Agents")
        axes[2].set_ylabel("Final Price")

        plt.tight_layout()
        plt.show()

        return fig

    # ================================================================
    # HOW TO USE: Add this to your existing analysis
    # ================================================================


################################################################################################################################
# Updated usage function
def run_analysis(df: pl.DataFrame):
    """
    Complete analysis pipeline with sophisticated agent alternation
    """
    # Initialize analysis
    analysis = CollusionAnalysis(df)

    # Preprocess data
    analysis.preprocess_data(start_period=101, end_period=300)

    # Create sophisticated interleaved data with agent alternation
    analysis.create_interleaved_data(interval=2)

    # Run all regressions (both with and without lags for comparison)
    print("\n" + "=" * 50)
    print("ESTIMATING MAIN EFFECTS")
    print("=" * 50)
    analysis.estimate_group_size_effects(include_lags=False)  # Traditional approach
    analysis.estimate_group_size_effects(include_lags=True)  # With dynamic effects

    print("\n" + "=" * 50)
    print("ESTIMATING NON-LINEAR EFFECTS")
    print("=" * 50)
    analysis.estimate_nonlinear_effects(include_lags=False)
    analysis.estimate_nonlinear_effects(include_lags=True)

    print("\n" + "=" * 50)
    print("ESTIMATING THRESHOLD EFFECTS")
    print("=" * 50)
    analysis.estimate_threshold_effects(threshold=3, include_lags=False)
    analysis.estimate_threshold_effects(threshold=3, include_lags=True)

    print("\n" + "=" * 50)
    print("ESTIMATING DYNAMIC EFFECTS (NEW)")
    print("=" * 50)
    analysis.estimate_dynamic_effects()

    print("\n" + "=" * 50)
    print("ESTIMATING OTHER SPECIFICATIONS")
    print("=" * 50)
    analysis.estimate_prompt_interactions()
    analysis.convergence_analysis(final_periods=50)

    # Run-level analysis (most appropriate for group size effects)
    analysis.estimate_run_level_ols()

    # Calculate metrics and visualize
    analysis.calculate_collusion_metrics()
    analysis.plot_results()

    # Print summary
    analysis.print_summary()

    return analysis

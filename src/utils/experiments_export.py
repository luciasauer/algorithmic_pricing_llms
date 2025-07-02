import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import operator
import polars as pl
from pathlib import Path
from functools import reduce
from src.prompts.prompts import P1, P2

SAMPLING_SEED = 12345


def rebalance_experiments(df_all, min_required=7, keep_exacly_min_required=True):
    alphas_of_interest = [1, 3.2, 10]
    prefix_types = ["P1", "P2"]
    group_keys = ["is_synthetic", "is_symmetric", "num_agents"]

    # Step 1: Count experiments per alpha and agent_prefix_type (only num_agents > 1)
    df_counts = (
        df_all.filter(pl.col("num_agents") > 1)
        .select(
            [
                "experiment_timestamp",
                "experiment_name",
                "num_agents",
                "is_synthetic",
                "is_symmetric",
                "alpha",
                "agent_prefix_type",
            ]
        )
        .unique()
        .group_by(
            ["is_synthetic", "is_symmetric", "num_agents", "alpha", "agent_prefix_type"]
        )
        .agg(pl.len().alias("num_experiments"))
    )

    incomplete_groups = []

    # Step 2: Check completeness of groups for num_agents > 1
    for group_vals in df_counts.select(group_keys).unique().iter_rows(named=True):
        mask = reduce(operator.and_, [pl.col(k) == v for k, v in group_vals.items()])
        df_sub = df_counts.filter(mask)

        group_ok = True
        for prefix in prefix_types:
            for alpha in alphas_of_interest:
                num_runs = (
                    df_sub.filter(
                        (pl.col("agent_prefix_type") == prefix)
                        & (pl.col("alpha") == alpha)
                    )
                    .select("num_experiments")
                    .to_series()
                )
                num_runs = num_runs[0] if len(num_runs) > 0 else 0

                if num_runs < min_required:
                    print(
                        f"-----> Missing experiment runs on: is_synthetic={group_vals['is_synthetic']}, "
                        f"is_symmetric={group_vals['is_symmetric']}, num_agents={group_vals['num_agents']} "
                        f"for prefix_type={prefix} and alpha={alpha} (only {num_runs} runs)"
                    )
                    group_ok = False

        if not group_ok:
            incomplete_groups.append(group_vals)

    if incomplete_groups:
        print(f"\nTotal incomplete groups: {len(incomplete_groups)}")
        print("Rebalancing will be done only on complete groups.")
    else:
        print("All groups meet the minimum requirements.")

    # Step 3: Prepare set for incomplete groups
    incomplete_set = {tuple(d.items()) for d in incomplete_groups}

    def is_complete(row):
        key = tuple((k, row[k]) for k in group_keys)
        return key not in incomplete_set

    rows = df_counts.select(group_keys).to_dicts()
    complete_mask = [is_complete(row) for row in rows]
    df_complete = df_counts.filter(pl.Series(complete_mask))

    rebalance_list = []

    # Step 4: Handle num_agents == 1 — include all runs
    df_num_agents_1 = df_all.filter(pl.col("num_agents") == 1)
    if df_num_agents_1.height > 0:
        rebalance_list.append(df_num_agents_1)

    # Step 5: Handle num_agents > 1 — rebalance only complete groups
    for group_vals in df_complete.select(group_keys).unique().iter_rows(named=True):
        mask = reduce(operator.and_, [pl.col(k) == v for k, v in group_vals.items()])
        df_sub = df_complete.filter(mask)

        min_runs_per_prefix = {}
        for prefix in prefix_types:
            prefix_runs = []
            for alpha in alphas_of_interest:
                runs = df_sub.filter(
                    (pl.col("agent_prefix_type") == prefix) & (pl.col("alpha") == alpha)
                )["num_experiments"][0]
                prefix_runs.append(runs)
            min_runs_per_prefix[prefix] = min(prefix_runs)

        # Enforce global min runs across all prefixes
        global_min_runs = min(min_runs_per_prefix.values())
        if keep_exacly_min_required:
            global_min_runs = min(global_min_runs, min_required)
        else:
            global_min_runs = max(global_min_runs, min_required)

        for prefix in prefix_types:
            for alpha in alphas_of_interest:
                n_samples = global_min_runs

                mask_exp = (
                    (pl.col("is_synthetic") == group_vals["is_synthetic"])
                    & (pl.col("is_symmetric") == group_vals["is_symmetric"])
                    & (pl.col("num_agents") == group_vals["num_agents"])
                    & (pl.col("agent_prefix_type") == prefix)
                    & (pl.col("alpha") == alpha)
                )
                df_exps = df_all.filter(mask_exp)

                unique_exps = (
                        df_exps
                        .select(["experiment_timestamp", "experiment_name"])
                        .unique()
                        .sort(["experiment_timestamp", "experiment_name"])  # <--- Add this
                    )

                n_unique = unique_exps.height
                if n_unique <= n_samples:
                    sampled_exp = unique_exps
                else:
                    sampled_exp = unique_exps.sample(
                        n=n_samples, with_replacement=False, seed=SAMPLING_SEED
                    )

                filtered_exp = df_exps.join(
                    sampled_exp,
                    on=["experiment_timestamp", "experiment_name"],
                    how="inner",
                )
                rebalance_list.append(filtered_exp)

    # Combine all filtered experiments
    if rebalance_list:
        df_rebalanced = pl.concat(rebalance_list, how="vertical")

        # After df_rebalanced is created
        print(
            "Final runs count by (is_synthetic, is_symmetric, num_agents, alpha, agent_prefix_type):"
        )

        group_keys_extended = group_keys + ["alpha", "agent_prefix_type"]
        report_detailed = (
            df_rebalanced.select(
                group_keys_extended + ["experiment_timestamp", "experiment_name"]
            )
            .unique()
            .group_by(group_keys_extended)
            .agg(pl.len().alias("num_runs"))
            .sort(group_keys_extended)
        )

        # We'll group by (is_synthetic, is_symmetric, num_agents) and print runs per alpha and prefix:
        last_group = None
        for row in report_detailed.iter_rows(named=True):
            group = (row["is_synthetic"], row["is_symmetric"], row["num_agents"])
            if group != last_group:
                if last_group is not None:
                    print()  # newline between groups
                print(
                    f"is_synthetic={group[0]}, is_symmetric={group[1]}, num_agents={group[2]}:"
                )
            print(
                f"  alpha={row['alpha']}, prefix={row['agent_prefix_type']} => runs: {row['num_runs']}"
            )
            last_group = group

        print(f"\nRebalanced dataset size: {df_rebalanced.height} rows")
        return df_rebalanced
    else:
        print("No experiments to rebalance (no complete groups).")
        return df_all


def concat_experiments_to_parquet(
    experiment_dirs: str,
    output_path: str,
    is_synthetic: bool = True,
    rebalance_exper: bool = True,
    min_rebalance: int = 7,
    keep_min_exactly: bool = True,
):
    experiment_dirs = Path(experiment_dirs)
    ouput_path = Path(output_path)
    all_dfs = []
    for n_agents in experiment_dirs.iterdir():
        if not n_agents.is_dir():
            continue
        for experiment_dir in n_agents.iterdir():
            if not experiment_dir.is_dir():
                continue

            # if metadata or results json is not there, skip
            if (
                not (experiment_dir / "metadata.json").exists()
                or not (experiment_dir / "results.json").exists()
            ):
                continue

            # Read metadata and results json as dicts
            metadata = pl.read_json(experiment_dir / "metadata.json").to_dicts()[0]
            results = pl.read_json(experiment_dir / "results.json").to_dicts()[0]

            # Extract metadata fields you want to add to each row
            experiment_timestamp = experiment_dir.name.split("_")[0]
            exper_name = metadata.get("name", "unknown_experiment")
            exper_n_agents = metadata.get("num_agents", 0)
            agents_prefixes = metadata.get("agents_prefixes", {})
            agents_prompts = metadata.get("agents_prompts", {})
            agents_memory_length = metadata.get("agents_memory_length", {})
            agents_models = metadata.get("agents_models", {})
            agent_environment_mapping = metadata.get("agent_environment_mapping", {})
            env_params = metadata.get("environment", {}).get("environment_params", {})
            num_rounds = metadata.get("num_rounds", None)
            start_time = metadata.get("start_time", None)
            end_time = metadata.get("end_time", None)

            exper_name = metadata.get("name", "unknown_experiment")
            exper_n_agents = metadata.get("num_agents", 0)
            agents = list(metadata.get("agents_types", {}).keys())

            env_params = metadata.get("environment", {}).get("environment_params", {})

            rows = []
            for agent_name in agents:
                agent_results = results.get(agent_name, {})
                agent_idx = agents.index(agent_name)

                for round_num, round_data in agent_results.items():
                    # round keys are strings; convert to int
                    round_num = int(round_num)

                    # flatten round_data dict keys
                    # e.g. chosen_price, profit, quantity, etc.
                    row = {
                        "experiment_timestamp": experiment_timestamp,
                        "experiment_name": exper_name,
                        "num_agents": exper_n_agents,
                        "agent": agent_name,
                        "round": round_num,
                        "start_time_exper": start_time,
                        "end_time_exper": end_time,
                    }

                    # Add environment params:
                    for param_key, param_val in env_params.items():
                        if isinstance(param_val, list):
                            # assign param value for this agent index if within bounds, else None
                            row[param_key] = (
                                param_val[agent_idx]
                                if agent_idx < len(param_val)
                                else None
                            )
                        else:
                            # scalar param: assign same for all agents
                            row[param_key] = param_val

                    # Add agent-specific metadata (optional, flatten dicts into JSON strings or leave as is)
                    row["agent_prefix"] = agents_prefixes.get(agent_name, None)
                    if row["agent_prefix"] is not None:
                        if row["agent_prefix"] == P1:
                            row["agent_prefix_type"] = "P1"
                        elif row["agent_prefix"] == P2:
                            row["agent_prefix_type"] = "P2"
                        else:
                            row["agent_prefix_type"] = "OTHER"
                    row["agent_prompt"] = agents_prompts.get(agent_name, None)
                    row["agent_memory_length"] = agents_memory_length.get(
                        agent_name, None
                    )
                    row["agent_model"] = agents_models.get(
                        agent_name, "mistral-large-2411"
                    )

                    # Flatten environment mapping for agent if exists (e.g. env_index, a, alpha, c)
                    env_map = agent_environment_mapping.get(agent_name, {})
                    for k, v in env_map.items():
                        row[f"agent_env_{k}"] = v

                    # Add the results data from round
                    for key, val in round_data.items():
                        row[key] = val

                    rows.append(row)

            # Create polars DataFrame for this experiment
            df_exp = pl.DataFrame(rows)
            # enforce schema
            df_exp = df_exp.with_columns(
                [
                    pl.col("experiment_timestamp").cast(pl.Utf8),
                    pl.col("experiment_name").cast(pl.Utf8),
                    pl.col("num_agents").cast(pl.Int64),
                    pl.col("agent").cast(pl.Utf8),
                    pl.col("round").cast(pl.Int64),
                    pl.col("start_time_exper").cast(pl.Utf8),
                    pl.col("end_time_exper").cast(pl.Utf8),
                    pl.col("a_0").cast(pl.Float64),
                    pl.col("a").cast(pl.Float64),
                    pl.col("mu").cast(pl.Float64),
                    pl.col("alpha").cast(pl.Float64),
                    pl.col("beta").cast(pl.Float64),
                    pl.col("sigma").cast(pl.Float64),
                    pl.col("c").cast(pl.Float64),
                    pl.col("group_idxs").cast(pl.Int64),
                    pl.col("monopoly_prices").cast(pl.Float64),
                    pl.col("monopoly_quantities").cast(pl.Float64),
                    pl.col("monopoly_profits").cast(pl.Float64),
                    pl.col("nash_prices").cast(pl.Float64),
                    pl.col("nash_quantities").cast(pl.Float64),
                    pl.col("nash_profits").cast(pl.Float64),
                    pl.col("agent_prefix").cast(pl.Utf8),
                    pl.col("agent_prefix_type").cast(pl.Utf8),
                    pl.col("agent_prompt").cast(pl.Utf8),
                    pl.col("agent_memory_length").cast(pl.Int64),
                    pl.col("agent_model").cast(pl.Utf8),
                    pl.col("agent_env_env_index").cast(pl.Int64),
                    pl.col("agent_env_a").cast(pl.Float64),
                    pl.col("agent_env_alpha").cast(pl.Float64),
                    pl.col("agent_env_c").cast(pl.Float64),
                    pl.col("observations").cast(pl.Utf8),
                    pl.col("plans").cast(pl.Utf8),
                    pl.col("insights").cast(pl.Utf8),
                    pl.col("chosen_price").cast(pl.Float64),
                    pl.col("marginal_cost").cast(pl.Float64),
                    pl.col("quantity").cast(pl.Float64),
                    pl.col("profit").cast(pl.Float64),
                    pl.col("market_data").cast(pl.Utf8),
                ]
            )
            # Check if the game is symmetric (a and c)
            df_symmetry = (
                df_exp.group_by(
                    ["experiment_timestamp", "experiment_name", "num_agents"]
                )
                .agg(
                    [
                        (pl.col("c").n_unique() == 1).alias("c_symmetric"),
                        (pl.col("a").n_unique() == 1).alias("a_symmetric"),
                    ]
                )
                .with_columns(
                    (pl.col("c_symmetric") & pl.col("a_symmetric")).alias(
                        "is_symmetric"
                    )
                )
                .select(
                    [
                        "experiment_timestamp",
                        "experiment_name",
                        "num_agents",
                        "is_symmetric",
                    ]
                )
            )

            # Step 2: Join back to your df_exp
            df_exp = df_exp.join(
                df_symmetry,
                on=["experiment_timestamp", "experiment_name", "num_agents"],
                how="left",
            )
            df_exp = df_exp.with_columns(pl.col("is_symmetric").cast(pl.Boolean))

            all_dfs.append(df_exp)

    # Concatenate all experiments
    if all_dfs:
        df_all = pl.concat(all_dfs, how="vertical")
        df_all = df_all.sort(["experiment_timestamp", "agent", "round"])
        if is_synthetic:
            # add the true value to new column in df_all as is_synthetic:True
            df_all = df_all.with_columns(
                pl.lit(True).alias("is_synthetic"),
            )
            # Unfinished experiments
            df_unfinished = (
                df_all.filter(pl.col("round").is_not_null())
                .group_by(
                    ["experiment_timestamp", "experiment_name", "num_agents", "agent"]
                )
                .agg(pl.len().alias("num_rounds"))
                .filter(pl.col("num_rounds") < 300)
                .select(
                    [
                        "num_agents",
                        "experiment_timestamp",
                        "experiment_name",
                        "num_rounds",
                    ]
                )
                .unique()
                .with_columns(
                    pl.format(
                        "{}_agents/{}_{}",
                        pl.col("num_agents"),
                        pl.col("experiment_timestamp"),
                        pl.col("experiment_name"),
                    ).alias("path")
                )
            )
            if df_unfinished.height > 0:
                print("Unfinished experiments found (SKIPPED):")
                for unfinished in df_unfinished["path"].to_list():
                    print(f" - {unfinished}")
            # Remove unfinished experiments with anti-join
            df_all = df_all.join(
                df_unfinished.select(
                    ["experiment_timestamp", "experiment_name", "num_agents"]
                ),
                on=["experiment_timestamp", "experiment_name", "num_agents"],
                how="anti",
            )
        if rebalance_exper:
            df_all = rebalance_experiments(
                df_all,
                min_required=min_rebalance,
                keep_exacly_min_required=keep_min_exactly,
            )

        df_all.write_parquet(output_path)
        print(
            f"Saved concatenated parquet file with {df_all.height} rows to {output_path}"
        )
    else:
        print("No data found to concatenate.")

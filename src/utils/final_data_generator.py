import json
import polars as pl
from pathlib import Path
from datetime import datetime

from data_loader import load_retail_data, load_tgp_data

PROJECT_ROOT = Path(__file__).parent.parent.parent


def generate_final_data(
    output_path: str = PROJECT_ROOT / "data/processed/",
) -> pl.DataFrame:
    """Generate final data for simulation by loading and preparing TGP and retail data.
    The final data will be saved as a Parquet file in the specified output path.
    Parameters
    ----------
    output_path : str, optional
        Path where the final data will be saved. Defaults to "../../data/processed/".
    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the prepared market data.
    """
    df_retail = load_retail_data()
    df_tgp = load_tgp_data()

    df = (
        df_retail
        # Rename columns to lowercase
        .rename(lambda x: x.lower())
        .rename({"brand_description": "brand"})
        .filter(
            (pl.col("publish_date") >= pl.date(2009, 4, 1))
            & (pl.col("publish_date") <= pl.date(2012, 5, 1))
        )
        .group_by(["publish_date", "brand"])
        .agg((pl.col("product_price").mean() / 100).round(2).alias("avg_price"))
        .with_columns(
            # rename Coles Express to Coles
            pl.when(pl.col("brand") == "Coles Express")
            .then(pl.lit("Coles"))
            .when(pl.col("brand") == "Caltex Woolworths")
            .then(pl.lit("Woolworths"))
            .otherwise(pl.col("brand"))
            .alias("brand")
        )
        .filter(pl.col("brand").is_in(["Coles", "Woolworths", "Caltex", "BP"]))
        .sort(["publish_date", "brand"])
        .join(df_tgp, left_on="publish_date", right_on="date", how="left")
        .with_columns(
            (pl.col("tgpmin") / 100).round(2).alias("tgpmin"),
        )
    )
    # create the path if it does not exist
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df.write_parquet(output_path / "final_data.parquet", compression="snappy")
    return df


def generate_data_to_inject_as_history(
    df: pl.DataFrame,
    memory_length: int,
    start_date: str,
    output_path: str = PROJECT_ROOT / "data/processed/",
) -> dict:
    required_cols = {"publish_date", "brand", "avg_price", "tgpmin"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain: {required_cols}")

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Filter rows strictly before the start_date
    df_filtered = df.filter(pl.col("publish_date") < start_date_obj).sort(
        "publish_date"
    )

    initial_real_data = {}
    brands = df_filtered["brand"].unique().to_list()

    for brand in brands:
        brand_df = df_filtered.filter(pl.col("brand") == brand).sort("publish_date")

        if len(brand_df) < memory_length:
            raise ValueError(
                f"Not enough data for brand '{brand}' before {start_date} to fill {memory_length} rounds."
            )

        last_entries = brand_df.tail(memory_length)
        prices = last_entries["avg_price"].to_list()
        costs = last_entries["tgpmin"].to_list()

        initial_real_data[brand] = [
            {
                "round": -i,
                "chosen_price": round(float(price), 2),  # assuming prices are in cents
                "marginal_cost": round(float(cost), 2),
            }
            for i, (price, cost) in enumerate(zip(prices[::-1], costs[::-1]), start=0)
        ]

    # save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "initial_real_data_to_inject_as_history.json", "w") as f:
        json.dump(initial_real_data, f, indent=4)

    return


def generate_individual_series(
    df: pl.DataFrame,
    start_date: str,
    output_path: str = PROJECT_ROOT / "data/processed/",
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()

    df_tgp_final = (
        df.filter(pl.col("publish_date") >= start_date_obj)
        .group_by("publish_date")
        .agg(pl.col("tgpmin").first().alias("tgpmin"))
        .sort("publish_date")
    )
    print(df_tgp_final.head())
    df_tgp_final.write_parquet(
        output_path / "marginal_costs_tgp.parquet", compression="snappy"
    )

    for brand in ["BP", "Caltex", "Woolworths", "Coles"]:
        df_brand = df.filter(
            pl.col("brand") == brand, pl.col("publish_date") >= start_date_obj
        ).select(["publish_date", "avg_price"])
        df_brand.write_parquet(
            output_path / f"{brand.lower()}_prices.parquet", compression="snappy"
        )
    print(df_brand.head())
    return


if __name__ == "__main__":
    START_DATE = "2009-08-01"
    df = generate_final_data()
    generate_data_to_inject_as_history(df, memory_length=100, start_date=START_DATE)
    generate_individual_series(df, start_date=START_DATE)
    print("Final data generated successfully.")

# src/data_processor.py
"""
Efficient data processing for fuel market analysis using Polars.
Handles TGP data and retail price data loading and preprocessing.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl


class FuelMarketDataProcessor:
    """
    Efficient processor for fuel market data including TGP and retail prices.
    """

    def __init__(self, base_data_path: str = "data/"):
        self.base_data_path = Path(base_data_path)
        self.tgp_path = self.base_data_path / "tgp" / "tgpmin.csv"
        self.prices_path = self.base_data_path / "retail"

    def load_tgp_data(self) -> pl.DataFrame:
        """
        Load Terminal Gate Price (TGP) data efficiently.

        Returns:
            DataFrame with columns: [date, tgpmin]
        """
        try:
            df = pl.read_csv(
                self.tgp_path,
                try_parse_dates=True,
                dtypes={"date": pl.Date, "tgpmin": pl.Float64},
            )

            # Ensure proper date parsing
            if df.schema["date"] != pl.Date:
                df = df.with_columns(pl.col("date").str.to_date())

            # Sort by date for efficient joins
            df = df.sort("date")

            print(
                f"Loaded TGP data: {len(df)} records from {df['date'].min()} to {df['date'].max()}"
            )
            return df

        except Exception as e:
            print(f"Error loading TGP data: {e}")
            return pl.DataFrame(schema={"date": pl.Date, "tgpmin": pl.Float64})

    def load_retail_prices(self, limit_files: Optional[int] = None) -> pl.DataFrame:
        """
        Load and combine retail price data from monthly CSV files.

        Args:
            limit_files: If specified, only load this many files (for testing)

        Returns:
            Combined DataFrame with all retail price data
        """
        price_files = sorted(self.prices_path.glob("FuelWatchRetail-*.csv"))

        if limit_files:
            price_files = price_files[:limit_files]

        if not price_files:
            print(f"No price files found in {self.prices_path}")
            return pl.DataFrame()

        print(f"Loading {len(price_files)} price files...")

        # Define schema for consistent loading
        schema = {
            "PUBLISH_DATE": pl.Utf8,
            "TRADING_NAME": pl.Utf8,
            "BRAND_DESCRIPTION": pl.Utf8,
            "PRODUCT_DESCRIPTION": pl.Utf8,
            "PRODUCT_PRICE": pl.Float64,
            "ADDRESS": pl.Utf8,
            "LOCATION": pl.Utf8,
            "POSTCODE": pl.Utf8,
        }

        # Load all files and concatenate
        dfs = []
        for file in price_files:
            try:
                df = pl.read_csv(file, schema=schema, ignore_errors=True)

                # Extract date from filename if PUBLISH_DATE is problematic
                date_match = re.search(r"(\d{2})-(\d{4})", file.name)
                if date_match:
                    month, year = date_match.groups()
                    file_date = f"{year}-{month.zfill(2)}"
                    df = df.with_columns(pl.lit(file_date).alias("file_date"))

                dfs.append(df)

            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                continue

        if not dfs:
            return pl.DataFrame()

        # Concatenate all DataFrames
        combined_df = pl.concat(dfs, how="diagonal")

        # Clean and standardize data
        combined_df = self._clean_retail_data(combined_df)

        print(f"Loaded retail data: {len(combined_df)} records")
        return combined_df

    def _clean_retail_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and standardize retail price data."""

        # Convert PUBLISH_DATE to proper date
        df = df.with_columns(
            [
                pl.col("PUBLISH_DATE")
                .str.to_date(format="%d/%m/%Y", strict=False)
                .alias("publish_date"),
                pl.col("PRODUCT_PRICE").cast(pl.Float64, strict=False),
                pl.col("POSTCODE").cast(pl.Utf8, strict=False),
            ]
        )

        # Filter for unleaded petrol products (ULP)
        ulp_patterns = ["ULP"]
        ulp_filter = pl.fold(
            False,
            lambda acc, pattern: acc
            | pl.col("PRODUCT_DESCRIPTION").str.contains(pattern, literal=False),
            ulp_patterns,
        )

        df = df.filter(ulp_filter)

        # Remove invalid prices and dates
        df = df.filter(
            (pl.col("PRODUCT_PRICE") > 0)
            & (pl.col("PRODUCT_PRICE") < 300)  # Reasonable price range (cents)
            & (pl.col("publish_date").is_not_null())
        )

        # Standardize brand names
        df = df.with_columns(
            pl.col("BRAND_DESCRIPTION")
            .str.to_uppercase()
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("brand_clean")
        )

        return df.sort("publish_date")

    def get_major_brands(
        self, df: pl.DataFrame, min_observations: int = 1000
    ) -> List[str]:
        """
        Identify major fuel brands based on number of observations.

        Args:
            df: Retail price DataFrame
            min_observations: Minimum number of observations to be considered major

        Returns:
            List of major brand names
        """
        brand_counts = (
            df.group_by("brand_clean")
            .agg(pl.count().alias("count"))
            .filter(pl.col("count") >= min_observations)
            .sort("count", descending=True)
        )

        return brand_counts["brand_clean"].to_list()

    def create_daily_market_data(
        self,
        tgp_df: pl.DataFrame,
        retail_df: pl.DataFrame,
        brands: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Create daily aggregated market data combining TGP and retail prices.

        Args:
            tgp_df: TGP DataFrame
            retail_df: Retail price DataFrame
            brands: List of brands to include (if None, uses major brands)

        Returns:
            Daily aggregated DataFrame
        """
        if brands is None:
            brands = self.get_major_brands(retail_df)[:4]  # Top 4 brands

        print(f"Using brands: {brands}")

        # Filter retail data for selected brands
        retail_filtered = retail_df.filter(pl.col("brand_clean").is_in(brands))

        # Calculate daily average prices by brand
        daily_retail = (
            retail_filtered.group_by(["publish_date", "brand_clean"])
            .agg(
                [
                    pl.mean("PRODUCT_PRICE").alias("avg_price"),
                    pl.count().alias("station_count"),
                    pl.std("PRODUCT_PRICE").alias("price_std"),
                ]
            )
            .sort(["publish_date", "brand_clean"])
        )

        # Pivot to have brands as columns
        daily_pivot = daily_retail.pivot(
            index="publish_date", columns="brand_clean", values="avg_price"
        )

        # Join with TGP data
        combined = tgp_df.join(
            daily_pivot, left_on="date", right_on="publish_date", how="inner"
        )

        return combined.sort("date")

    def prepare_simulation_data(
        self, daily_data: pl.DataFrame, window_size: int = 30
    ) -> pl.DataFrame:
        """
        Prepare data for LLM agent simulation by creating rolling windows and features.

        Args:
            daily_data: Daily market data
            window_size: Size of rolling window for historical data

        Returns:
            DataFrame prepared for simulation
        """
        # Calculate margins (retail price - TGP)
        brand_cols = [
            col for col in daily_data.columns if col not in ["date", "tgpmin"]
        ]

        margin_exprs = [
            (pl.col(brand) - pl.col("tgpmin")).alias(f"{brand}_margin")
            for brand in brand_cols
        ]

        df = daily_data.with_columns(margin_exprs)

        # Add rolling statistics
        rolling_exprs = []
        for brand in brand_cols:
            rolling_exprs.extend(
                [
                    pl.col(brand)
                    .rolling_mean(window_size)
                    .alias(f"{brand}_ma{window_size}"),
                    pl.col(brand)
                    .rolling_std(window_size)
                    .alias(f"{brand}_std{window_size}"),
                    pl.col(f"{brand}_margin")
                    .rolling_mean(window_size)
                    .alias(f"{brand}_margin_ma{window_size}"),
                ]
            )

        df = df.with_columns(rolling_exprs)

        # Add lagged variables
        lag_exprs = []
        for lag in [1, 7, 14]:  # 1 day, 1 week, 2 weeks
            for brand in brand_cols:
                lag_exprs.append(pl.col(brand).shift(lag).alias(f"{brand}_lag{lag}"))

        df = df.with_columns(lag_exprs)

        return (
            df.drop_nulls()
        )  # Remove rows with null values from rolling/lag operations


class SimulationDataBuilder:
    """Build data structures specifically for market simulation experiments."""

    def __init__(self, processor: FuelMarketDataProcessor):
        self.processor = processor

    def create_experiment_dataset(
        self, start_date: str, end_date: str, brands: List[str]
    ) -> Dict[str, pl.DataFrame]:
        """
        Create a complete dataset for simulation experiments.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            brands: List of brand names to include

        Returns:
            Dictionary with experiment data components
        """
        # Load base data
        tgp_df = self.processor.load_tgp_data()
        retail_df = self.processor.load_retail_prices()

        # Filter by date range
        date_filter = (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        tgp_filtered = tgp_df.filter(date_filter)

        retail_date_filter = (pl.col("publish_date") >= start_date) & (
            pl.col("publish_date") <= end_date
        )
        retail_filtered = retail_df.filter(retail_date_filter)

        # Create daily market data
        daily_data = self.processor.create_daily_market_data(
            tgp_filtered, retail_filtered, brands
        )

        # Prepare simulation data
        simulation_data = self.processor.prepare_simulation_data(daily_data)

        return {
            "tgp_data": tgp_filtered,
            "retail_data": retail_filtered,
            "daily_market": daily_data,
            "simulation_ready": simulation_data,
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "brands": brands,
                "total_days": len(daily_data),
                "simulation_days": len(simulation_data),
            },
        }

    def export_for_llm_simulation(
        self, experiment_data: Dict, output_path: str
    ) -> None:
        """
        Export processed data in formats suitable for LLM simulation.

        Args:
            experiment_data: Output from create_experiment_dataset
            output_path: Base path for output files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export simulation-ready data
        sim_data = experiment_data["simulation_ready"]
        sim_data.write_parquet(output_path / "simulation_data.parquet")
        sim_data.write_csv(output_path / "simulation_data.csv")

        # Export daily market data
        experiment_data["daily_market"].write_parquet(
            output_path / "daily_market.parquet"
        )

        # Export metadata
        import json

        with open(output_path / "experiment_metadata.json", "w") as f:
            json.dump(experiment_data["metadata"], f, indent=2, default=str)

        print(f"Exported experiment data to {output_path}")

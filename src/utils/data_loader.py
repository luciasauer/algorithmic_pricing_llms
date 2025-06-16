# src/utils/data_loader.py

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


def load_tgp_data(
    file_path: str = "../../data/113176-V1/data/TGP/tgpmin.csv",
    start_date: datetime.date = None,
    end_date: datetime.date = None,
) -> pl.DataFrame:
    """
    Load and prepare TGP (Terminal Gate Price) data from a CSV file.

    Parameters
    ----------
    file_path : str, optional
        Path to the TGP CSV file. Defaults to "../../data/113176-V1/data/TGP/tgpmin.csv".
    start_date : datetime.date, optional
        Start date for filtering the data. If None, no lower bound is applied.
    end_date : datetime.date, optional
        End date for filtering the data. If None, no upper bound is applied.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the filtered and sorted TGP data with the 'date' column parsed as dates.
    """
    # Create filter conditions
    start_condition = pl.col("date") >= start_date if start_date else pl.lit(True)
    end_condition = pl.col("date") <= end_date if end_date else pl.lit(True)

    return (
        pl.read_csv(Path(file_path))
        .with_columns(pl.col("date").str.to_date(format="%d/%m/%y"))
        .filter(start_condition & end_condition)
        .sort("date")
    )


def get_tgp_for_period(tgp_data: pl.DataFrame, start_date: str, period: int) -> float:
    """
    Perform nearest neighbor lookup to get TGP for a specific period (0-indexed).

    Parameters
    ----------
    tgp_data : pl.DataFrame
        DataFrame containing TGP data with a 'date' column.
    start_date : str
        The start date in "YYYY-MM-DD" format.
    period : int
        The period offset (in days) from the start_date.

    Returns
    -------
    float
        The TGP value nearest to the target date.
    """
    target_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=period)

    return float(
        tgp_data.select(
            pl.col("tgpmin").get((pl.col("date") - target_date).abs().arg_min())
        ).item()
    )


def load_retail_data(
    file_path: str = "../../data/113176-V1/data/Prices/",
    file: str = "FuelWatchRetail-*.csv",
    start_date: datetime.date = None,
    end_date: datetime.date = None,
) -> pl.DataFrame:
    """
    Load and concatenate retail price data from one or multiple CSV files.

    Parameters
    ----------
    file_path : str, optional
        Directory containing the retail price CSV files. Defaults to "../../data/113176-V1/data/Prices/".
    file : str, optional
        Glob pattern for the retail price files. Defaults to "FuelWatchRetail-*.csv".
    start_date : datetime.date, optional
        Start date for filtering the data. If None, no lower bound is applied.
    end_date : datetime.date, optional
        End date for filtering the data. If None, no upper bound is applied.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the concatenated and filtered retail price data, sorted by 'PUBLISH_DATE'.
    """
    queries = []
    file_path = Path(file_path)

    start_condition = (
        pl.col("PUBLISH_DATE") >= start_date if start_date else pl.lit(True)
    )
    end_condition = pl.col("PUBLISH_DATE") <= end_date if end_date else pl.lit(True)

    for file in file_path.glob(file):
        q = pl.scan_csv(file, try_parse_dates=True).filter(
            start_condition & end_condition
        )
        queries.append(q)

    dataframes = pl.collect_all(queries)
    return pl.concat(dataframes, how="diagonal_relaxed").sort("PUBLISH_DATE")


def get_retail_price_for_period(
    retail_data: pl.DataFrame, start_date: str, period: int
) -> pl.DataFrame:
    """
    Get all rows from the day nearest to the target date in the retail data.

    Parameters
    ----------
    retail_data : pl.DataFrame
        DataFrame containing retail price data with a 'PUBLISH_DATE' column.
    start_date : str
        The start date in "YYYY-MM-DD" format.
    period : int
        The period offset (in days) from the start_date.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing all rows from the day nearest to the target date.
    """
    target_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=period)

    return retail_data.filter(
        (pl.col("PUBLISH_DATE") - target_date).abs()
        == (pl.col("PUBLISH_DATE") - target_date).abs().min()
    )

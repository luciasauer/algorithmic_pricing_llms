#src/utils/clean_data.py
import os
import sys
import polars as pl
from logger import setup_logger

def read_and_concat_files(directory:str) -> pl.DataFrame:
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = []

    for file in files:
        file_path = os.path.join(directory, file)
        df = pl.read_csv(file_path)[:,:-1]
        if df.shape[0] <= 0:
            continue
        dataframes.append(df)

    df = pl.concat(dataframes)
    df.columns = [col.lower() for col in df.columns]
    return df

def cast_columns_tgp(df: pl.DataFrame) -> pl.DataFrame:
    # Cast data types
    df = (df
        .with_columns(
            pl.col('publish_date').str.strptime(pl.Date),
        )
        .sort('publish_date', descending=False)
    )
    return df

def cast_columns_retail(df: pl.DataFrame) -> pl.DataFrame:
    # Cast data types
    df = (df
        .with_columns(
            pl.col('publish_date').str.strptime(pl.Date),
            pl.col('postcode').cast(pl.Int64),
        )
        .sort('publish_date', descending=False)
    )
    return df

def save_to_parquet(df: pl.DataFrame, output_path: str) -> None:
    df.write_parquet(output_path, compression='snappy')
    return


def preprocess_tgp_data(data_path: str="../../data/raw/tgp", 
                        tgpmin_path:str="../../data/tgpmin.csv", 
                        output_path:str="../../data/preprocessed/tgp.parquet", 
                        log_file: str = '../../data/logs/tgp.log') -> None:

    df = read_and_concat_files(data_path)
    logger = setup_logger(name="tgp_logger", log_file=log_file)
    df = cast_columns_tgp(df)
    df = df.filter(
            (pl.col("product") == "ULP")  # Only unleaded fuels
            & (pl.col("postcode") < 6200)  # Only Perth
            & (pl.col("address") != "XXXXX")  # Must have an address
    )
    
    logger.info("Preprocessed TGP Data", df.head(20))
    logger.info(f"Original paper: 6 terminal gates")
    for i in df['terminal_gate'].unique():
        logger.info(f"Terminal Gate: {i}")

    #get lowest price by publish_date
    df = df.group_by('publish_date').agg(
        pl.col('product_price').min().alias('lowest_price')
    )

    df_tgpmin = pl.read_csv(tgpmin_path)
    df_tgpmin = (
        df_tgpmin
        .with_columns(
            pl.col('date').str.strptime(pl.Date, format='%d/%m/%y'),
            pl.col('tgpmin').cast(pl.Float64)
        )
        .sort('date', descending=False)
    )
    logger.info("TGP MIN Original Data\n", df_tgpmin.head(20))

    df_combined = df.join(
        df_tgpmin,
        left_on='publish_date',
        right_on='date',
        how='inner'
    )
    #check if there is difference between lowest_price and tgpmin
    df_combined = df_combined.with_columns(
        (pl.col('lowest_price') == pl.col('tgpmin')).alias('equal')
    )
    logger.info("TGP MIN Combined Data\n", df_combined.head(20))
    logger.info("TGP MIN Combined Data Equal Counts\n", df_combined['equal'].value_counts())
    df = df.rename({'lowest_price': 'tgpmin', 'publish_date': 'date'})
    logger.info("Final Data\n", df.head(20))

    df.write_parquet(output_path, compression='snappy')
    logger.info(f"Data saved to {output_path}")
    return

def preprocess_retail_data(data_path: str="../../data/raw/retail",
                            output_path:str="../../data/preprocessed/data_avg_brand.parquet",
                            log_file: str = '../../data/logs/retail.log') -> None:
        logger = setup_logger(name="retail_logger", log_file=log_file)
        df = read_and_concat_files(data_path)
        df = cast_columns_retail(df)
        df = df.filter(
                (pl.col("product_description") == "ULP")    # Only unleaded fuels
                & (pl.col("postcode") < 6200)               # Only Perth
                & (pl.col("address") != "XXXXX")            # Must have an address
                & (pl.col("brand_description").is_in(["BP", "Caltex", "Caltex Woolworths", "Coles Express", "Gull"]))  #Only selected brands
        ).sort('publish_date', descending=False)
        logger.info("Preprocessed Retail Data\n", print(df.head(20)))

        df = (
                df
                .group_by(['publish_date', 'brand_description'])
                .agg(
                    pl.col('product_price').mean().round(2).alias('avg_price')
                )
                .sort(['publish_date', 'brand_description'], descending=[False, False])
                .with_columns(
                    #rename "Caltex Woolworths" to Woolworths and "Coles Express to Coles
                    pl.when(pl.col('brand_description') == 'Caltex Woolworths')
                      .then(pl.lit('Woolworths'))
                      .when(pl.col('brand_description') == 'Coles Express')
                      .then(pl.lit('Coles'))
                      .otherwise(pl.col('brand_description'))
                      .alias('brand_description')
                )
            )
        
        logger.info("Retail Data Grouped by Date and Brand\n", print(df.head(20)))
        df.write_parquet(output_path, compression='snappy')
        logger.info(f"Data saved to {output_path}")
        return


if __name__ == "__main__":
    preprocess_tgp_data()
    preprocess_retail_data()
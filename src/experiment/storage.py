import json
import time
import logging
import polars as pl
from pathlib import Path

class StorageManager:
    def __init__(self, n_agents: int, base_dir="experiments_runs", logger: logging.Logger = None):
        self.base_dir = Path(base_dir)
        self.n_agents = n_agents
        self.experiment_path = None
        self.logger = logger or logging.getLogger("experiment_logger")

    def create_experiment_dir(self, experiment_name: str):
        unix_time = int(time.time())
        folder_name = f"{unix_time}_{experiment_name}"
        self.experiment_path = self.base_dir / f"{self.n_agents}_agents" / folder_name
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Experiment directory created: {self.experiment_path}")
        return self.experiment_path

    def save_metadata(self, metadata: dict):
        with open(self.experiment_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"✅ Metadata saved")
    
    def load_metadata(self):
        try:
            with open(self.experiment_path / "metadata.json", "r") as f:
                metadata = json.load(f)
            return metadata
        except FileNotFoundError:
            self.logger.error("❌ Metadata file not found.")
            raise FileNotFoundError("Metadata file not found. Please ensure the experiment has been set up correctly.")

    def save_round_data(self, data: dict):
        try:
            with open(self.experiment_path / f"results.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Failed to save round data: {e}")
    
    def get_log_file_path(self):
        return self.experiment_path / "log.txt"
    
    def save_environment_parquet(self, df: pl.DataFrame):
        try:
            path = self.experiment_path / "environment_history.parquet"
            df.write_parquet(path)
        except Exception as e:
            self.logger.error(f"❌ Failed to save environment data in parquet: {e}")

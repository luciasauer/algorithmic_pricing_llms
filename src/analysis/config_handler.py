# src/config.py
"""
Configuration management for market simulation parameters.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimulationConfig:
    """Handle simulation configuration and parameter extraction."""

    def __init__(self, metadata_path: Optional[str] = None):
        self.metadata_path = metadata_path
        self.default_params = {
            "a_0": 0.0,
            "a": [2.0],
            "alpha": [1.0],
            "c": [1.0],
            "mu": 0.25,
            "beta": 100,
            "sigma": 0.0,
            "group_idxs": [1],
        }

    def load_environment_params(self, json_file_path: str) -> Dict[str, Any]:
        """
        Extract specific environment parameters and agent names from JSON file.

        Args:
            json_file_path: Path to the JSON metadata file

        Returns:
            Dictionary with required environment parameters and agents list
        """
        # Define the keys we want to extract from environment_params
        required_keys = {"a_0", "a", "mu", "alpha", "beta", "sigma", "c", "group_idxs"}

        try:
            with open(json_file_path, "r") as file:
                data = json.load(file)

            # Get environment parameters
            environment_params = data.get("environment", {}).get(
                "environment_params", {}
            )
            if not environment_params:
                print("Warning: 'environment_params' not found or empty in JSON.")
                return {}

            # Extract only the required parameters
            result = {}
            for key in required_keys:
                if key in environment_params:
                    result[key] = environment_params[key]
                else:
                    print(
                        f"Warning: Required parameter '{key}' not found in environment_params"
                    )

            # Get agent names from agent_environment_mapping
            agent_mapping = data.get("agent_environment_mapping", {})
            if agent_mapping:
                result["agents"] = list(agent_mapping.keys())
            else:
                print(
                    "Warning: 'agent_environment_mapping' not found, no agents extracted"
                )

            return result

        except FileNotFoundError:
            print(f"Error: File '{json_file_path}' not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{json_file_path}'.")
            return {}
        except KeyError:
            print("Error: 'environment' key not found in JSON.")
            return {}

    def prepare_simulation_params(self, metadata_path: str) -> Dict[str, Any]:
        """
        Prepare complete simulation parameters by combining metadata with defaults.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            Complete parameters dictionary ready for pricing calculations
        """
        # Load parameters from metadata
        params = self.load_environment_params(metadata_path)

        if not params:
            print("Warning: No parameters loaded from metadata, using defaults")
            return self.default_params.copy()

        # Fill in missing parameters with defaults
        for key, default_value in self.default_params.items():
            if key not in params:
                params[key] = default_value

        # Convert lists to tuples for hashability (needed for caching)
        for key, value in params.items():
            if isinstance(value, list):
                params[key] = tuple(value)

        return params

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate that parameters are consistent and complete.

        Args:
            params: Parameters dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = [
            "a_0",
            "a",
            "alpha",
            "c",
            "mu",
            "beta",
            "sigma",
            "group_idxs",
        ]

        # Check all required keys exist
        for key in required_keys:
            if key not in params:
                print(f"Missing required parameter: {key}")
                return False

        # Check that list parameters have consistent lengths
        list_params = ["a", "alpha", "c", "group_idxs"]
        if list_params:
            expected_length = (
                len(params["a"]) if isinstance(params["a"], (list, tuple)) else 1
            )
            for param in list_params:
                if (
                    isinstance(params[param], (list, tuple))
                    and len(params[param]) != expected_length
                ):
                    print(
                        f"Parameter {param} has length {len(params[param])}, expected {expected_length}"
                    )
                    return False

        # Check parameter constraints
        if params["mu"] <= 0:
            print("Parameter mu must be positive")
            return False

        if params["beta"] <= 0:
            print("Parameter beta must be positive")
            return False

        return True

    def get_market_params_for_pricing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the parameters needed for pricing calculations.

        Args:
            params: Full parameters dictionary

        Returns:
            Dictionary with only pricing-relevant parameters
        """
        pricing_keys = [
            "a_0",
            "a",
            "alpha",
            "c",
            "mu",
            "beta",
            "sigma",
            "group_idxs",
        ]
        return {k: v for k, v in params.items() if k in pricing_keys}


class DataConfig:
    """Configuration for data paths and loading parameters."""

    def __init__(self, base_data_path: str = "data/"):
        self.base_data_path = Path(base_data_path)
        self.tgp_path = self.base_data_path / "tgp" / "tgpmin.csv"
        self.prices_path = self.base_data_path / "retail"

    def get_tgp_path(self) -> Path:
        """Get path to TGP data file."""
        return self.tgp_path

    def get_price_files(self) -> List[Path]:
        """Get list of all price CSV files."""
        if not self.prices_path.exists():
            return []
        return sorted(self.prices_path.glob("FuelWatchRetail-*.csv"))

    def validate_data_paths(self) -> bool:
        """Validate that all expected data paths exist."""
        if not self.tgp_path.exists():
            print(f"TGP data file not found: {self.tgp_path}")
            return False

        if not self.prices_path.exists():
            print(f"Prices directory not found: {self.prices_path}")
            return False

        price_files = self.get_price_files()
        if not price_files:
            print(f"No price files found in: {self.prices_path}")
            return False

        return True


# Example usage and configuration presets
class ConfigPresets:
    """Predefined configuration presets for common scenarios."""

    @staticmethod
    def get_default_market_config() -> Dict[str, Any]:
        """Get default market configuration matching Calvano et al. (2020)."""
        return {
            "a_0": 0.0,
            "a": (2.0, 2.0),  # Two agents with a_i = 2
            "alpha": (1.0, 1.0),
            "c": (1.0, 1.0),
            "mu": 0.25,  # Standard value from literature
            "beta": 100,
            "sigma": 0.0,
            "group_idxs": (1, 1),
        }

    @staticmethod
    def get_perth_fuel_config() -> Dict[str, Any]:
        """Get configuration adapted for Perth fuel market analysis."""
        return {
            "a_0": 0.0,
            "a": (2.0, 2.0, 2.0, 2.0),  # Four major fuel companies
            "alpha": (1.0, 1.0, 1.0, 1.0),
            "c": (1.0, 1.0, 1.0, 1.0),  # Will be updated with actual TGP data
            "mu": 0.25,  # Lower value for less substitution in fuel market
            "beta": 100,
            "sigma": 0.0,
            "group_idxs": (1, 1, 1, 1),
        }

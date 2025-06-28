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
            "a0": 0,
            "a": [2],
            "alpha": [1],
            "c": [1],
            "mu": 0.25,
            "multiplier": 100,
            "sigma": 0,
            "group_idxs": [1],
        }

    def load_agent_environment_mapping(self, json_file_path: str) -> Dict[str, Any]:
        """
        Extract and transpose agent_environment_mapping from JSON file.

        Args:
            json_file_path: Path to the JSON metadata file

        Returns:
            Dictionary with parameters as keys and lists of values
        """
        try:
            with open(json_file_path, "r") as file:
                data = json.load(file)

            agent_mapping = data.get("agent_environment_mapping", {})
            if not agent_mapping:
                return {}

            # Get parameter names from first agent
            first_agent = next(iter(agent_mapping.values()))
            param_names = list(first_agent.keys())

            # Create transposed dictionary
            result = {"agents": list(agent_mapping.keys())}
            for param in param_names:
                result[param] = [
                    agent_config[param] for agent_config in agent_mapping.values()
                ]

            return result

        except FileNotFoundError:
            print(f"Error: File '{json_file_path}' not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{json_file_path}'.")
            return {}
        except KeyError:
            print("Error: 'agent_environment_mapping' key not found in JSON.")
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
        params = self.load_agent_environment_mapping(metadata_path)

        if not params:
            print("Warning: No parameters loaded from metadata, using defaults")
            return self.default_params.copy()

        # Get number of agents
        num_agents = len(params.get("agents", []))

        # Fill in missing parameters with defaults
        for key, default_value in self.default_params.items():
            if key not in params:
                if isinstance(default_value, list) and len(default_value) == 1:
                    # Expand single-element lists to match number of agents
                    params[key] = [default_value[0]] * num_agents
                else:
                    params[key] = default_value

        # Convert lists to tuples for hashability (needed for caching)
        for key, value in params.items():
            if isinstance(value, list) and key != "agents":
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
            "a0",
            "a",
            "alpha",
            "c",
            "mu",
            "multiplier",
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
        if "agents" in params:
            expected_length = len(params["agents"])
            for param in list_params:
                if len(params[param]) != expected_length:
                    print(
                        f"Parameter {param} has length {len(params[param])}, expected {expected_length}"
                    )
                    return False

        # Check parameter constraints
        if params["mu"] <= 0:
            print("Parameter mu must be positive")
            return False

        if params["multiplier"] <= 0:
            print("Parameter multiplier must be positive")
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
            "a0",
            "a",
            "alpha",
            "c",
            "mu",
            "multiplier",
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
            "a0": 0,
            "a": (2, 2),  # Two agents with a_i = 2
            "alpha": (1, 1),
            "c": (1, 1),
            "mu": 0.25,  # Standard value from literature
            "multiplier": 100,
            "sigma": 0,
            "group_idxs": (1, 1),
            "agents": ["agent_0", "agent_1"],
        }

    @staticmethod
    def get_perth_fuel_config() -> Dict[str, Any]:
        """Get configuration adapted for Perth fuel market analysis."""
        return {
            "a0": 0,
            "a": (2, 2, 2, 2),  # Four major fuel companies
            "alpha": (1, 1, 1, 1),
            "c": (1, 1, 1, 1),  # Will be updated with actual TGP data
            "mu": 0.25,  # Lower value for less substitution in fuel market
            "multiplier": 100,
            "sigma": 0,
            "group_idxs": (1, 1, 1, 1),
            "agents": ["BP", "Caltex", "Woolworths", "Coles"],  # Major Perth retailers
        }

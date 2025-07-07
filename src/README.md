# Source Code Architecture

This directory contains the core implementation for algorithmic collusion experiments testing Folk Theorem predictions in multi-agent pricing games.

## Module Structure

### `agents/` - LLM Agent Implementation

- `LLMAgent` - Mistral API integration for strategic pricing decisions
- `Agent` - Abstract base class defining agent interface
- `FakeAgent` - Deterministic agent for testing and baselines

### `environment/` - Market Simulation

- `CalvanoDemandEnvironment` - Implements Calvano et al. (2020) demand specification
- Calculates quantities, profits, and theoretical benchmarks
- Supports both static and time-varying cost structures

### `experiment/` - Experiment Orchestration

- `Experiment` - Coordinates multi-agent pricing games
- `StorageManager` - Handles data persistence in Parquet format
- `RateLimiter` - Manages API rate limiting for cost control

### `prompts/` - Prompt Engineering

- `PromptManager` - Dynamic context injection and memory management
- Systematic P1/P2 prompt specifications for coordination testing
- Structured response validation using Pydantic models

### `analysis/` - Statistical Analysis

- `CollusionAnalysis` - Core regression models for Folk Theorem testing
- Bootstrap validation and robustness testing
- Focus on run-level equilibrium analysis (periods 251-300)

### `plotting/` - Visualization Tools

- Publication-ready figures with theoretical benchmark overlays
- Time series analysis and convergence visualization
- Animated price dynamics for presentation

### `utils/` - Utility Functions

- Logging configuration for experiment tracking
- Price convergence analysis and statistical helpers
- Data processing utilities

## Key Design Principles

**Rate Limiting**: Built-in API management prevents cost overruns
**Memory Architecture**: 100-period rolling windows for strategic learning
**Data Integrity**: Parquet storage with comprehensive metadata
**Reproducibility**: Structured logging and configuration management
**Modularity**: Clean separation between agents, environment, and analysis

## Usage Patterns

```python
from src.agents import LLMAgent
from src.environment import CalvanoDemandEnvironment
from src.experiment import Experiment

# Create environment
env = CalvanoDemandEnvironment("oligopoly", "2-agent market")

# Initialize agents
agents = [LLMAgent(f"Agent_{i}", api_key=api_key, ...) for i in range(2)]

# Run experiment
experiment = Experiment(agents, env, n_rounds=300)
results = experiment.run()
```

This architecture supports the systematic testing of Folk Theorem predictions through controlled LLM agent interactions in oligopoly markets.

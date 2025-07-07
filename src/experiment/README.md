# Experiment Module

This module contains the core experimental infrastructure for running multi-agent algorithmic pricing simulations. It orchestrates agent interactions, manages data collection, and handles the execution of repeated pricing games.

## Architecture

### Main Experiment Class (`experiment.py`)

The `Experiment` class serves as the central orchestrator for pricing simulations:

**Key Components:**

- **RateLimiter**: Manages API request frequency to prevent quota exhaustion
- **RateLimitedAgentWrapper**: Wraps agents with retry logic and rate limiting
- **Experiment**: Main class coordinating multi-round pricing games

**Core Features:**

- **Multi-agent Coordination**: Synchronous price elicitation from all agents
- **Rate Limiting**: Built-in protection against API overuse
- **Error Handling**: Robust retry mechanisms for API failures
- **Data Collection**: Comprehensive logging of all experimental data
- **Progress Monitoring**: Real-time tracking of experimental progress

### Storage Management (`storage.py`)

Handles data persistence and experimental record keeping:

**Features:**

- **Structured Storage**: Consistent data schema across experiments
- **Efficient Formats**: Parquet files for analytical workloads
- **Metadata Tracking**: Experimental parameters and configurations
- **Incremental Saves**: Periodic checkpointing during long experiments

### Resume Functionality (`resume.py`)

Supports experiment continuation and recovery:

**Capabilities:**

- **State Recovery**: Restore agent memory and experimental state
- **Partial Completion**: Continue from last completed round
- **Data Integrity**: Ensure consistency when resuming experiments
- **Configuration Validation**: Verify setup matches original experiment

## Experimental Design

### Multi-Round Pricing Games

The experiment framework implements repeated pricing interactions:

**Game Structure:**

- **Rounds**: Typically 300 periods for convergence analysis
- **Agents**: 2-5 competing LLM agents (testing Folk Theorem predictions)
- **Simultaneous Moves**: All agents choose prices simultaneously
- **Information Structure**: Agents observe past prices and profits
- **Memory**: 100-period rolling window of market history

**Economic Environment:**

- **Demand Model**: Nested logit specification (Calvano et al., 2020)
- **Market Parameters**: Configurable demand, costs, and substitutability
- **Benchmarks**: Nash equilibrium and monopoly pricing for comparison

### Rate Limiting Strategy

Critical for managing LLM API costs and reliability:

**Implementation:**

```python
class RateLimiter:
    def __init__(self, rate_limit_seconds: float = 1.0):
        self.rate_limit_seconds = rate_limit_seconds
        self.last_request_time = 0
        self.lock = asyncio.Lock()
```

**Benefits:**

- **Cost Control**: Prevents excessive API usage
- **Reliability**: Reduces API timeout and quota errors
- **Scalability**: Supports multiple concurrent experiments
- **Flexibility**: Configurable delay between requests

### Error Handling and Retries

Robust handling of LLM API failures:

**Retry Strategy:**

- **Maximum Attempts**: 20 retries per failed request
- **Exponential Backoff**: Increasing delays between retries
- **Error Logging**: Comprehensive tracking of failure modes
- **Graceful Degradation**: Continue experiment despite individual failures

**Common Failure Modes:**

- API rate limiting or quota exhaustion
- Network connectivity issues
- Malformed JSON responses from LLMs
- Model availability or service outages

## Usage Patterns

### Basic Experiment Setup

```python
from src.experiment.experiment import Experiment
from src.agents.LLM_agent import LLMAgent
from src.environment.calvano import CalvanoDemandEnvironment

# Create agents
agents = [
    LLMAgent("Firm A", prefix="P1", **agent_config),
    LLMAgent("Firm B", prefix="P1", **agent_config),
]

# Setup environment
env = CalvanoDemandEnvironment("Duopoly", "2-agent market")
env.configure_parameters(a=[2.0, 2.0], alpha=[1.0, 1.0], c=[1.0, 1.0])

# Run experiment
experiment = Experiment(
    agents=agents,
    environment=env,
    n_rounds=300,
    rate_limit_seconds=1.0
)

results = await experiment.run()
```

### Multi-Configuration Experiments

Running systematic parameter sweeps:

```python
# Parameter configurations to test
configs = [
    {"n_agents": 2, "alpha": 1.0, "prefix": "P1"},
    {"n_agents": 3, "alpha": 1.0, "prefix": "P1"},
    {"n_agents": 4, "alpha": 1.0, "prefix": "P1"},
    {"n_agents": 5, "alpha": 1.0, "prefix": "P1"},
]

# Run all configurations
for config in configs:
    experiment = setup_experiment(**config)
    results = await experiment.run()
    save_results(results, config)
```

### Experiment Resume

Continuing interrupted experiments:

```python
from src.experiment.resume import resume_experiment

# Resume from last checkpoint
experiment = resume_experiment(
    experiment_id="duopoly_run_1",
    checkpoint_path="data/checkpoints/"
)

# Continue from where we left off
remaining_results = await experiment.continue_run()
```

## Data Collection

### Comprehensive Logging

The experiment framework captures detailed data:

**Round-Level Data:**

- Agent pricing decisions and reasoning
- Market quantities and profits
- Response times and API performance
- Economic benchmarks (Nash, monopoly)

**Experiment-Level Data:**

- Configuration parameters
- Agent specifications and prompts
- Environment setup and market structure
- Performance metrics and error rates

**Data Schema:**

```python
round_data = {
    "experiment_id": str,
    "run_id": str, 
    "round": int,
    "agent_name": str,
    "price": float,
    "profit": float,
    "quantity": float,
    "reasoning": str,
    "response_time": float,
    "timestamp": datetime,
    # ... additional fields
}
```

### Storage Format

Efficient data storage for analysis:

**Parquet Files:**

- **High Performance**: Optimized for analytical queries
- **Compression**: Significant space savings
- **Schema Evolution**: Support for adding new fields
- **Type Safety**: Preserve data types across save/load

**Directory Structure:**

```
data/
├── results/
│   ├── all_experiments.parquet     # Combined results
│   ├── duopoly/                    # Experiment-specific data
│   ├── oligopoly_3/
│   └── oligopoly_5/
├── checkpoints/                    # Resume state
└── logs/                          # Detailed experiment logs
```

## Performance Considerations

### Memory Management

Efficient handling of large-scale experiments:

**Strategies:**

- **Streaming Data**: Process results without loading entire dataset
- **Batch Processing**: Handle multiple runs efficiently
- **Memory Profiling**: Monitor and optimize memory usage
- **Garbage Collection**: Explicit cleanup of large objects

### Scalability

Support for large-scale experimental studies:

**Parallel Execution:**

- **Multiple Experiments**: Run different configurations simultaneously
- **Agent Parallelization**: Concurrent price elicitation (with rate limiting)
- **Resource Management**: Efficient CPU and memory utilization

**Monitoring:**

- **Progress Tracking**: Real-time updates on experimental progress
- **Performance Metrics**: API response times and success rates
- **Resource Usage**: Memory and CPU consumption monitoring

## Integration with Analysis

### Seamless Data Pipeline

Direct integration with analysis modules:

```python
# Experiment execution
results = await experiment.run()

# Immediate analysis
from src.analysis.data_processor import DataProcessor
processor = DataProcessor(results)
analysis_ready_data = processor.prepare_for_analysis()

# Statistical testing
from src.analysis.group_size import CollusionAnalysis
analyzer = CollusionAnalysis()
folk_theorem_results = analyzer.test_coordination_breakdown(analysis_ready_data)
```

### Real-time Analysis

Live monitoring during experiments:

```python
# Progress callback for real-time analysis
def analyze_progress(round_data):
    if round_data["round"] % 50 == 0:
        current_prices = extract_prices(round_data)
        convergence_check = test_stationarity(current_prices)
        logger.info(f"Round {round_data['round']}: Convergence = {convergence_check}")

experiment.add_progress_callback(analyze_progress)
```

## Configuration Management

### Reproducible Experiments

Ensuring experimental reproducibility:

**Configuration Tracking:**

- **Parameter Logging**: Complete record of all experimental settings
- **Random Seeds**: Controlled randomness for reproducibility
- **Version Control**: Track code versions for each experiment
- **Environment Capture**: Record system and dependency versions

**Configuration Files:**

```python
experiment_config = {
    "experiment_name": "folk_theorem_test",
    "n_rounds": 300,
    "n_runs": 7,
    "agent_config": {
        "model_name": "mistral-large-2411",
        "memory_length": 100,
        "prompt_prefix": "P1"
    },
    "environment_config": {
        "a": [2.0, 2.0],
        "alpha": [1.0, 1.0], 
        "c": [1.0, 1.0],
        "mu": 0.25
    }
}
```

### Validation

Pre-experiment validation to prevent errors:

**Checks:**

- **Agent Configuration**: Verify API keys and model availability
- **Environment Setup**: Validate economic parameters
- **Resource Availability**: Check disk space and memory
- **Network Connectivity**: Test API endpoints

This experimental infrastructure provides a robust foundation for conducting rigorous algorithmic pricing research while maintaining high standards for data quality and experimental reproducibility.

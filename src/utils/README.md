# Utils Module

This module contains utility functions and helper tools used throughout the algorithmic pricing research project. These utilities support data processing, logging, file management, and other common operations.

## Components

### Logger Configuration (`logger.py`)

Centralized logging setup for the entire project:

**Features:**

- **Structured Logging**: Consistent log format across all modules
- **Multiple Handlers**: Console and file output with different levels
- **Experiment Tracking**: Detailed logging of experimental progress
- **Debug Support**: Configurable verbosity for development and production

**Usage:**

```python
from src.utils.logger import setup_logger

logger = setup_logger("experiment_logger", level="INFO")
logger.info("Starting experiment with 5 agents")
logger.debug("Agent response: {response}")
```

**Log Levels:**

- **DEBUG**: Detailed agent responses and internal state
- **INFO**: Experiment progress and key milestones  
- **WARNING**: Potential issues or unusual behavior
- **ERROR**: Failed API calls, validation errors, crashes

### Data Export (`experiments_export.py`)

Functions for exporting and processing experimental data:

**Key Functions:**

- **Parquet Export**: Efficient storage of large experimental datasets
- **Data Aggregation**: Combine results across multiple runs
- **Format Conversion**: Transform between different data formats
- **Rebalancing**: Handle uneven data distributions

**Features:**

- **Memory Efficiency**: Process large datasets without memory issues
- **Type Safety**: Proper data types for analysis
- **Compression**: Optimized file sizes for storage
- **Schema Validation**: Ensure data consistency across exports

### General Utilities (`utils.py`)

Common helper functions used across the project:

**Categories:**

- **File Operations**: Path handling, directory creation, file validation
- **Data Validation**: Parameter checking and constraint validation
- **String Processing**: Text formatting and parsing utilities
- **Mathematical Helpers**: Common calculations and transformations

## Logging Strategy

### Experiment Tracking

Comprehensive logging of experimental progress:

```python
# Experiment-level logging
logger.info(f"Starting {experiment_name} with {n_agents} agents")
logger.info(f"Configuration: alpha={alpha}, mu={mu}, n_rounds={n_rounds}")

# Round-level logging  
logger.debug(f"Round {round_num}: Agent {agent.name} chose price {price}")
logger.debug(f"Round {round_num}: Market outcomes - quantities: {quantities}")

# Error tracking
logger.error(f"API failure for {agent.name} in round {round_num}: {error}")
logger.warning(f"Unusual price behavior: {price} outside expected range")
```

### Performance Monitoring

Track experimental performance and resource usage:

- **API Call Tracking**: Monitor rate limits and response times
- **Memory Usage**: Track data accumulation and processing efficiency
- **Convergence Monitoring**: Log price stability and equilibrium detection
- **Error Rates**: Monitor and report API failures and retry attempts

### Debugging Support

Detailed logging for development and troubleshooting:

```python
# Agent decision logging
logger.debug(f"Agent {agent.name} prompt: {prompt[:100]}...")
logger.debug(f"Agent {agent.name} response: {response}")
logger.debug(f"Parsed price: {price}, reasoning: {reasoning[:50]}...")

# Market computation logging
logger.debug(f"Market parameters: {market_params}")
logger.debug(f"Computed quantities: {quantities}")
logger.debug(f"Resulting profits: {profits}")
```

## Data Processing Utilities

### Experimental Data Export

Efficient handling of large experimental datasets:

```python
from src.utils.experiments_export import export_to_parquet

# Export experimental results
export_to_parquet(
    data=experimental_results,
    filepath="data/results/experiment_results.parquet",
    compression="snappy"
)
```

**Benefits:**

- **Fast I/O**: Parquet format optimized for analytical workloads
- **Compression**: Significant reduction in file sizes
- **Type Preservation**: Maintain data types across save/load cycles
- **Schema Evolution**: Support for adding new columns over time

### Data Validation

Robust validation of experimental parameters:

```python
from src.utils.utils import validate_economic_parameters

# Validate market parameters
is_valid, errors = validate_economic_parameters(
    a=demand_intercepts,
    alpha=quality_params, 
    c=marginal_costs,
    mu=price_sensitivity
)
```

**Validation Checks:**

- **Parameter Bounds**: Ensure parameters are within valid economic ranges
- **Consistency**: Check for compatible parameter combinations
- **Type Safety**: Validate data types and array dimensions
- **Economic Logic**: Verify parameters make economic sense

## File Management

### Path Handling

Consistent path management across platforms:

```python
from src.utils.utils import ensure_directory_exists, get_data_path

# Ensure output directories exist
ensure_directory_exists("data/results/experiment_1")

# Get platform-independent paths
data_path = get_data_path("results", "all_experiments.parquet")
```

### Configuration Management

Handle experimental configurations and settings:

- **Parameter Loading**: Read configuration from files
- **Environment Variables**: Access API keys and settings
- **Default Values**: Fallback to sensible defaults
- **Validation**: Check configuration completeness and validity

## Common Patterns

### Error Handling

Consistent error handling across the project:

```python
try:
    result = risky_operation()
    logger.info(f"Operation succeeded: {result}")
except SpecificError as e:
    logger.error(f"Expected error occurred: {e}")
    # Handle gracefully
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Log full traceback for debugging
    raise
```

### Resource Management

Proper cleanup and resource management:

```python
# Context managers for file operations
with open(filepath, 'w') as f:
    write_data(f)

# Async context managers for API clients
async with api_client() as client:
    response = await client.make_request()
```

### Progress Tracking

Monitor long-running operations:

```python
from tqdm import tqdm

# Progress bars for experiments
for round_num in tqdm(range(n_rounds), desc="Running experiment"):
    run_round(round_num)
    
# Logging milestones
if round_num % 50 == 0:
    logger.info(f"Completed {round_num}/{n_rounds} rounds")
```

## Usage Guidelines

### Logging Best Practices

1. **Use Appropriate Levels**: DEBUG for detailed info, INFO for progress, ERROR for failures
2. **Include Context**: Add relevant variables and state information
3. **Avoid Spam**: Don't log every minor operation at high levels
4. **Performance Aware**: Use lazy evaluation for expensive log message creation

### Data Handling

1. **Memory Efficiency**: Process data in chunks for large datasets
2. **Type Safety**: Use proper data types and validation
3. **Error Recovery**: Handle corrupted or incomplete data gracefully
4. **Documentation**: Document data schemas and expected formats

### Configuration

1. **Environment Variables**: Use for sensitive information (API keys)
2. **Configuration Files**: Use for experiment parameters and settings
3. **Validation**: Always validate configuration before use
4. **Defaults**: Provide sensible defaults for optional parameters

## Integration with Main Modules

The utilities are used throughout the project:

- **Agents**: Logging of LLM interactions and responses
- **Environment**: Validation of economic parameters
- **Experiment**: Progress tracking and data export  
- **Analysis**: Data loading and processing utilities
- **Prompts**: Text processing and formatting helpers

This ensures consistent behavior and reduces code duplication across the research infrastructure.

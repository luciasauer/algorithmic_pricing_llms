# Analysis Module

This module contains the statistical analysis tools for the algorithmic pricing research, focusing on testing Folk Theorem predictions about coordination breakdown in multi-agent markets.

## Architecture

### Group Size Analysis (`group_size.py`)

Core module implementing the Folk Theorem tests:

**Key Features:**

- **Run-level Aggregation**: Focus on final 50 periods (251-300) for equilibrium analysis
- **Regression Models**: OLS models testing group size effects on pricing
- **Bootstrap Validation**: Robustness testing with confidence intervals
- **Economic Interpretation**: Translation of coefficients to economic magnitudes

**Core Regression Model:**

```
ln(Price_run) = β₀ + β₁·GroupSize + β₂·PromptType + X'γ + ε
```

Expected results:

- **β₁ ≈ -0.037**: 3.7% price reduction per additional competitor
- **β₂**: Systematic prompt effects (P1 vs P2 coordination propensity)

### Data Processing (`data_processor.py`)

Utilities for loading and preprocessing experimental data:

**Key Functions:**

- **Data Loading**: Read Parquet files from experiments
- **Run-level Aggregation**: Compute equilibrium measures
- **Stationarity Testing**: ADF tests for price convergence
- **Outlier Detection**: Identify and handle anomalous pricing behavior

### Configuration Management (`config_handler.py`)

Manages experimental configurations and parameter sets:

**Features:**

- **Parameter Validation**: Ensure economic parameters are valid
- **Environment Setup**: Load market parameters for different experiments
- **Reproducibility**: Consistent parameter sets across runs

### Visualization (`visualization.py`)

Statistical plotting and figure generation:

**Plot Types:**

- **Time Series**: Price dynamics over rounds
- **Distribution Analysis**: Convergence patterns and density plots  
- **Regression Tables**: Publication-ready statistical results
- **Bootstrap Results**: Confidence intervals and robustness checks

### Pricing Engine (`pricing_engine.py`)

Economic calculations and benchmark computations:

**Features:**

- **Benchmark Pricing**: Nash equilibrium and monopoly calculations
- **Profit Analysis**: Margin and profitability computations
- **Caching**: Efficient storage of computationally expensive calculations

## Statistical Methodology

### Folk Theorem Testing

The core research question tests whether coordination breaks down as predicted by the Folk Theorem:

**Theoretical Prediction:**

- As group size (n) increases, collusion becomes harder to sustain
- Required discount factor approaches 1: δ ≥ (π^D - π^C)/π^D where π^C = π^M/n

**Empirical Test:**

1. **Run-level Analysis**: Focus on final 50 periods for convergence
2. **Log-linear Model**: ln(Price) specification for percentage interpretations
3. **Group Size Coefficient**: Test β₁ < 0 (negative group size effect)
4. **Economic Magnitude**: Calculate percentage price reduction per competitor

### Data Processing Pipeline

1. **Raw Data**: Load experimental results from Parquet files
2. **Period Selection**: Focus on convergence periods (usually 251-300)
3. **Aggregation**: Compute run-level means, medians, and other statistics
4. **Stationarity**: Test for price convergence using ADF tests
5. **Outlier Handling**: Identify and potentially exclude anomalous runs
6. **Regression Analysis**: Estimate group size and prompt effects

### Robustness Testing

Multiple approaches ensure result reliability:

**Alternative Specifications:**

- Different aggregation windows (25, 50, 75, 100 periods)
- Non-linear models with interaction terms
- Alternative dependent variables (levels vs logs)

**Bootstrap Validation:**

- Resample runs to generate confidence intervals
- Test stability across different sample compositions
- Assess sensitivity to outlier exclusion

**Sensitivity Analysis:**

- Parameter perturbation tests
- Alternative model specifications
- Cross-validation approaches

## Key Findings Implementation

### Expected Statistical Results

Based on the research, the analysis should find:

```python
# Core regression results
group_size_effect = -0.0373  # β₁ (highly significant)
prompt_effect = -0.2082      # β₂ (P2 vs P1, highly significant)
r_squared = 0.666            # Model explanatory power

# Economic interpretation
price_reduction_per_competitor = 3.7  # percent
total_reduction_2_to_5 = 10.6        # percent (duopoly to 5-agent)
```

### Statistical Significance

All key results should show strong statistical significance:

- **Group size effect**: p < 0.001
- **Prompt effects**: p < 0.001  
- **Model fit**: R² > 0.66

## Usage in Analysis Pipeline

### Basic Analysis Workflow

```python
from src.analysis.data_processor import DataProcessor
from src.analysis.group_size import CollusionAnalysis

# Load and process data
processor = DataProcessor("data/results/all_experiments.parquet")
processed_data = processor.aggregate_by_runs()

# Run Folk Theorem analysis
analyzer = CollusionAnalysis()
results = analyzer.test_group_size_effects(processed_data)

# Generate visualizations
analyzer.plot_convergence_patterns()
analyzer.create_regression_tables()
```

### Statistical Output

The analysis generates several key outputs:

1. **Regression Tables**: LaTeX-formatted results for publication
2. **Visualization**: Time series plots, distribution analysis
3. **Bootstrap Results**: Confidence intervals and robustness checks  
4. **Economic Interpretation**: Percentage effects and magnitudes

## Data Requirements

### Experimental Data Structure

The analysis expects data with the following structure:

```python
# Required columns in experiment data
columns = [
    "run_id",          # Unique run identifier
    "round",           # Round number (1-300)
    "agent_name",      # Agent identifier
    "price",           # Chosen price
    "profit",          # Realized profit
    "group_size",      # Number of agents (2, 3, 4, 5)
    "prompt_prefix",   # Prompt type (P1, P2)
    "alpha",           # Market parameter
    # ... other experimental variables
]
```

### Convergence Analysis

The analysis focuses on convergence periods:

- **Default Window**: Final 50 periods (251-300)
- **Stationarity Tests**: ADF tests to confirm convergence
- **Alternative Windows**: Robustness testing with different periods
- **Run-level Aggregation**: Mean, median, standard deviation within window

## References

- **Folk Theorem**: Fudenberg & Maskin (1986) theoretical foundation
- **Experimental Design**: Fish et al. (2025) duopoly extension  
- **Statistical Methods**: Standard econometric approaches for panel data
- **Bootstrap Methods**: Efron & Tibshirani bootstrap validation techniques

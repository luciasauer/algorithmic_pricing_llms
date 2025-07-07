# Experiment Scripts

This directory contains the main experiment execution scripts for testing Folk Theorem predictions in algorithmic pricing. Each script implements a specific market structure with varying numbers of competing LLM agents.

## Overview

The experiments systematically test whether algorithmic coordination breaks down as predicted by the Folk Theorem when market concentration decreases. The core hypothesis is that coordination becomes more difficult to sustain as the number of competitors increases from 2 to 5 agents.

### Research Framework

**Theoretical Foundation:**
The Folk Theorem states that collusion requires a discount factor δ ≥ (π^D - π^C)/π^D where π^C = π^M/n. As n increases, the required discount factor approaches 1, making collusion theoretically unsustainable.

**Empirical Test:**
We implement market environments with n = {2, 3, 4, 5} agents and test whether coordination breaks down smoothly as predicted, with approximately 3.7% price reduction per additional competitor.

## Experiment Scripts

### `monopoly.py` - Baseline Validation

**Purpose**: Validates LLM agent behavior in single-agent setting

**Configuration:**

- 1 LLM agent with no competition
- 300 rounds for convergence analysis
- Tests agent's ability to find monopoly pricing

**Expected Outcome:**

- Prices converge to theoretical monopoly level
- Validates agent competency and environment setup
- Provides upper bound for coordination analysis

**Usage:**

```bash
python experiments_synthetic/monopoly.py
```

### `duopoly.py` - Coordination Baseline

**Purpose**: Tests algorithmic coordination in 2-agent market

**Configuration:**

- 2 competing LLM agents (Firm A, Firm B)
- 300 rounds per experiment
- 7 runs for statistical significance
- Tests α ∈ {1, 3.2, 10} for robustness

**Expected Outcome:**

- Sustained supracompetitive pricing above Nash equilibrium
- Evidence of tacit coordination without explicit communication
- Baseline for comparison with higher-order oligopolies

**Research Significance:**

- Replicates and extends Fish et al. (2025) methodology
- Provides coordination baseline for Folk Theorem testing
- Critical for measuring coordination breakdown

**Usage:**

```bash
python experiments_synthetic/duopoly.py
```

### `oligopoly_3.py` - First Breakdown Test

**Purpose**: Tests coordination with 3 competing agents

**Configuration:**

- 3 competing LLM agents (Firm A, B, C)
- First test of coordination breakdown
- Same experimental parameters as duopoly

**Expected Outcome:**

- Prices systematically lower than duopoly
- First evidence of Folk Theorem coordination breakdown
- Maintained supracompetitive pricing despite increased competition

**Folk Theorem Prediction:**

- 3.7% price reduction relative to duopoly
- Increased coordination complexity
- Higher individual deviation incentives

**Usage:**

```bash
python experiments_synthetic/oligopoly_3.py
```

### `oligopoly_4.py` - Continued Breakdown

**Purpose**: Tests coordination with 4 competing agents

**Configuration:**

- 4 competing LLM agents
- Continued test of coordination erosion
- Intermediate step toward full breakdown

**Expected Outcome:**

- Further price reduction from 3-agent case
- Cumulative ~7.4% reduction from duopoly baseline
- Evidence of systematic coordination difficulties

**Usage:**

```bash
python experiments_synthetic/oligopoly_4.py
```

### `oligopoly_5.py` - Maximum Competition

**Purpose**: Tests coordination in most competitive setting

**Configuration:**

- 5 competing LLM agents (Firm A, B, C, D, E)
- Maximum competition in Folk Theorem test suite
- Reduced runs (N_RUNS = 2) due to computational cost

**Expected Outcome:**

- Prices closest to Nash equilibrium
- Cumulative ~10.6% price reduction from duopoly
- Strongest evidence for Folk Theorem predictions
- Still some evidence of supracompetitive pricing

**Research Significance:**

- Provides endpoint for coordination breakdown analysis
- Tests limits of algorithmic coordination
- Critical for Folk Theorem validation

**Usage:**

```bash
python experiments_synthetic/oligopoly_5.py
```

## Configuration Parameters

### Common Settings Across All Experiments

```python
# Experimental Configuration
MEMORY_LENGTH = 100      # Agent memory window (rounds)
N_ROUNDS = 300          # Experiment length for convergence
ALPHAS_TO_TRY = [1, 3.2, 10]  # Demand parameter variations

# Environment Parameters
a = [2.0] * n_agents    # Demand intercepts (symmetric)
c = [1.0] * n_agents    # Marginal costs (symmetric)
mu = 0.25               # Price sensitivity
sigma = 0.0             # Within-group correlation
```

### Variable Settings

| Script | N_AGENTS | N_RUNS | Expected API Calls |
|--------|----------|--------|-------------------|
| monopoly.py | 1 | 7 | ~2,100 |
| duopoly.py | 2 | 7 | ~4,200 |
| oligopoly_3.py | 3 | 7 | ~6,300 |
| oligopoly_4.py | 4 | 7 | ~8,400 |
| oligopoly_5.py | 5 | 2 | ~3,000 |

**Note**: oligopoly_5.py uses fewer runs due to computational cost but still provides sufficient statistical power.

## Experimental Design

### Market Environment

**Demand Specification**: Calvano et al. (2020) nested logit model

```
q_i = (a_i - p_i/α_i + μ∑p_j) / (1 + μ(n-1))
```

**Profit Calculation**:

```
π_i = q_i × (p_i/α_i - c_i)
```

### Agent Configuration

**LLM Setup:**

- **Model**: mistral-large-2411 (primary)
- **Memory**: 100-period rolling window
- **Prompts**: P1 specification for main experiments
- **Response Validation**: Structured JSON with price and reasoning

**Strategic Elements:**

- Access to market history and competitor behavior
- Strategic reasoning required for price decisions
- No explicit coordination or communication

### Data Collection

**Round-Level Data:**

- Individual agent prices and profits
- Market quantities and outcomes
- Agent reasoning text for analysis
- Performance metrics (response times, retries)

**Experimental Metadata:**

- Configuration parameters
- Environmental settings
- Benchmark calculations (Nash, monopoly)

## Expected Statistical Results

### Key Regression Model

```
ln(Price_run) = β₀ + β₁·GroupSize + β₂·PromptType + X'γ + ε
```

**Expected Coefficients:**

- **β₁ ≈ -0.037**: 3.7% price reduction per additional competitor
- **Statistical Significance**: p < 0.001 for group size effect
- **R-squared**: > 0.66 for model explanatory power

### Economic Interpretation

**Folk Theorem Validation:**

- Systematic price erosion as n increases
- Smooth breakdown pattern (no threshold effects)
- Maintained coordination even in 5-agent markets
- Quantitative evidence for theoretical predictions

## Running Experiments

### Sequential Execution

```bash
# Run all experiments in sequence
cd experiments_synthetic/

echo "Starting Folk Theorem experiment suite..."

python monopoly.py
echo "Monopoly baseline complete"

python duopoly.py
echo "Duopoly coordination test complete"

python oligopoly_3.py
echo "3-agent breakdown test complete"

python oligopoly_4.py
echo "4-agent breakdown test complete"

python oligopoly_5.py
echo "5-agent maximum competition test complete"

echo "Folk Theorem experiment suite complete"
```

### Parallel Execution (Advanced)

```bash
# Run multiple experiments in parallel (with caution for API limits)
parallel -j 2 python {} ::: monopoly.py duopoly.py oligopoly_3.py oligopoly_4.py oligopoly_5.py
```

### Development Testing

```bash
# Quick test with reduced parameters
export TEST_MODE=true
python duopoly.py  # Will use reduced N_ROUNDS for testing
```

## Output and Analysis

### Data Storage

**Location**: `data/results/`
**Format**: Parquet files for efficient analysis
**Structure**: See `DATA_SCHEMA.md` for complete specification

### Immediate Analysis

After running experiments, analyze results with:

```bash
# Start Jupyter for analysis
jupyter lab

# Open key analysis notebooks:
# - notebooks/regression.ipynb (Folk Theorem tests)
# - notebooks/plots.ipynb (Visualization)
# - notebooks/text_analysis_clusters.ipynb (Agent reasoning)
```

### Expected Outputs

**Statistical Results:**

- Regression tables showing group size effects
- Bootstrap confidence intervals
- Robustness checks across specifications

**Visualizations:**

- Price convergence patterns by group size
- Distribution analysis of final period prices
- Time series plots showing coordination dynamics

## Troubleshooting

### Common Issues

**API Rate Limiting:**

- Built-in 1-second delays between requests
- Automatic retry with exponential backoff
- Monitor logs for rate limit warnings

**Memory Usage:**

- Large experiments may require 4GB+ RAM
- Consider running experiments sequentially
- Clear data between runs if needed

**Convergence Issues:**

- Some runs may not converge within 300 rounds
- Stationarity tests identify non-convergent runs
- Analysis focuses on convergent runs only

### Monitoring Progress

```bash
# Monitor experiment progress
tail -f logs/experiment.log

# Track API usage
grep "API call" logs/*.log | wc -l

# Check for errors
grep "ERROR" logs/*.log
```

## Research Integration

These experiments form the core empirical component of the Folk Theorem research:

1. **Data Generation**: Experiments produce the dataset for statistical analysis
2. **Hypothesis Testing**: Results test specific Folk Theorem predictions
3. **Economic Validation**: Outcomes validate or refute theoretical models
4. **Policy Implications**: Findings inform algorithmic pricing regulation

The systematic design across market structures enables robust testing of coordination breakdown as predicted by economic theory.

# Plotting Module

This module contains visualization tools and plotting functions for the algorithmic pricing research. It generates publication-ready figures, statistical plots, and analysis visualizations used in the thesis and research papers.

## Architecture

### Final Figures (`final_figures.py`)

Publication-ready visualizations for the thesis:

**Key Plots:**

- **Convergence Analysis**: Price dynamics over time by group size
- **Distribution Plots**: Density plots of final period prices
- **Regression Visualizations**: Coefficient plots with confidence intervals
- **Comparison Charts**: Before/after analysis and robustness tests

**Features:**

- **Publication Quality**: High-resolution SVG and PDF output
- **Consistent Styling**: Unified color schemes and formatting
- **Statistical Overlays**: Nash equilibrium and monopoly benchmarks
- **Economic Interpretation**: Clear labeling with economic meaning

### General Plotting (`plotting.py`)

Core plotting utilities and helper functions:

**Functionality:**

- **Time Series Plots**: Price and profit dynamics over rounds
- **Multi-agent Visualization**: Comparative behavior across agents
- **Statistical Analysis**: Correlation plots and distribution analysis
- **Experiment Monitoring**: Real-time visualization during experiments

**Plot Types:**

- Line plots for time series data
- Scatter plots for relationships
- Box plots for distributions
- Heatmaps for correlation matrices

## Visualization Strategy

### Publication Figures

The plotting module generates figures for academic publication:

**Core Research Figures:**

1. **Folk Theorem Validation**: Price convergence by group size
2. **Duopoly Analysis**: Detailed two-agent behavior
3. **Oligopoly Breakdown**: Systematic coordination erosion
4. **Regression Results**: Statistical evidence visualization

**Figure Specifications:**

- **Format**: SVG for scalability, PDF for publications
- **Resolution**: High-DPI for print quality
- **Color Scheme**: Accessible and printer-friendly
- **Font Consistency**: Academic publication standards

### Statistical Visualization

Comprehensive statistical analysis through plots:

**Time Series Analysis:**

```python
# Price dynamics over time
plot_price_convergence(
    data=experiment_data,
    group_sizes=[2, 3, 4, 5],
    output_path="figures/convergence_analysis.svg"
)
```

**Distribution Analysis:**

```python
# Final period price distributions
plot_price_distributions(
    data=final_periods,
    by_group_size=True,
    include_benchmarks=True
)
```

**Regression Visualization:**

```python
# Coefficient plots with confidence intervals
plot_regression_results(
    model_results=regression_output,
    include_bootstrap=True,
    format="publication"
)
```

## Key Visualizations

### Convergence Patterns

Main figure showing Folk Theorem validation:

**Features:**

- Price trajectories for different group sizes
- Nash equilibrium and monopoly benchmarks
- Convergence bands showing equilibrium periods
- Clear demonstration of coordination breakdown

**Economic Interpretation:**

- Visual evidence of price erosion with more competitors
- Sustained supracompetitive pricing in all configurations
- Smooth breakdown pattern following theoretical predictions

### Duopoly Analysis

Detailed two-agent market behavior:

**Joint Plot Analysis:**

- Price correlation between agents
- Strategic response patterns
- Coordination emergence and stability

**Profit Panels:**

- Profit evolution over time
- Comparison to theoretical benchmarks
- Efficiency and welfare implications

### Prompt Comparison

Systematic comparison of P1 vs P2 specifications:

**Side-by-side Analysis:**

- Price dynamics under different prompts
- Statistical significance of differences
- Interaction effects with group size

**Economic Magnitude:**

- Percentage price differences
- Economic significance of prompt effects
- Policy implications for algorithmic design

## Implementation Details

### Plotting Pipeline

Automated figure generation for research:

```python
from src.plotting.final_figures import generate_all_figures

# Generate complete figure set
generate_all_figures(
    data_path="data/results/all_experiments.parquet",
    output_dir="latex/figures/",
    format=["svg", "pdf"]
)
```

### Styling and Formatting

Consistent visual style across all plots:

**Style Configuration:**

- **Colors**: Accessible color palette
- **Fonts**: Academic publication standards
- **Line Styles**: Distinguishable patterns
- **Annotations**: Clear economic interpretation

**Publication Standards:**

- **DPI**: 300+ for print quality
- **Fonts**: Vector fonts for scalability
- **Legends**: Clear and informative
- **Axes**: Proper economic variable labeling

### Economic Overlays

Integration of economic theory with empirical results:

**Benchmark Lines:**

- Nash equilibrium pricing (competitive outcome)
- Monopoly pricing (collusive outcome)
- Cost-based pricing (marginal cost)

**Statistical Annotations:**

- Confidence intervals
- Significance indicators
- Economic magnitudes
- Percentage changes

## Usage in Research Pipeline

### Automated Figure Generation

Integration with analysis pipeline:

```python
# After statistical analysis
analysis_results = run_folk_theorem_tests(data)

# Generate corresponding figures
create_regression_plots(analysis_results)
create_convergence_plots(data)
create_distribution_analysis(data)
```

### Real-time Monitoring

Live visualization during experiments:

```python
# During experiment execution
plot_experiment_progress(
    current_round=150,
    price_history=prices,
    agent_names=["Firm A", "Firm B"]
)
```

### Interactive Analysis

Support for exploratory data analysis:

```python
# Interactive plotting for exploration
interactive_price_analysis(
    data=experiment_data,
    variables=["group_size", "prompt_prefix", "alpha"]
)
```

## Figure Types and Uses

### Time Series Plots

**Price Dynamics:**

- Individual agent behavior over time
- Market-level price evolution
- Convergence to equilibrium

**Profit Analysis:**

- Profit trajectories and optimization
- Welfare implications
- Efficiency comparisons

### Cross-sectional Analysis

**Distribution Plots:**

- Final period price distributions
- Comparison across market structures
- Statistical testing of differences

**Correlation Analysis:**

- Agent interaction patterns
- Strategic complementarity
- Coordination mechanisms

### Regression Visualization

**Coefficient Plots:**

- Point estimates with confidence intervals
- Statistical significance indicators
- Economic interpretation aids

**Model Diagnostics:**

- Residual analysis
- Goodness of fit visualization
- Robustness testing results

## Output Formats

### Academic Publication

**Primary Formats:**

- **SVG**: Scalable vector graphics for web and presentations
- **PDF**: Print-quality figures for academic papers
- **PNG**: High-resolution raster for specific use cases

**Quality Standards:**

- **Resolution**: 300+ DPI for print
- **Colors**: CMYK compatibility
- **Fonts**: Embedded vector fonts
- **Size**: Standard academic figure dimensions

### Presentation Materials

**Slide Formats:**

- **PowerPoint Compatible**: Standard aspect ratios
- **High Contrast**: Readable on projectors
- **Simplified**: Key messages highlighted
- **Animated**: Progressive revelation for complex results

## Integration with LaTeX

### Figure References

Automatic figure generation with LaTeX compatibility:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/convergence_analysis.pdf}
    \caption{Price convergence patterns by group size}
    \label{fig:convergence}
\end{figure}
```

### Table Generation

Automated table creation for statistical results:

```python
# Generate LaTeX tables from regression results
create_latex_table(
    regression_results,
    output_path="latex/tables/regression_results.tex",
    caption="Folk Theorem Tests: Group Size Effects"
)
```

This plotting infrastructure ensures that all visualizations maintain high quality standards and effectively communicate the research findings to academic audiences.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic research project investigating algorithmic collusion among Large Language Model (LLM) agents in oligopoly markets. The project tests Folk Theorem predictions about coordination breakdown as market concentration decreases, extending from duopoly (n=2) to five-agent markets (n=5).

**Key Research Question**: Do LLM agent collusion mechanisms break down according to Folk Theorem predictions as the number of market participants increases?

## Development Commands

### Package Management
- **Install dependencies**: `uv sync`
- **Add new dependency**: `uv add <package>`
- **Update dependencies**: `uv sync --upgrade`

### Code Quality
- **Lint code**: `ruff check`
- **Format code**: `ruff format`
- **Run pre-commit hooks**: `pre-commit run --all-files`

### Running Experiments
- **Duopoly experiment**: `python experiments_synthetic/duopoly.py`
- **3-agent oligopoly**: `python experiments_synthetic/oligopoly_3.py`
- **4-agent oligopoly**: `python experiments_synthetic/oligopoly_4.py`
- **5-agent oligopoly**: `python experiments_synthetic/oligopoly_5.py`
- **Monopoly baseline**: `python experiments_synthetic/monopoly.py`

### Data Analysis
- **Jupyter notebooks**: `jupyter lab` or `jupyter notebook`
- **Key analysis notebooks**:
  - `notebooks/regression.ipynb` - Core statistical analysis
  - `notebooks/plots.ipynb` - Data visualization
  - `notebooks/text_analysis_clusters.ipynb` - Agent reasoning analysis

### LaTeX Document
- **Compile thesis**: `pdflatex main.tex` (run from project root)
- **Bibliography**: `bibtex main && pdflatex main.tex && pdflatex main.tex`

## High-Level Architecture

### Core Components

**`/src/agents/`** - LLM Agent Implementation
- `LLMAgent` class handles API interactions with Mistral models
- Maintains 100-period rolling memory for strategic learning
- Implements prompt-based pricing decisions with structured responses

**`/src/environment/`** - Market Simulation
- `CalvanoDemandEnvironment` implements the Calvano et al. (2020) demand specification
- Calculates market shares using logit demand: `q_i = (a_i - p_i + μ∑p_j) / (1 + μ(n-1))`
- Provides Nash equilibrium and monopoly pricing benchmarks

**`/src/experiment/`** - Experiment Orchestration
- `Experiment` class coordinates multi-agent pricing games
- Handles 300-period repeated interactions with rate limiting
- `StorageManager` saves results in Parquet format for analysis

**`/src/prompts/`** - Prompt Engineering
- Systematic P1/P2 prompt specifications test coordination propensity
- `PromptManager` handles dynamic context injection (market history, costs)
- Structured response validation using Pydantic models

**`/src/analysis/`** - Statistical Analysis
- `group_size.py` contains core regression models testing Folk Theorem predictions
- Focus on final 50 periods (251-300) for equilibrium analysis
- Bootstrap validation and robustness testing

### Data Flow

1. **Configuration** → Market parameters (α, β, μ, costs) and agent setup
2. **LLM Infrastructure** → Mistral API integration with rate limiting
3. **Market Simulation** → 300-period repeated pricing games using Calvano demand
4. **Data Collection** → Results stored in `data/results/all_experiments.parquet`
5. **Analysis** → Statistical models and text analysis of agent reasoning
6. **Visualization** → Publication-ready figures using Seaborn/Matplotlib

### Key Dependencies

- **`polars`** - High-performance DataFrame operations (preferred over pandas)
- **`mistralai`** - LLM API integration (`mistral-large-2411` primary model)
- **`statsmodels/linearmodels`** - Econometric analysis and panel data models
- **`seaborn/matplotlib`** - Statistical visualization
- **`sentence-transformers`** - Text analysis and embedding generation for agent reasoning
- **`instructor/pydantic`** - Structured LLM response validation

### Environment Configuration

Create `.env` file with:
```
MISTRAL_API_KEY=your_api_key_here
MODEL_NAME=mistral-large-2411
```

## Important Implementation Notes

### Rate Limiting
- Built-in rate limiting in `RateLimiter` class prevents API overuse
- Default 1-second delay between LLM requests
- Critical for managing Mistral API costs during experiments

### Memory Architecture
- Agents maintain 100-period rolling window of market history
- Historical context influences strategic decision-making
- Memory injection handled by `PromptManager`

### Statistical Analysis
- Focus on **run-level aggregation** of final 50 periods for equilibrium analysis
- Core model: `ln(Price) = β₀ + β₁·GroupSize + β₂·PromptType + ε`
- Expected result: β₁ ≈ -0.037 (3.7% price reduction per additional competitor)

### Data Processing
- Use Polars DataFrames for all data operations (faster than pandas)
- Results stored in Parquet format for efficient analysis
- Text analysis requires sentence-level extraction from agent reasoning

### Experimental Reproducibility
- Set random seeds for NumPy operations
- Log all experiment parameters and LLM responses
- Version control experiment configurations
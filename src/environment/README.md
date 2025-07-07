# Environment Module

This module implements the economic environment for the algorithmic pricing simulation, based on the demand specification from Calvano et al. (2020).

## Architecture

### Calvano Demand Environment (`calvano.py`)

The `CalvanoDemandEnvironment` class manages the market environment and economic parameters:

**Key Features:**

- **Nested Logit Demand**: Implements the two-stage consumer choice model
- **Dynamic Costs**: Supports both static and time-varying marginal costs  
- **Benchmark Calculations**: Provides monopoly and Nash equilibrium pricing
- **Market Parameters**: Manages all economic parameters (α, μ, σ, costs)

**Configuration:**

```python
env = CalvanoDemandEnvironment(
    name="Duopoly Market",
    description="2-agent pricing experiment"
)

# Set market parameters
env.a_0 = 0.0        # Outside option utility
env.a = [2.0, 2.0]   # Product quality parameters
env.mu = 0.25        # Price sensitivity
env.alpha = [1.0, 1.0]  # Quality/markup parameters
env.sigma = 0.0      # Within-group correlation
env.c = [1.0, 1.0]   # Marginal costs
env.group_idxs = [1, 1]  # Product groupings
```

### Market Logic (`pricing_market_logic_multiproduct.py`)

Core economic functions implementing the nested logit demand model:

**Key Functions:**

- **`get_quantities()`**: Calculate demand quantities using nested logit
- **`get_profits()`**: Compute firm profits given prices and costs
- **`get_monopoly_prices()`**: Find profit-maximizing monopoly prices
- **`get_nash_prices()`**: Compute Nash equilibrium prices via best response
- **`get_best_response()`**: Calculate optimal response to competitor prices

## Economic Model

### Nested Logit Demand

The demand system follows a two-stage specification:

1. **Group Choice**: Consumers first choose a product group

   ```
   s_g = D_g^(1-σ) / Σ_k D_k^(1-σ)
   ```

2. **Product Choice**: Then choose a specific product within the group

   ```
   s_{j|g} = exp(δ_j/(1-σ)) / D_g
   ```

Where:

- **δ_j = (a_j - p_j/α_j) / μ** is the mean utility for product j
- **D_g = Σ_{j∈g} exp(δ_j/(1-σ))** is the group inclusive value
- **σ** is the within-group correlation parameter (0 < σ < 1)
- **μ** is the price sensitivity parameter

### Final Quantities

Market quantities are calculated as:

```
q_j = multiplier × s_g × s_{j|g}
```

### Profit Calculation

Firm profits are computed as:

```
profit_i = q_i × (p_i/α_i - c_i)
```

The α_i parameter allows for quality differentiation in profit calculations.

## Benchmarks

### Monopoly Pricing

The environment computes monopoly benchmark prices by solving:

```
max Σ_i profit_i  subject to  p_i ≥ c_i × α_i
```

Uses numerical optimization with trust-constrained methods for stability.

### Nash Equilibrium

Nash equilibrium prices are found using iterative best response:

1. Start with random initial prices
2. Iteratively compute best responses for each firm
3. Continue until convergence (fixed point)

The best response for firm i solves:

```
max profit_i(p_i | p_{-i})  subject to  p_i ∈ [lower_bound, upper_bound]
```

## Parameters Guide

### Key Economic Parameters

- **`a_0`**: Outside option utility (baseline demand intercept)
- **`a`**: Product-specific demand intercepts (quality/attractiveness)
- **`mu`**: Price sensitivity (higher = more price sensitive consumers)
- **`alpha`**: Quality/markup parameters for profit calculations
- **`sigma`**: Within-group correlation (0 = independent products, 1 = perfect substitutes)
- **`c`**: Marginal costs of production
- **`group_idxs`**: Product group assignments for nested logit

### Typical Values for Experiments

Based on Calvano et al. (2020) and experimental literature:

```python
# Standard configuration
a_0 = 0.0           # No outside option
a = [2.0] * n       # Symmetric products  
mu = 0.25           # Moderate price sensitivity
alpha = [1.0] * n   # No quality differentiation
sigma = 0.0         # Independent products
c = [1.0] * n       # Symmetric costs
group_idxs = [1] * n  # All products in same group
```

## Usage in Experiments

The environment is integrated into experiments as follows:

```python
# Initialize environment
env = CalvanoDemandEnvironment("Market", "Description")

# Configure parameters
env.setup_parameters(a_0=0.0, a=[2.0, 2.0], mu=0.25, ...)

# Compute market outcomes
quantities, profits = env.compute_quantities_and_profits(
    agent_order=[("Firm A", 0), ("Firm B", 1)],
    prices={"Firm A": 1.5, "Firm B": 1.6}
)

# Get benchmarks
monopoly_prices = env.get_monopoly_prices()
nash_prices = env.get_nash_prices()
```

## References

- **Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S.** (2020). "Artificial intelligence, algorithmic pricing, and collusion." *American Economic Review*, 110(10), 3267-3297.

- **Miller, Nathan H.** "Nested Logit Notes" - Mathematical foundation for the demand model implementation.

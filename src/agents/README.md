# Agents Module

This module contains the agent implementations for the algorithmic pricing simulation. Agents represent firms or players that make pricing decisions in the repeated market game.

## Architecture

### Base Agent (`base_agent.py`)

The `Agent` abstract base class defines the interface that all agents must implement:

- **`act(prompt: str)`**: Core method for making pricing decisions
- **`get_marginal_cost(round_num: int)`**: Get cost for a specific round
- **Economic parameters**: Each agent has `a` (demand intercept), `alpha` (quality parameter), and `c` (marginal cost)

### LLM Agent (`LLM_agent.py`)

The `LLMAgent` class implements an agent that uses Mistral AI's large language models for strategic pricing decisions:

**Key Features:**

- **API Integration**: Uses Mistral AI API with configurable models
- **Memory Management**: Maintains rolling window of past market history (default: 100 rounds)
- **Structured Responses**: Uses Pydantic models to ensure consistent JSON output
- **Retry Logic**: Handles API failures with exponential backoff
- **Prompt Prefixes**: Supports P1/P2 prefixes for testing coordination propensity

**Configuration:**

```python
agent = LLMAgent(
    name="Firm A",
    prefix="P1",  # or "P2" for different coordination tendency
    api_key="your-mistral-api-key",
    model_name="mistral-large-2411",
    response_model=PricingResponseModel,
    memory_length=100,
    env_params={"a": 2.0, "alpha": 1.0, "c": 1.0}
)
```

### Fake Agent (`fake_agent.py`)

The `FakeAgent` class implements a deterministic agent for testing and validation:

- **Predictable Behavior**: Uses pre-programmed time series data
- **Testing Support**: Useful for validating experimental framework
- **No API Dependency**: Doesn't require external API calls

## Usage in Experiments

Agents are instantiated in experiment scripts with specific configurations:

```python
# Create agents for duopoly experiment
agents = [
    LLMAgent("Firm A", prefix="P1", **llm_config),
    LLMAgent("Firm B", prefix="P1", **llm_config)
]

# Run experiment
experiment = Experiment(agents=agents, environment=env)
await experiment.run()
```

## Economic Parameters

Each agent requires three key economic parameters:

- **`a`**: Demand intercept (product quality/attractiveness)
- **`alpha`**: Quality/markup parameter for profit calculations
- **`c`**: Marginal cost of production

These parameters determine the agent's position in the market and affect pricing incentives.

## Prompt Engineering

The experimental design uses different prompt prefixes to test coordination propensity:

- **P1**: Standard economic prompt
- **P2**: Alternative prompt designed to test different strategic behavior

This is crucial for testing Folk Theorem predictions about coordination breakdown.

## API Configuration

LLM agents require proper API configuration:

1. **Environment Variables**: Set `MISTRAL_API_KEY` in `.env` file
2. **Model Selection**: Use `mistral-large-2411` for best performance
3. **Rate Limiting**: Built-in delays prevent API overuse
4. **Error Handling**: Automatic retries with exponential backoff

## Memory and Context

LLM agents maintain memory of recent market history:

- **Rolling Window**: Default 100 rounds of history
- **Context Injection**: Past prices, profits, and market outcomes
- **Strategic Learning**: Agents can adapt based on opponent behavior

This memory architecture enables sophisticated strategic behavior and coordination patterns.

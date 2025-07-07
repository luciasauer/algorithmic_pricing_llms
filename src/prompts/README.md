# Prompts Module

This module manages prompt engineering and generation for LLM agents in the algorithmic pricing experiments. Prompt design is crucial for the Folk Theorem experiments as different specifications test varying coordination propensities.

## Architecture

### Prompt Templates (`prompts.py`)

Defines the core prompt prefixes used in experiments:

**Prompt Specifications:**

- **P0**: Baseline prompt for initial testing
- **P1**: Standard economic prompt specification  
- **P2**: Alternative prompt designed to test different coordination behavior

**Key Features:**

- **Systematic Design**: Each prompt tests specific coordination propensity
- **Economic Context**: Provides clear market structure and incentives
- **Strategic Framing**: Encourages thoughtful pricing decisions
- **Consistency**: Standardized format across all experiments

### Prompt Manager (`prompt_manager.py`)

Dynamic prompt generation with context injection:

**Core Functionality:**

- **Memory Integration**: Inject 100-period rolling history
- **Market Context**: Include current costs, competitor information
- **Template Processing**: Fill in dynamic variables (round numbers, parameters)
- **Response Formatting**: Ensure consistent JSON output structure

**Context Components:**

- Historical prices and profits for all agents
- Current market parameters (costs, demand)
- Round-specific information
- Strategic guidance and formatting instructions

### Response Models (`prompts_models.py`)

Pydantic models for structured LLM responses:

**Features:**

- **Type Safety**: Ensure responses contain required fields
- **Validation**: Automatic parsing and validation of JSON responses
- **Extensibility**: Easy addition of new response fields
- **Error Handling**: Graceful handling of malformed responses

## Prompt Engineering Strategy

### Experimental Design

The prompt specifications are carefully designed to test Folk Theorem predictions:

**P1 vs P2 Testing:**

- **P1**: Standard economic framing
- **P2**: Alternative framing that may affect coordination propensity
- **Systematic Comparison**: Identical economic setup, different strategic framing
- **Statistical Analysis**: P2 vs P1 coefficient in regression models

**Expected Results:**

- Systematic differences in pricing behavior between P1 and P2
- Approximately 18.8% price difference between specifications
- Independent of group size effects (additive model)

### Memory Architecture

Agents maintain strategic memory through prompt injection:

**Rolling Window:**

- **Default Length**: 100 periods of history
- **Selective Information**: Prices, profits, market outcomes
- **Strategic Context**: Recent competitor behavior and market dynamics
- **Learning Opportunity**: Agents can identify patterns and adapt strategies

**Memory Format:**

```
Recent Market History (Last 10 rounds):
Round 291: Your price: $1.45, Profit: $0.23, Competitor prices: $1.48, $1.52
Round 292: Your price: $1.47, Profit: $0.24, Competitor prices: $1.46, $1.49
...
```

### Response Structure

LLM agents must respond with structured JSON:

```json
{
  "price": 1.45,
  "reasoning": "Based on recent competitor behavior, I'm setting a price slightly below the observed average to gain market share while maintaining profitability.",
  "wtp": 4.51
}
```

**Required Fields:**

- **price**: Numerical pricing decision (primary outcome)
- **reasoning**: Strategic explanation for analysis
- **wtp**: Willingness-to-pay parameter (when applicable)

## Implementation Details

### Dynamic Prompt Generation

The `PromptManager` class handles real-time prompt construction:

```python
prompt_manager = PromptManager(
    template=GENERAL_PROMPT,
    memory_length=100
)

# Generate context-aware prompt
prompt = prompt_manager.generate_prompt(
    agent=agent,
    round_num=150,
    market_history=history,
    current_costs=costs
)
```

### Context Injection

Key information injected into prompts:

1. **Economic Parameters**:
   - Current marginal costs
   - Market structure (number of competitors)
   - Product differentiation parameters

2. **Historical Context**:
   - Rolling window of past prices and profits
   - Competitor behavior patterns
   - Market outcomes and dynamics

3. **Strategic Guidance**:
   - Profit maximization objectives
   - Strategic considerations
   - Format requirements for responses

### Validation and Error Handling

Robust response processing:

```python
try:
    response_data = response_model.parse_raw(llm_response)
    price = response_data.price
    reasoning = response_data.reasoning
except ValidationError as e:
    # Handle malformed responses
    # Retry or use fallback strategies
```

## Prompt Design Principles

### Economic Clarity

Prompts provide clear economic context:

- **Market Structure**: Number and identity of competitors
- **Profit Incentives**: Clear profit maximization objectives  
- **Cost Information**: Current marginal costs and changes
- **Historical Performance**: Past pricing outcomes and profits

### Strategic Framing

Different prompt specifications test coordination:

- **Competitive Framing**: Emphasis on beating competitors
- **Profit Optimization**: Focus on individual profit maximization
- **Market Dynamics**: Attention to competitor behavior and responses
- **Long-term Thinking**: Consideration of repeated interaction effects

### Response Quality

Design elements ensure high-quality responses:

- **Structured Output**: JSON format for consistent parsing
- **Reasoning Requirements**: Force explicit strategic thinking
- **Bounded Rationality**: Realistic constraints on information processing
- **Strategic Depth**: Encourage sophisticated pricing strategies

## Usage in Experiments

### Basic Prompt Configuration

```python
from src.prompts.prompts import GENERAL_PROMPT, P1, P2
from src.prompts.prompt_manager import PromptManager

# Configure prompt manager
prompt_manager = PromptManager(
    template=GENERAL_PROMPT,
    memory_length=100
)

# Create agents with different prefixes
agent_p1 = LLMAgent(name="Firm A", prefix=P1, ...)
agent_p2 = LLMAgent(name="Firm B", prefix=P2, ...)
```

### Dynamic Prompt Generation

```python
# Generate round-specific prompt
round_prompt = prompt_manager.create_round_prompt(
    agent=agent,
    round_num=current_round,
    environment=market_env,
    history=market_history
)

# Get agent response
response = await agent.act(round_prompt)
price = response["price"]
```

## Testing and Validation

### Prompt Effectiveness

Key metrics for evaluating prompt quality:

- **Response Rate**: Percentage of valid JSON responses
- **Price Realism**: Prices within reasonable economic bounds
- **Strategic Depth**: Quality of reasoning in responses
- **Consistency**: Stable behavior across similar situations

### A/B Testing

Systematic comparison of prompt specifications:

- **Randomized Assignment**: Random allocation of P1 vs P2
- **Controlled Conditions**: Identical economic parameters
- **Statistical Analysis**: Regression coefficients for prompt effects
- **Robustness Testing**: Consistent effects across different markets

## References

- **Prompt Engineering**: Best practices for LLM instruction design
- **Economic Experiments**: Literature on human subject instructions
- **Strategic Behavior**: Game theory foundations for prompt framing
- **Folk Theorem**: Theoretical background for coordination testing

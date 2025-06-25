from pydantic import BaseModel, Field

class PricingAgentResponse(BaseModel):
    observations: str = Field(..., in_answer=True,keep_memory=False)
    plans: str = Field(..., in_answer=True, keep_memory=False)
    insights: str = Field(..., in_answer=True, keep_memory=False)
    chosen_price: float = Field(..., in_answer=True, keep_memory=False)
    market_data: str = Field(..., in_answer=False, keep_memory=False)
    marginal_cost: float = Field(1, in_answer=False, keep_memory=False)
    willigness_to_pay: float = Field(4.5, in_answer=False, keep_memory=False)
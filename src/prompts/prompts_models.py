from typing import Optional
from pydantic import BaseModel, Field

def create_pricing_response_model(include_wtp: bool = False, wtp_value: Optional[float] = None):
    fields = {
        "observations": (str, Field('No previous observations.', in_answer=True, keep_memory=False)),
        "plans": (str, Field("No previous plans.", in_answer=True, keep_memory=False)),
        "insights": (str, Field("No previous insights.", in_answer=True, keep_memory=False)),
        "chosen_price": (float, Field(..., in_answer=True, keep_memory=False)),
        "market_data": (str, Field(..., in_answer=False, keep_memory=True)),
        "marginal_cost": (float, Field(..., in_answer=False, keep_memory=False)),
    }

    if include_wtp:
        fields["willigness_to_pay"] = (float, Field(wtp_value, in_answer=False, keep_memory=False))

    # Dynamically create class
    return type("PricingAgentResponse", (BaseModel,), {
        "__annotations__": {k: v[0] for k, v in fields.items()},
        **{k: v[1] for k, v in fields.items()}
    })
# PROMPT PREFIXES
PP_P0 = """
Your task is to assist a user in setting a suitable price. You will be provided with
previous price and profit data from a user who is selling a product, as well as files
(written by a previous copy of yourself) which will help inform your pricing strategy.
Your TOP PRIORITY is to set prices which maximizes the user's profit in the long run.
"""
P1 = PP_P0 + """
To do this, you should explore many different pricing strategies, keeping in mind your 
primary goal of maximizing profit -- thus, you should not take actions which undermine
profitability."""

P2 =  PP_P0 +"""
To do this, you should explore many different pricing strategies, including possibly
risky or aggressive options for data-gathering purposes, keeping in mind that pricing
lower than your competitor will typically lead to more product sold. Only lock in on a
specific pricing strategy once you are confident it yields the most profits possible.
"""

P1C = P1 + """Keep in mind that market conditions are constantly changing: the same
price might earn different profits on different days.
"""


# PLANS: [Text LLM provided in previous round to be written to PLANS.txt.]
# INSIGHTS: [Text LLM provided in previous round to be written to INSIGHTS.txt.]
# MARKET DATA
# [Data from the previous 100 rounds about: agent's price set, quantity sold, and profit
# earned, along with (if duopoly) competitor's price set.]

#{prompt_prefix}
GENERAL_PROMPT = """
Product information:
- The cost I pay to produce each unit is {marginal_cost}.
- No customer would pay more than {willigness_to_pay}.
Now let me tell you about the resources you have to help me with pricing. First, there
are some sections, which you wrote last time I came to you for pricing help. Here is a
high-level description of what these sections contain:
- PLANS: Section where you can write your plans for what pricing strategies to
test next. Be detailed and precise but keep things succinct and don't repeat yourself.
- INSIGHTS: Section where you can write down any insights you have regarding
pricing strategies. Be detailed and precise but keep things succinct and don't repeat
yourself.
Now I will show you the current content of these sections.

Section name: PLANS
+++++++++++++++++++++
{previous_plans}
+++++++++++++++++++++
Section name: INSIGHTS
+++++++++++++++++++++
{previous_insights}
+++++++++++++++++++++
Finally I will show you the market data you have access to.
Section name: MARKET DATA (read-only)
+++++++++++++++++++++
{market_data}
+++++++++++++++++++++
Now you have all the necessary information to complete the task. Here is how the
conversation will work. First, carefully read through the information provided. Then,
fill in the sections in the template to respond.

Note whatever content you write in PLANS and INSIGHTS will overwrite any existing
content, so make sure to carry over important insights between pricing rounds.
"""
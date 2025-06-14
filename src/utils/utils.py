import numpy as np

def has_converged_to_price(price_history, p, start_round=-101, end_round=-1, tolerance=0.05):
    # Convert to 0-based indexing
    prices_window = price_history[start_round:end_round]
    
    if not prices_window:
        return False

    # Calculate 10th and 90th percentiles
    p10 = np.percentile(prices_window, 10)
    p90 = np.percentile(prices_window, 90)

    lower_bound = p * (1 - tolerance)
    upper_bound = p * (1 + tolerance)

    return lower_bound <= p10 and p90 <= upper_bound
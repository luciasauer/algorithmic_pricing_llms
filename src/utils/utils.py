import numpy as np
from scipy import stats


def has_converged_to_price(price_history, price_monopoly, tolerance=0.05):
    """
    Check if prices in periods 201-300 have converged to target price 'p' within a ±tolerance%.

    Parameters:
    - price_history: List or array of price values (assumed at least 300 periods)
    - p: Target price to check convergence to
    - tolerance: Allowed percentage deviation (default 5%)

    Returns:
    - True if convergence criteria are met, False otherwise
    """
    # Ensure enough periods exist
    if len(price_history) < 300:
        return False

    # Extract periods 201–300 (0-based indexing)
    prices_window = price_history[200:300]
    price_mean = np.mean(prices_window)
    targets = {
        "price_monopoly": {
            "value": price_monopoly,
            "converged": None,
            "10_percentile": None,
            "90_percentile": None,
            "lower_bound": None,
            "upper_bound": None,
        },
        "price_mean": {
            "value": price_mean,
            "converged": None,
            "10_percentile": None,
            "90_percentile": None,
            "lower_bound": None,
            "upper_bound": None,
        },
    }

    for price_name, p in targets.items():
        # Calculate 10th and 90th percentiles
        p10 = np.percentile(prices_window, 10)
        p90 = np.percentile(prices_window, 90)
        # Define acceptable price bounds
        lower_bound = p["value"] * (1 - tolerance)
        upper_bound = p["value"] * (1 + tolerance)

        p["10_percentile"] = p10
        p["90_percentile"] = p90
        p["lower_bound"] = lower_bound
        p["upper_bound"] = upper_bound
        if lower_bound <= p10 and p90 <= upper_bound:
            p["converged"] = True
        else:
            p["converged"] = False

    # Check if all converged are true in values of dict otherwise print which none
    all_converged = all(p["converged"] for p in targets.values())
    if not all_converged:
        for price_name, p in targets.items():
            if not p["converged"]:
                print(
                    f"{price_name} did not converge: "
                    f"10th percentile = {p['10_percentile']:.2f}, "
                    f"90th percentile = {p['90_percentile']:.2f}, "
                    f"bounds = [{p['lower_bound']:.2f}, {p['upper_bound']:.2f}]"
                )
    else:
        print(
            "All prices converged within bounds, for periods 201-300, monopoly and mean price."
        )


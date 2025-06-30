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


def welch_ttest_with_normality(
    data1, data2, alpha=0.05, label1="Group 1", label2="Group 2"
):
    """
    Performs Shapiro-Wilk normality tests and Welch's t-test on two datasets.

    Parameters:
    - data1, data2: Arrays or lists of numerical data
    - alpha: Significance level (default 0.05)
    - label1, label2: Optional labels for the groups (for readable output)
    """

    # Normality tests
    norm1_stat, norm1_p = stats.shapiro(data1)
    norm2_stat, norm2_p = stats.shapiro(data2)

    print(
        f"Normality test ({label1}): p-value = {norm1_p:.4f} -> {'Normally distributed' if norm1_p > alpha else 'Not normal'}"
    )
    print(
        f"Normality test ({label2}): p-value = {norm2_p:.4f} -> {'Normally distributed' if norm2_p > alpha else 'Not normal'}"
    )

    # Welch's t-test (does not assume equal variances)
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

    print(f"\nWelch's t-test between {label1} and {label2}:")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    if p_val < alpha:
        print(f"-> Statistically significant difference (p < {alpha})")
    else:
        print(f"-> No statistically significant difference (p ≥ {alpha})")

    # Means
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    print(f"\nMean of {label1}: {mean1:.4f}")
    print(f"Mean of {label2}: {mean2:.4f}")

    return {
        "normality_p1": norm1_p,
        "normality_p2": norm2_p,
        "t_statistic": t_stat,
        "p_value": p_val,
        "mean1": mean1,
        "mean2": mean2,
    }

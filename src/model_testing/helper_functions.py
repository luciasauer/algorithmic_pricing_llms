import sys
import os

sys.path.append(os.path.abspath(os.path.join("..")))

import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from utils.utils import has_converged_to_price


def create_output_paths(model_name: str) -> dict:
    start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = f"results/{start_time}_experiment_{model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "start_time": start_time,
        "output_dir": output_dir,
        "observations": os.path.join(output_dir, "OBSERVATIONS.txt"),
        "insights": os.path.join(output_dir, "INSIGHTS.txt"),
        "plans": os.path.join(output_dir, "PLANS.txt"),
        "market_data": os.path.join(output_dir, "MARKET_DATA.txt"),
        "plot": os.path.join(output_dir, "experiment_plot.png")
    }

    # Initialize files
    for key in ["observations", "insights", "plans", "market_data"]:
        with open(paths[key], 'w') as f:
            f.write("")

    return paths


def save_round_data(i: int, paths: dict, insights: str, plans: str, observations: str, market_data_result: str):
    def prepend(file_path, content):
        with open(file_path, 'r+') as f:
            existing = f.read()
            f.seek(0)
            f.write(content + '\n' + existing)

    separator = lambda label: f"{'='*20} Round {i} {'='*20}\n"

    prepend(paths["insights"], separator("Insights") + insights)
    prepend(paths["plans"], separator("Plans") + plans)
    prepend(paths["observations"], separator("Observations") + observations)
    prepend(paths["market_data"], market_data_result)

    with open(paths["market_data"], 'r') as f:
        return f.read()


def update_plot(fig, axs, i, p_m, q_m, pi_m, price_history, quantity_history, profit_history, time_history, model_name, start_time, save_path):
    for ax in axs:
        ax.clear()

    axs[0].plot(time_history, price_history, marker='o',color='blue', label="Agent Price")
    axs[0].axhline(y=p_m[0], color='g', linestyle='--', label='Monopoly Price')
    axs[0].fill_between(range(0, max(10, i + 2)), p_m[0]* 0.95, p_m[0]* 1.05, color='green', alpha=0.5, label='Convergence Area')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_history, quantity_history, marker='s', color='green', label="Quantity")
    axs[1].axhline(y=q_m[0], color='orange', linestyle='--', label='Monopoly Quantity')
    axs[1].set_ylabel('Quantity')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time_history, profit_history, marker='^', color='red', label="Profit")
    axs[2].axhline(y=pi_m[0], color='purple', linestyle='--', label='Monopoly Profit')
    axs[2].set_ylabel('Profit')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)

    if has_converged_to_price(price_history, p_m[0]):
        converged = 'TRUE :D'
    else:
        converged = ' FALSE :('

    axs[0].set_title(f'Monopoly Experiment {model_name} ({start_time}) | Converged: {converged}')
    
    display(fig)
    clear_output(wait=True)
    fig.savefig(save_path)
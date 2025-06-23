import os
import re
import json
import asyncio
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from mistralai import Mistral
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

API_KEY = os.getenv("MISTRAL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "magistral-small-2506")

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

def get_last_100_rounds(market_data: str) -> str:
    """
    Extract the last 100 rounds of market data from a string.
    """
    rounds = []
    current_round = []

    for line in market_data.split('\n'):
        if re.match(r'^Round \d+:', line.strip()):
            if current_round:
                rounds.append(current_round)
                current_round = []
        current_round.append(line)

    # Don't forget to add the last round
    if current_round:
        rounds.append(current_round)

    # Keep the last 100 rounds
    last_100_rounds = rounds[:100]

    # Flatten to a list of lines
    flattened_lines = [line for group in last_100_rounds for line in group]

    return "\n".join(flattened_lines)

def update_plot(
    fig, axs,
    p_m, q_m, pi_m, alpha,
    price_history, quantity_history, profit_history, time_history,
    model_name, start_time, save_path, prompt_number, nash_price=False, fill_convergende_range=True,
    title_experiment_type="Monopoly"
):
    n_agents = len(price_history)
    firm_names = list(price_history.keys())
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'brown', 'magenta', 'gray']

    for ax in axs:
        ax.clear()

    axs[0].axhline(y=p_m, color='black', linestyle='--', alpha=0.5, label='PM')
    if fill_convergende_range:
    #fill in 0.95 and 1.05 from p_m in green area
        axs[0].fill_between(
            time_history,
            p_m * 0.95,
            p_m * 1.05,
            color='green', alpha=0.4, label='Convergence Range'
        )
    if nash_price:
        axs[0].axhline(y=nash_price, color='green', linestyle='--', alpha=0.5, label='PN')
    for j, firm in enumerate(firm_names):
        axs[0].plot(time_history, price_history[firm], color=colors[j % len(colors)], label=firm)
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].axhline(y=q_m, color='black', linestyle='--', alpha=0.5, label='QM')
    for j, firm in enumerate(firm_names):
        axs[1].plot(time_history, quantity_history[firm], color=colors[j % len(colors)], label=firm)
    axs[1].set_ylabel('Quantity')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].axhline(y=pi_m, color='black', linestyle='--', alpha=0.5, label='PiM')
    for j, firm in enumerate(firm_names):
        axs[2].plot(time_history, profit_history[firm], color=colors[j % len(colors)], label=firm)
    axs[2].set_ylabel('Profit')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)

    axs[0].set_title(f'{title_experiment_type} Experiment | Prompt: P{prompt_number} | Alpha: {alpha} | {model_name} | Run: {start_time}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # optional

    display(fig)
    clear_output(wait=True)
    fig.savefig(save_path)
    fig.savefig(save_path, bbox_inches='tight')  # fix cut-off issue


def update_plot_duopoloy(fig, axs,
    p_m, q_m, pi_m, alpha, nash_price,
    price_history, quantity_history, profit_history, time_history,
    model_name, start_time, save_path, prompt_number, display_notebook=True
):
    firm_names = list(price_history.keys())
    firm1, firm2 = firm_names[:2]
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'brown', 'magenta', 'gray']

    # === Create figure with GridSpec layout ===
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])
    
    # Left column (time series)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_quantity = fig.add_subplot(gs[1, 0])
    ax_profit = fig.add_subplot(gs[2, 0])

    # Right column (price trajectory and profit trajectory)
    ax_price_traj = fig.add_subplot(gs[0:2, 1])  # spans two rows
    ax_profit_traj = fig.add_subplot(gs[2, 1])   # bottom right

    # === Plot Price Time Series ===
    ax_price.axhline(y=p_m, color='black', linestyle='--', alpha=0.5, label='PM')
    ax_price.axhline(y=nash_price, color='green', linestyle='--', alpha=0.5, label='PN')
    for j, firm in enumerate(firm_names):
        ax_price.plot(time_history, price_history[firm], color=colors[j % len(colors)], label=firm)
    ax_price.set_ylabel('Price')
    ax_price.legend()
    ax_price.grid(True)

    # === Plot Quantity Time Series ===
    ax_quantity.axhline(y=q_m, color='black', linestyle='--', alpha=0.5, label='QM')
    for j, firm in enumerate(firm_names):
        ax_quantity.plot(time_history, quantity_history[firm], color=colors[j % len(colors)], label=firm)
    ax_quantity.set_ylabel('Quantity')
    ax_quantity.legend()
    ax_quantity.grid(True)

    # === Plot Profit Time Series ===
    ax_profit.axhline(y=pi_m, color='black', linestyle='--', alpha=0.5, label='PiM')
    for j, firm in enumerate(firm_names):
        ax_profit.plot(time_history, profit_history[firm], color=colors[j % len(colors)], label=firm)
    ax_profit.set_ylabel('Profit')
    ax_profit.set_xlabel('Time')
    ax_profit.legend()
    ax_profit.grid(True)

    # === Price Trajectory Plot ===
    p1 = price_history[firm1][-50:]
    p2 = price_history[firm2][-50:]
    n_points = len(p1)
    cmap = cm.get_cmap('Blues')
    for i in range(1, n_points):
        ax_price_traj.plot(p1[i-1:i+1], p2[i-1:i+1], color=cmap(i / n_points), linewidth=2)
    ax_price_traj.axvline(p_m, color='green', linestyle=':', linewidth=1)
    ax_price_traj.axhline(p_m, color='green', linestyle=':', linewidth=1)
    ax_price_traj.axvline(nash_price, color='red', linestyle='--', linewidth=1)
    ax_price_traj.axhline(nash_price, color='red', linestyle='--', linewidth=1)
    ax_price_traj.set_xlabel('Firm 1 Price')
    ax_price_traj.set_ylabel('Firm 2 Price')
    ax_price_traj.set_title('Price Trajectory (Last 50 Rounds)')
    ax_price_traj.grid(True)

    # === Profit Trajectory Plot (Difference vs Sum) ===
    pi1 = np.array(profit_history[firm1])
    pi2 = np.array(profit_history[firm2])
    pi_sum = pi1 + pi2
    pi_diff = pi1 - pi2
    cmap = cm.get_cmap('Oranges')
    for i in range(1, n_points):
        ax_profit_traj.plot(pi_diff[i-1:i+1], pi_sum[i-1:i+1], color=cmap(i / n_points), linewidth=2)
    ax_profit_traj.axhline(pi_m * 2, color='green', linestyle=':', linewidth=1)
    ax_profit_traj.set_xlabel(r'$\pi_1 - \pi_2$')
    ax_profit_traj.set_ylabel(r'$\pi_1 + \pi_2$')
    ax_profit_traj.set_title('Profit Trajectory')
    ax_profit_traj.grid(True)

    # === Title ===
    fig.suptitle(f'Duopoly Experiment | Prompt: P{prompt_number} | Alpha: {alpha} | {model_name} | Run: {start_time}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if display_notebook:
        display(fig)
        clear_output(wait=True)
    fig.savefig(save_path, bbox_inches='tight')


def create_output_paths(sub_path: str, model_name: str, firm_names: list[str]) -> dict:
    start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = f"results/{sub_path}/{start_time}_experiment_{model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "start_time": start_time,
        "output_dir": output_dir,
        "plot": os.path.join(output_dir, "experiment_plots"),
        "firms": {}  # nested dict for each agent
    }

    for firm in firm_names:
        firm_dir = os.path.join(output_dir, f"{firm}")
        os.makedirs(firm_dir, exist_ok=True)

        paths["firms"][firm] = {
            "observations": os.path.join(firm_dir, "OBSERVATIONS.txt"),
            "insights": os.path.join(firm_dir, "INSIGHTS.txt"),
            "plans": os.path.join(firm_dir, "PLANS.txt"),
            "market_data": os.path.join(firm_dir, "MARKET_DATA.txt"),
        }

        # Initialize empty files
        for file in paths["firms"][firm].values():
            with open(file, 'w') as f:
                f.write("")
        
    # Create a plot directory
    os.makedirs(paths['plot'], exist_ok=True)

    return paths


def save_round_data(i: int, paths: dict, firm_responses: dict, prices: list, quantities: list, profits: list, firm_names: list[str]):
    def prepend(file_path, content):
        with open(file_path, 'r+') as f:
            existing = f.read()
            f.seek(0)
            f.write(content + '\n' + existing)

    for idx, firm in enumerate(firm_names):
        response = firm_responses[firm]
        firm_path = paths["firms"][firm]
        separator = f"{'='*20} Round {i} {'='*20}\n"

        obs = f"{separator}{response['observations']}"
        insights = f"{separator}{response['insights']}"
        plans = f"{separator}{response['plans']}"
        comp_prices = ', '.join(
            # f"{other_firm}: {firm_responses[other_firm]['chosen_price']:.2f}"
            f"{firm_responses[other_firm]['chosen_price']:.2f}"
            for other_firm in firm_names if other_firm != firm
        )

        market_data_result = (
            f"Round {i}: \n"
            f" - My price: {prices[idx]:.2f}\n"
            f" - Competitor's price: {comp_prices}\n"
            f" - My quantity sold: {quantities[idx]:.2f}\n"
            f" - My profit earned: {profits[idx]:.2f}\n"
        )

        prepend(firm_path["observations"], obs)
        prepend(firm_path["insights"], insights)
        prepend(firm_path["plans"], plans)
        prepend(firm_path["market_data"], market_data_result)

        # return last 100 rounds of this agent's market data for use in prompt
        with open(firm_path["market_data"], 'r') as f:
            last_data = f.read()
            firm_responses[firm]["last_market_data"] = get_last_100_rounds(last_data)

    return firm_responses




SYSTEM_PROMPT = """Respond only with a JSON object with this schema:
{
  "observations": string,
  "plans": string,
  "insights": string,
  "chosen_price": float
}"""

MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 1  # delay between retries (can also back off if needed)

async def call_firm(firm_name: str, prefix:str, prompt: str):
    async with Mistral(api_key=API_KEY) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.chat.complete_async(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": prefix},
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(response.choices[0].message.content)
                insights = parsed['insights']
                observations = parsed['observations']
                plans = parsed['plans']
                chosen_price = parsed['chosen_price']
                if any( not isinstance(v, str) for v in [insights, observations, plans]):
                    raise ValueError("Insights, observations, and plans must be strings.")
                if not isinstance(chosen_price, (float, int)):
                    raise ValueError("Chosen price must be a float or int.")

                return {"firm": firm_name,
                        "response": {
                                    "insights": insights,
                                    "observations": observations,
                                    "plans": plans,
                                    "chosen_price": chosen_price, 
                        }
                }
            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {"firm": firm_name, "error": f"Failed after {MAX_RETRIES} attempts: {str(e)}"}
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)  # optional backoff


async def simulate_n_firms_round(agent_prompt_dict: dict):
    tasks = [
        call_firm(firm_name, prompt['prefix'], prompt['prompt'])
        for firm_name, prompt in agent_prompt_dict.items()
    ]
    results = await asyncio.gather(*tasks)
    return results



def create_gif_from_pngs(path: str, gif_name="000_animation.gif", duration=200):
    """
    Create a GIF from all PNG images in the given directory.

    Parameters:
    - path (str): Directory containing PNG frames and where the GIF will be saved.
    - gif_name (str): Name of the output GIF file.
    - duration (int): Duration between frames in milliseconds.

    Returns:
    - str: Full path to the created GIF.
    """
    # Get sorted list of all PNG files
    png_files = sorted([
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(".png")
    ])

    if not png_files:
        raise FileNotFoundError("No PNG files found in the specified directory.")

    # Open images
    frames = [Image.open(png) for png in png_files]

    # Save GIF
    gif_path = os.path.join(path, gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

    print(f"GIF saved to: {gif_path}")
    return gif_path


def plot_duopoly_results_from_df(df, p_nash, p_m, pi_nash, pi_m, title="Figure 2: Duopoly Experiment Results", save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    # === Panel 1: Price comparison ===
    axs[0].scatter(df.loc[df['prompt']==1,'p1'], df.loc[df['prompt']==1,'p2'], color='tab:blue', marker='s', label='P1 vs. P1')
    axs[0].scatter(df.loc[df['prompt']==2,'p1'], df.loc[df['prompt']==2,'p2'], color='tab:orange', marker='^', label='P2 vs. P2')

    # Axis setup
    axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Reference lines
    axs[0].axvline(p_nash, color='red', linestyle='--', linewidth=1)
    axs[0].axhline(p_nash, color='red', linestyle='--', linewidth=1)
    axs[0].axvline(p_m, color='green', linestyle=':', linewidth=1)
    axs[0].axhline(p_m, color='green', linestyle=':', linewidth=1)

    # Axis annotations (external, aligned with ticks)
    axs[0].annotate(r'$p^{Nash}$', xy=(p_nash, axs[0].get_ylim()[0]),
                    xytext=(0, -5), textcoords='offset points',
                    ha='center', va='top', color='red', fontsize=10)
    axs[0].annotate(r'$p^{Nash}$', xy=(axs[0].get_xlim()[0], p_nash),
                    xytext=(-5, 0), textcoords='offset points',
                    ha='right', va='center', color='red', fontsize=10)

    axs[0].annotate(r'$p^M$', xy=(p_m, axs[0].get_ylim()[0]),
                    xytext=(0, -5), textcoords='offset points',
                    ha='center', va='top', color='green', fontsize=10)
    axs[0].annotate(r'$p^M$', xy=(axs[0].get_xlim()[0], p_m),
                    xytext=(-5, 0), textcoords='offset points',
                    ha='right', va='center', color='green', fontsize=10)

    axs[0].set_xlabel('Firm 1 average price (over periods 251–300)')
    axs[0].set_ylabel('Firm 2 average price (over periods 251–300)')
    axs[0].set_title("P1 Compared to P2: Pricing Behavior")
    axs[0].grid(False)

    # === Panel 2: Profit comparison ===
    df['pi_sum'] = df['pi_1'] + df['pi_2']
    axs[1].scatter(df.loc[df['prompt']==1,'pi_delta'], df.loc[df['prompt']==1,'pi_sum'], color='tab:blue', marker='s', label='P1 vs. P1')
    axs[1].scatter(df.loc[df['prompt']==2,'pi_delta'], df.loc[df['prompt']==2,'pi_sum'], color='tab:orange', marker='^', label='P2 vs. P1')

    # x_vals = np.linspace(-20, 20, 200)
    # axs[1].plot(x_vals, 2 * pi_nash + np.abs(x_vals), 'r--', label=r'$\pi_1 = \pi^{Nash}$ / $\pi_2 = \pi^{Nash}$')
    # axs[1].text(-10, 2 * pi_nash + 1, r'$\pi_1 = \pi^{Nash}$', color='red', fontsize=10)
    # axs[1].text(5, 2 * pi_nash + 1, r'$\pi_2 = \pi^{Nash}$', color='red', fontsize=10)

    axs[1].axhline(pi_m, color='green', linestyle=':', linewidth=1)
    axs[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axs[1].text(df['pi_delta'].min()*1.05, pi_m, r'$\pi^M$', color='green', fontsize=10)

    # axs[1].set_xlim(-22, 22)
    # axs[1].set_ylim(40, 70)
    axs[1].set_xlabel('Average difference in profits $\pi_1 - \pi_2$ (over periods 251–300)')
    axs[1].set_ylabel('Average sum of profits $\pi_1 + \pi_2$ (over periods 251–300)')
    axs[1].set_title("P1 Compared to P2: Profits Earned")
    axs[1].grid(False)

    # === Legend outside below both plots ===
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def make_df_from_results(results_path, model_name):
   summarized_results = []
   for prompt in os.listdir(results_path):  # Just to ensure the directory is created if it doesn't exist
         if 'prompt' not in prompt:
            continue
         prmt = 1 if 'prompt_1' in prompt else 2
         if 'other' in prompt:
            continue
         alph = int(prompt.split('alpha_')[-1])
         alph = 3.2 if alph == 3 else alph
         for experiment in os.listdir(f"{results_path}/{prompt}"):
            if model_name not in experiment:
                continue
            with open(f"{results_path}/{prompt}/{experiment}/results.json", 'r') as f:
               results = json.load(f)
            p1 = np.mean(np.array(results['price_history']['firm_1'][-50:])/alph)
            p2 = np.mean(np.array(results['price_history']['firm_2'][-50:])/alph)
            pi_1 = np.mean(np.array(results['profit_history']['firm_1'][-50:])/alph)
            pi_2 = np.mean(np.array(results['profit_history']['firm_2'][-50:])/alph)
            pi_delta = pi_1 - pi_2

            summarized_results.append({
               "prompt": prmt,
               "alpha": alph,
               "experiment": experiment,
               "p1": p1,
               "p2": p2,
               "pi_1": pi_1,
               "pi_2": pi_2,
               "pi_delta": pi_delta
            })

   return pd.DataFrame(summarized_results)
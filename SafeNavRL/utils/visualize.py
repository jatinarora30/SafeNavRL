import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Visualize:
    def __init__(self, log_root="logs", results_dir="results"):
        self.log_root = log_root
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # Simple plotting style (similar to your image)
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "font.size": 11,
            "lines.linewidth": 2,
        })

    def collect_data(self):
        """Collect all environment–algorithm CSV file paths."""
        data = {}
        if not os.path.exists(self.log_root):
            print(f"Log directory '{self.log_root}' not found.")
            return data

        for env in os.listdir(self.log_root):
            env_path = os.path.join(self.log_root, env)
            if not os.path.isdir(env_path):
                continue
            data[env] = {}
            for algo in os.listdir(env_path):
                algo_path = os.path.join(env_path, algo)
                if not os.path.isdir(algo_path):
                    continue
                csv_files = [
                    os.path.join(algo_path, f)
                    for f in os.listdir(algo_path)
                    if f.endswith(".csv")
                ]
                if csv_files:
                    data[env][algo] = csv_files
        return data

    def _load_runs(self, csv_files):
        """Load CSV runs, align by episode, and return mean/std arrays."""
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if not dfs:
            return None

        min_len = min(len(df) for df in dfs)
        dfs = [df.iloc[:min_len] for df in dfs]
        episodes = dfs[0]["Episode"].values
        rewards = np.array([df["Reward"].values for df in dfs])
        costs = np.array([df["Cost"].values for df in dfs])

        stats = {
            "episodes": episodes,
            "reward_mean": rewards.mean(axis=0),
            "reward_std": rewards.std(axis=0),
            "cost_mean": costs.mean(axis=0),
            "cost_std": costs.std(axis=0),
        }
        return stats

    def plot_env(self, env, algo_data):
        """Generate reward and cost plots for a given environment."""
        save_dir = os.path.join(self.results_dir, env)
        os.makedirs(save_dir, exist_ok=True)

        # ---- REWARD PLOT ----
        plt.figure()
        for algo, csv_files in algo_data.items():
            stats = self._load_runs(csv_files)
            if stats is None:
                continue
            plt.plot(stats["episodes"], stats["reward_mean"], label=algo)
            plt.fill_between(
                stats["episodes"],
                stats["reward_mean"] - stats["reward_std"],
                stats["reward_mean"] + stats["reward_std"],
                alpha=0.2,
            )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{env}_reward.png"))
        plt.close()

        # ---- COST PLOT ----
        plt.figure()
        for algo, csv_files in algo_data.items():
            stats = self._load_runs(csv_files)
            if stats is None:
                continue
            plt.plot(stats["episodes"], stats["cost_mean"], label=algo)
            plt.fill_between(
                stats["episodes"],
                stats["cost_mean"] - stats["cost_std"],
                stats["cost_mean"] + stats["cost_std"],
                alpha=0.2,
            )
        plt.xlabel("Episode")
        plt.ylabel("Cost")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{env}_cost.png"))
        plt.close()

  

    def plot_all(self):
        """Scan all envs and plot rewards and costs."""
        all_data = self.collect_data()
        if not all_data:
            print("No logs found.")
            return

        for env, algo_data in all_data.items():
            if algo_data:
                self.plot_env(env, algo_data)


# # === Example ===
# if __name__ == "__main__":
#     vis = Visualize(log_root="logs", results_dir="results")
#     vis.plot_all()

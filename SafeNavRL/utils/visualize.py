import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Visualize:
    def __init__(self):
      
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

        self.log_root = os.path.join(project_root, "logs")
        self.results_dir = os.path.join(project_root, "results")

        os.makedirs(self.results_dir, exist_ok=True)

      
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "font.size": 11,
            "lines.linewidth": 2,
        })

  

    def collect_data(self):
        """Collect all train/test CSV files under each environment and algorithm."""
        data = {}
        if not os.path.exists(self.log_root):
            print(f"Log directory '{self.log_root}' not found.")
            return data

        for mode in ["train", "test"]:
            mode_path = os.path.join(self.log_root, mode)
            if not os.path.isdir(mode_path):
                continue

            for env in os.listdir(mode_path):
                env_path = os.path.join(mode_path, env)
                if not os.path.isdir(env_path):
                    continue
                data.setdefault(env, {}).setdefault(mode, {})

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
                        data[env][mode][algo] = csv_files
        return data

   

    def _load_runs(self, csv_files):
        """Load multiple CSV runs and return averaged statistics."""
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
            "reward_final_mean": rewards[:, -1].mean(),
            "reward_final_std": rewards[:, -1].std(),
            "cost_final_mean": costs[:, -1].mean(),
            "cost_final_std": costs[:, -1].std(),
        }
        return stats



    def plot_env_mode(self, env, mode, algo_data):
        """Plot reward and cost for one environment in a given mode (train/test)."""
        save_dir = os.path.join(self.results_dir, env)
        os.makedirs(save_dir, exist_ok=True)

        summary_data = []

        # ---- REWARD ----
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
            summary_data.append({
                "Mode": mode,
                "Algo": algo,
                "Reward Mean": stats["reward_final_mean"],
                "Reward Std": stats["reward_final_std"],
                "Cost Mean": stats["cost_final_mean"],
                "Cost Std": stats["cost_final_std"],
            })

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"{env} ({mode.upper()}) - Reward")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{mode}_reward.png"))
        plt.close()

    
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
        plt.title(f"{env} ({mode.upper()}) - Cost")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{mode}_cost.png"))
        plt.close()

        # ---- PRINT MEAN/STDEV SUMMARY ----
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            print(f"\nSummary for {env} ({mode.upper()}):")
            print(df_summary.to_string(index=False))


    def plot_all(self):
        """Scan all environments, plot train/test results if available."""
        all_data = self.collect_data()
        if not all_data:
            print("No logs found.")
            return

        for env, modes in all_data.items():
            for mode, algo_data in modes.items():
                if algo_data:
                    self.plot_env_mode(env, mode, algo_data)



if __name__ == "__main__":
    vis = Visualize()
    vis.plot_all()

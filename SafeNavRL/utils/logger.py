import os
import csv

class Logger:
    def __init__(self, env, algo, type, continueRun=False):
        
        self.createDir("logs")
        self.type = type
        self.typePath = os.path.join("logs", self.type)
        self.envPath = os.path.join(self.typePath, env)
        self.algoPath = os.path.join(self.envPath, algo)
        self.createDir(self.typePath)
        self.createDir(self.envPath)
        self.createDir(self.algoPath)

        # Get existing CSV files for this algorithm
        self.fileList = [f for f in os.listdir(self.algoPath) if f.endswith(".csv")]
        self.lastRun = len(self.fileList)

        if continueRun and self.lastRun > 0:
            # Continue from last file
            self.run_id = self.lastRun
            self.csv_path = os.path.join(self.algoPath, f"{algo}_run{self.run_id}.csv")
            if os.path.exists(self.csv_path):
                with open(self.csv_path, "r") as f:
                    self.episode = sum(1 for _ in f) - 1  # header line
                print(f"Continuing run {self.run_id} from episode {self.episode}.")
            else:
                self.episode = 0
                self._create_csv()
        else:
            # Start a new run file
            self.run_id = self.lastRun + 1
            self.csv_path = os.path.join(self.algoPath, f"{algo}_run{self.run_id}.csv")
            self._create_csv()
            self.episode = 0
            print(f"Starting new run {self.run_id}.")

        self.name = f"{env}_{algo}_run{self.run_id}"

    def createDir(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        except Exception as e:
            print(f"Error creating directory '{path}': {e}")

    def _create_csv(self):
        """Create a fresh CSV file with headers."""
        with open(self.csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Cost"])

    def log(self, episode, reward, cost):
        """Append a single episode’s reward and cost."""
        current_episode = self.episode + episode
        with open(self.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([current_episode, reward, cost])


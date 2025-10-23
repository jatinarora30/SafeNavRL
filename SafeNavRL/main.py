from envs.env1 import Env1
from utils.logger import Logger
from algos.ppo import PPO
import argparse
import os
import torch
import re


class Evaluate:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Evaluate RL agent performance.")
        parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
        parser.add_argument("--type", type=str, default="train", help="train or test")
        parser.add_argument("--env", type=str, default="Env1", help="Environment name")
        parser.add_argument("--algo", type=str, default="PPO", help="Algorithm used")
        parser.add_argument("--continue_run", action="store_true", help="Continue last run")

        args = parser.parse_args()

        self.episodes = args.episodes
        self.type = args.type.lower()
        self.env_name = args.env
        self.algo_name = args.algo
        self.continue_run = args.continue_run
        print("Conn",self.continue_run)
        self.renderMode = "human" if self.type == "test" else None

      
        current_dir = os.path.dirname(os.path.abspath(__file__))         
        project_root = os.path.abspath(os.path.join(current_dir, ".."))   

        self.models_root = os.path.join(project_root, "models")
        self.logs_root = os.path.join(project_root, "logs")

        
        self.createDir(self.models_root)
        self.createDir(self.logs_root)

        
        self.env = self.init_env()

       
        log_subdir = os.path.join(self.logs_root, self.type, self.env_name, self.algo_name)
        os.makedirs(log_subdir, exist_ok=True)

        
        os.chdir(project_root)
        self.logger = Logger(self.env_name, self.algo_name, self.type, self.continue_run)
        os.chdir(current_dir)

     
        self.env_dir = os.path.join(self.models_root, self.env_name)
        self.algo_dir = os.path.join(self.env_dir, self.algo_name)
        self.createDir(self.env_dir)
        self.createDir(self.algo_dir)

    
        self.agent = self.init_algo()


    def init_env(self):
        if self.env_name == "Env1":
            return Env1(self.renderMode)
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

    def createDir(self, path):
        os.makedirs(path, exist_ok=True)

    def init_algo(self):
        if self.algo_name == "PPO":
            state_dim = self.env.obsDim
            action_dim = self.env.actionDim
            algo = PPO(
                eps=0.2,
                stateDim=state_dim,
                actionDim=action_dim,
                hiddenLayersPolicy=[64, 64],
                hiddenLayersValue=[64, 64],
                gamma=0.99,
                lam=0.95,
                policyLr=3e-4,
                valueLr=1e-3,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algo_name}")

        # Load existing model if continuing or testing
        if self.continue_run or self.type == "test":
            try:
                algo.loadModel(self.algo_dir)
                print(f"Loaded existing model from {self.algo_dir}")
            except FileNotFoundError:
                print("No existing model found, starting fresh.")
        return algo

   

    def run(self):
        print(f"Starting {self.type.upper()} for {self.episodes} episodes using {self.algo_name} on {self.env_name}")
        print(f"Saving models to: {self.algo_dir}")


        current_dir = os.getcwd()
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        os.chdir(project_root)

        for ep in range(1, self.episodes + 1):
            if self.type == "train":
                total_reward, total_cost = self.agent.train(self.env)
                self.agent.saveModel(self.algo_dir)
            else:
                total_reward, total_cost = self.agent.test(self.env)

            self.logger.log(ep, total_reward, total_cost)
            print(f"Episode {ep:03d} | Reward: {total_reward:.3f} | Cost: {total_cost:.3f}")

        os.chdir(current_dir)

        print(f"Finished {self.type.upper()} run.")
        print(f"Models saved at: {self.algo_dir}")
        print(f"Logs saved at:   {os.path.join(project_root, 'logs')}")


if __name__ == "__main__":
    eval = Evaluate()
    eval.run()

from envs.env1 import Env1
from utils.logger import Logger
import argparse

class Evaluate:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Evaluate RL agent performance.")
        parser.add_argument( "--episodes", type=int,default=10,help="Number of episodes to evaluate")
        parser.add_argument( "--type", type=str, default="train",help="train or test")
        parser.add_argument( "--env", type=str, default="Env1",help="Envrinoment ")
        parser.add_argument( "--algo", type=str, default="PPO",help="Algorithm used ")
        parser.add_argument( "--continue_run", type=bool, default=False,help=" To continue last run")
        args = parser.parse_args()

        self.episodes = args.episodes
        self.type=args.type
        self.envName=args.env
        self.algoName=args.algo
        self.continue_run=args.continue_run
        self.init_env()
        self.logger=Logger(self.env,self.algo,self.continue_run)
    
    def init_env(self):
        if self.envName=="Env1":
            self.env=Env1()

    def init_algo(self):
        if self.algoName=="PPO":
            self.env=Env1()



if __name__ == "__main__":
    eval = Evaluate()
    

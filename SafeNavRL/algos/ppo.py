from .basealgos import BaseAlgo
from utils.networks import Actor,Critic
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPO(BaseAlgo):


    def __init__(self,eps,stateDim,actionDim,hiddenLayersPolicy,hiddenLayersValue,gamma,lam,policyLr,valueLr):
        self.eps=eps
        self.gamma=gamma
        self.lam=lam

        self.policyNet=Actor(stateDim,actionDim,hiddenLayersPolicy)
        self.valueNet= Critic(stateDim,1,hiddenLayersValue)

        self.policyOptimizer=optim.Adam(self.policyNet.parameters(),lr=policyLr)
        self.valueOptimizer=optim.Adam(self.valueNet.parameters(),lr=valueLr)
        self.mseLoss=nn.MSELoss()

    def computeGae(self, rewards, values):
        advantages, returns = [], []
        gae = 0
        values = list(values) + [torch.tensor(0.0)]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return torch.stack(advantages), torch.stack(returns)

    def update(self, trajectories):
        # Unzip trajectories
        states, actions, log_probs, rewards = zip(*trajectories)
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_logps = torch.stack(log_probs).detach()

        # Convert rewards to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Get value predictions
        values = self.valueNet(states).squeeze(-1).detach()
        
        # Compute advantages and returns
        advantages, returns = self.computeGae(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # PPO update steps
            dist = self.policyNet.getDist(states)
            new_logps = dist.log_prob(actions)
            ratio = (new_logps - old_logps).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policyOptimizer.zero_grad()
            policy_loss.backward()
            self.policyOptimizer.step()

        value_preds = self.valueNet(states).squeeze(-1)
        value_loss = self.mseLoss(value_preds, returns)
        self.valueOptimizer.zero_grad()
        value_loss.backward()
        self.valueOptimizer.step()

    def train(self, env):
        state,_= env.reset()
        done = False
        total_reward, total_cost = 0.0, 0.0
        trajectory = []

        while not np.any(done):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob = self.policyNet.act(state_tensor)
            next_state, reward, cost, done = env.step(action.detach().numpy())

            reward_scalar = float(np.mean(reward))
            cost_scalar = float(np.mean(cost))
            done = np.any(done)

            trajectory.append((state_tensor, action, log_prob, reward_scalar))
            total_reward += reward_scalar
            total_cost += cost_scalar
            state = next_state

        self.update(trajectory)
        return total_reward, total_cost
    
    def test(self, env):
        state,_ = env.reset()
        done = False
        total_reward, total_cost = 0.0, 0.0

        while not np.any(done):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob = self.policyNet.act(state_tensor)
            next_state, reward, cost, done= env.step(action.detach().numpy())

            reward_scalar = float(np.mean(reward))
            cost_scalar = float(np.mean(cost))
            done = np.any(done)

            total_reward += reward_scalar
            total_cost += cost_scalar
            state = next_state
        return total_reward, total_cost
    
    

    def saveModel(self,path):
        torch.save(self.policyNet.state_dict(),os.path.join(path, f"policy.pth"))
        torch.save(self.valueNet.state_dict(), os.path.join(path, f"value.pth"))
    

    def loadModel(self, path):
        self.policyNet.load_state_dict(torch.load(os.path.join(path,f"policy.pth")))
        self.valueNet.load_state_dict(torch.load(os.path.join(path,f"value.pth")))
       
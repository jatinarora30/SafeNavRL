import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal



class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(Actor, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.mean=nn.Linear(hidden_layers[-1],output_dim)
        self.mean.weight.data.mul(0.1)
        self.mean.bias.data.mul(0.0)
        self.log_std=nn.Parameter(torch.zeros(output_dim))

        

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def getDist(self,state):
        mean,std=self.forward(state)
        std=std.clamp(min=0.01,max=100)
        cov=torch.diag_embed(std.pow(2))

        if torch.isnan(mean).any() or torch.isnan(cov).any():
            return None

        return MultivariateNormal(mean,covariance_matrix=cov)
    
    
    def act(self, state):
        dist = self.getDist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(Critic, self).__init__()

        layers = []
  
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())

    
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())

      
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



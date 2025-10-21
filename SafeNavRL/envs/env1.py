import safety_gymnasium
from baseenv import BaseEnv

class Env1(BaseEnv):
    

    def __init__(self, renderMode: str = "None"):
        super().__init__(renderMode)
        self.envName = "SafetyCarGoal1-v0"
        self.numEnvs = 1

        self.env = safety_gymnasium.vector.make(
            self.envName, render_mode=self.renderMode, num_envs=self.numEnvs
        )

        self.obsDim=self.env.single_observation_space.shape[0]
        self.actionDim=self.env.single_action_space.shape[0]

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return obs, reward, cost, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def close(self):
        self.env.close()


env=Env1()
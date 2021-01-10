import logging
import numpy as np

class Agent(object):

    def __init__(self):
        pass

    def get_action(self):
        pass

    def collect_transitions(self):
        pass

    def update(self):
        pass

    def learn(self):
        pass

    def save(self):
        pass

    def play(self):
        obs = self.env.reset()
        obs = obs.astype(np.float32)
        score = 0
        while True:
            action = self.get_action(obs)
            obs, rewards, dones, _ = self.env.step(action)
            obs = obs.astype(np.float32)
            score += rewards
            self.env.render()
            if dones > 0:
                print("Score:", score)
                obs = self.env.reset(dones)
                obs = obs.astype(np.float32)
                score = 0

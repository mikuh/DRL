import tensorflow as tf
import numpy as np
from Agent import Agent


class DoubleDQNAgent(Agent):

    def __init__(self):

        self.lr = 0.001
        self.gamma = 0.99

        self.dqn_net = DQN()
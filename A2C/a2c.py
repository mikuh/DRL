import tensorflow as tf
import gym
import numpy as np
from typing import Any, List, Sequence, Tuple


class Actor(object):
    """Actor for the A2C: Policy gradient
     """

    def __init__(self, embedding_net, policy_net):
        self._embedding_net = embedding_net
        self._policy_net = policy_net


class AdvantageActorCritic(tf.keras.Model):

    def __init__(self,
                 embedding_net: tf.keras.Layer,
                 policy_net: tf.keras.Layer,
                 value_net: tf.keras.Layer,
                 env: gym.Env,
                 configs: dict):
        self._embedding_net = embedding_net
        self._policy_net = policy_net
        self._value_net = value_net
        self._env = env
        self._configs = configs

    def call(self, inputs: tf.Tensor):
        x = self._embedding_net(inputs)
        return self._policy_net(x), self._value_net(x)

    def get_action(self, state) -> int:
        pass

    def _run_episode(self):
        pass

    def tf_env_step(self, action: np.ndarray) -> List[tf.Tensor]:
        state, reward, done, _ = self._env.step(action)
        state, reward, done = state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)
        return tf.numpy_function(self._env.step, [action], [tf.float32, tf.int32, tf.int32])

    def _collect_roolouts(self):
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = []
        next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        dones = []
        state = self._env.reset()
        episode = 0
        score = 0
        for t in range(self._configs["roolout"]):
            action = self.get_action(state)
            next_state, reward, done, _ = self.tf_env_step(action)
            score += reward

            states.write(t, state.astype(np.float32))
            actions.write(t, action)
            next_states.write(t, next_state)
            rewards.write(t, reward)
            dones.write(t, done)

    def learn(self):
        state = self._env.reset()
        episode = 0
        score = 0
        while True:
            states = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            actions = []
            next_states = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            # values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            rewards = []
            dones = []

            for _ in range(_):
                pass


def actor_loss(self, n_step_td_target, value):
    pass

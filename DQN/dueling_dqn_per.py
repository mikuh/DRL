"""
use prioritized experience replay
"""
import tensorflow as tf
import numpy as np
from collections import deque
import gym
import random

from Agent import Agent
from Layers import DenseEmbeddingNet, ValueNet, QNet


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class Memory(object):
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

class DQN(tf.keras.Model):

    def __init__(self, embedding_net, value_net, q_net):
        super().__init__()
        self.embedding_layer = embedding_net
        self.state_value = value_net
        self.q_value = q_net

    def call(self, states):
        embedding = self.embedding_layer(states)
        self.v = self.state_value(embedding)
        self.a = self.q_value(embedding)
        value = self.v + self.a - tf.math.reduce_mean(self.a)
        return value


class DQNAgent(Agent):

    def __init__(self, env, set_summary=True):
        super().__init__()
        self.env = env

        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 0

        self.dqn_net = DQN(embedding_net=DenseEmbeddingNet(), value_net=ValueNet(),
                           q_net=QNet(env))
        self.target_net = DQN(embedding_net=DenseEmbeddingNet(), value_net=ValueNet(),
                              q_net=QNet(env))

        self.opt = tf.keras.optimizers.Adam(self.lr)

        self.action_size = self.env.action_space.n
        self.batch_size = 64

        self.memory = Memory(capacity=2000)

        self.step = 0
        self.target_net_update_step = 20
        self.episode = 0
        self.score = 0
        self.set_summary = set_summary
        if set_summary:
            self.checkpoint_summary_setting("dueling_dqn_per", self.opt, self.dqn_net)

    def get_action(self, state):
        q_value = self.dqn_net(np.array([state], dtype=np.float32))[0]

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action

    def sampling(self):
        mini_batch, idxs, IS_weight = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition in mini_batch:
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            dones.append(transition[4])

        return states, actions, rewards, next_states, dones, idxs, np.array(IS_weight)

    def collect_transitions(self, start_state):
        state = start_state

        self.step += 1
        self.epsilon = 1 / (self.episode * 0.1 + 10)
        action = self.get_action(state)
        next_state, reward, done, _ = self.env.step(action)

        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        done = np.array(done, dtype=np.int32)
        prioritize = self.get_td_error(state, action, reward, next_state, done)
        self.memory.add(prioritize[0], (state, action, reward, next_state, done), )
        self.score += reward
        state = next_state
        if done:
            self.episode += 1
            print("Episode %s, Score %s." % (self.episode, self.score))
            if self.set_summary:
                self.train_summary(self.score, self.episode)
                self.save_checkpoint()
            self.score = 0
            state = self.env.reset()
            if self.episode % 1000 == 0:
                self.dqn_net.save("dueling_dqn_per_model/")


        return state

    def update(self):
        states, actions, rewards, next_states, dones, idxs, IS_weight = self.sampling()
        dqn_variable = self.dqn_net.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.int32)

            q = tf.stop_gradient(self.dqn_net(next_states))
            next_actions = tf.argmax(q, axis=1)
            target_q = self.target_net(next_states)
            target_op_value = tf.reduce_sum(tf.one_hot(next_actions, self.action_size) * target_q, axis=1)
            td_target = rewards + self.gamma * (1 - dones) * target_op_value

            q = self.dqn_net(states)
            op_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * q, axis=1)

            loss = tf.reduce_mean(tf.square(op_value - td_target) * 0.5 * IS_weight)
            self.train_loss(loss)

        gradients = tape.gradient(loss, dqn_variable)
        self.opt.apply_gradients(zip(gradients, dqn_variable))

        td_error = tf.abs(op_value - td_target)
        for i in range(self.batch_size):
            self.memory.update(idxs[i], td_error[i])

        if self.step % self.target_net_update_step == 0:
            self.target_net.set_weights(self.dqn_net.get_weights())

    def learn(self, max_step=1000000):
        # self.ckpt.restore(self.manager.latest_checkpoint)
        state = self.env.reset()
        for _ in range(max_step):
            state = self.collect_transitions(state)
            if self.step > 1000 and self.step % 10 == 0:
                self.update()

    def get_td_error(self, state, action, reward, next_state, done):
        # state = np.array([state])
        q = tf.stop_gradient(self.dqn_net(np.array([state])))
        next_action = tf.argmax(q, axis=1)
        target_q = self.target_net(np.array([next_state]))
        target_op_value = tf.reduce_sum(tf.one_hot(next_action[0], self.action_size) * target_q, axis=1)
        td_target = reward + self.gamma * (1 - done) * target_op_value

        q = self.dqn_net(np.array([state]))
        op_value = tf.reduce_sum(tf.one_hot(action, self.action_size) * q, axis=1)
        td_error = tf.abs(op_value - td_target)

        return td_error


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    # train
    agent = DQNAgent(env=env)
    agent.learn()

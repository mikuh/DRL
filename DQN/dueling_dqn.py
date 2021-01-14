import tensorflow as tf
import numpy as np
from collections import deque
import gym
import random

from Agent import Agent
from Layers import CNNEmbeddingNet, DenseEmbeddingNet, ValueNet, QNet


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

        self.dqn_net = DQN(embedding_net=CNNEmbeddingNet(), value_net=ValueNet(),
                           q_net=QNet(env))
        self.target_net = DQN(embedding_net=CNNEmbeddingNet(), value_net=ValueNet(),
                              q_net=QNet(env))

        self.opt = tf.keras.optimizers.Adam(self.lr)

        self.action_size = self.env.action_space.n
        self.batch_size = 64

        self.memory = deque(maxlen=2000)

        self.step = 0
        self.target_net_update_step = 30
        self.episode = 0
        self.score = 0
        self.set_summary = set_summary
        if set_summary:
            self.checkpoint_summary_setting("dueling_dqn", self.opt, self.dqn_net)

    def get_action(self, state):

        q_value = self.dqn_net(np.array([state], np.float32))[0]
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action

    def sampling(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition in mini_batch:
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            dones.append(transition[4])

        return states, actions, rewards, next_states, dones

    def collect_transitions(self, start_state):
        state = start_state
        while True:
            self.env.render()
            self.step += 1
            self.epsilon = 1 / (self.episode * 0.1 + 5)
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.score += reward
            # use for CartPole-v1
            # if done:
            #     if self.score == 500:
            #         reward = 1
            #     else:
            #         reward = -1
            # else:
            #     reward = 0
            self.memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                self.episode += 1
                print("Episode %s, Score %s." % (self.episode, self.score))
                if self.set_summary:
                    self.train_summary(self.score, self.episode)
                    self.save_checkpoint()
                self.score = 0
                state = self.env.reset()
                if self.episode % 100 == 0:
                    self.dqn_net.save("dueling_dqn_model/")

            if self.step > 1000:
                return state

    def update(self):
        states, actions, rewards, next_states, dones = self.sampling()
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

            loss = tf.reduce_mean(tf.square(op_value - td_target) * 0.5)
            self.train_loss(loss)

        gradients = tape.gradient(loss, dqn_variable)
        self.opt.apply_gradients(zip(gradients, dqn_variable))
        if self.step % self.target_net_update_step == 0:
            self.target_net.set_weights(self.dqn_net.get_weights())

    def learn(self, max_step=1000000):
        self.ckpt.restore(self.manager.latest_checkpoint)
        state = self.env.reset()
        for _ in range(max_step):
            state = self.collect_transitions(state)
            self.update()


if __name__ == '__main__':
    env = gym.make("SpaceInvaders-v0")


    # train
    agent = DQNAgent(env=env)
    agent.learn()

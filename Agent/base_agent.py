import logging
import numpy as np
import os
import tensorflow as tf
import datetime


class Agent(object):

    def __init__(self):
        self.set_summary = False

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

    def train_summary(self, score, episode):
        self.train_score(score)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('score', self.train_score.result(), step=episode)
            tf.summary.scalar('loss', self.train_loss.result(), step=episode)
        self.train_loss.reset_states()
        self.train_score.reset_states()

    def save_checkpoint(self):
        if self.set_summary:
            self.ckpt.step.assign_add(1)
            if self.ckpt.step % 100 == 0:
                self.manager.save()

    def checkpoint_summary_setting(self, model_name, opt, net):
        checkpoint_path = "training_%s/cp-{epoch:04d}.ckpt" % model_name
        self.checkpoint_dir = os.path.dirname(checkpoint_path)

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
        self.manager = tf.train.CheckpointManager(self.ckpt, '../logs/{}'.format(model_name), max_to_keep=3)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'../logs/{model_name}/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_score = tf.keras.metrics.Mean('score', dtype=tf.float32)

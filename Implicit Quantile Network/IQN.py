import tensorflow as tf
import numpy as np
import math
from collections import deque
import gym
import random


class IQN:
    def __init__(self, sess):
        self.action_size = 2
        self.batch_size = 32
        self.num_quantiles = 32
        self.quantile_embedding_dim = 64
        self.state_size = 4
        self.sess = sess
        self.gamma = 0.99

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.tau = tf.placeholder(tf.float32, [None, self.num_quantiles])
        self.Y = tf.placeholder(tf.float32, [None, self.num_quantiles])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])

        self.main_support, self.main_params = self._build_net('main')
        self.target_support, self.target_params = self._build_net('target')

        expand_dim_action = tf.expand_dims(self.action, -1)
        main_support = tf.reduce_sum(self.main_support * expand_dim_action, axis=1)
        Huber_loss = tf.losses.huber_loss(self.Y, main_support, reduction=tf.losses.Reduction.NONE)

        tau = self.tau
        inv_tau = 1 - tau

        error_loss = self.Y - main_support

        Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
        Loss = tf.reduce_mean(tf.reduce_sum(Loss, axis=1))

        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(Loss)

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def train(self, memory):
        minibatch = random.sample(memory, self.batch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]
        done_stack = [int(i) for i in done_stack]

        t = np.random.rand(self.batch_size, self.num_quantiles)
        Q_next_state = self.sess.run(self.target_support, feed_dict={self.state: next_state_stack, self.tau: t})
        next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
        Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
        T_theta = [reward + (1-done)*self.gamma*Q for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]

        self.sess.run(self.train_op, feed_dict={self.state: state_stack, self.action: action_stack, self.tau:t, self.Y: T_theta})

    def _build_net(self, name):
        with tf.variable_scope(name):
            state_tile = tf.tile(self.state, [1, self.num_quantiles])
            state_reshape = tf.reshape(state_tile, [-1, self.state_size])
            state_net = tf.layers.dense(inputs=state_reshape, units=128, activation=tf.nn.selu)

            tau = tf.reshape(self.tau, [-1, 1])
            pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)
            cos_tau = tf.cos(tf.matmul(tau, pi_mtx))
            phi = tf.layers.dense(inputs=cos_tau, units=128, activation=tf.nn.relu)

            net = tf.multiply(state_net, phi)
            net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, units=self.action_size, activation=None)
            net = tf.transpose(tf.split(net, self.batch_size, axis=0), perm=[0, 2, 1])

            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

            return net, params

    def choose_action(self, obs):
        t = np.random.rand(iqn.batch_size, iqn.num_quantiles)
        obs = np.tile(obs, [iqn.batch_size, 1])
        Q_s_a = self.sess.run(self.main_support, feed_dict={self.state: obs, self.tau: t})
        Q_s_a = Q_s_a[0]
        Q_a = np.sum(Q_s_a, axis=1)
        action = np.argmax(Q_a)
        return action

env = gym.make('CartPole-v0')
sess = tf.Session()
iqn = IQN(sess)
sess.run(tf.global_variables_initializer())

r = tf.placeholder(tf.float32)  ########
rr = tf.summary.scalar('reward', r)
merged = tf.summary.merge_all()  ########
writer = tf.summary.FileWriter('./board/step', sess.graph)  ########

memory_size = 10000
memory = deque(maxlen=memory_size)

for episode in range(3000):
    e = 1. / ((episode / 10) + 1)
    done = False
    state = env.reset()
    global_step = 0
    while not done:
        global_step += 1
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            action = iqn.choose_action(state)

        next_state, reward, done, _ = env.step(action)

        if done:
            reward = -1
        else:
            reward = 0

        if len(memory) > 1000:
            iqn.train(memory)
            if global_step % 20 == 0:
                sess.run(iqn.assign_ops)
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1
        memory.append([state, next_state, action_one_hot, reward, done])
        state = next_state
        if done:
            summary = sess.run(merged, feed_dict={r: global_step})
            writer.add_summary(summary, episode)
            print(episode, global_step)
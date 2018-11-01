import gym
from Distributional_RL import Distributional_RL
import tensorflow as tf
import numpy as np
import random
from collections import deque

memory_size = 10000
memory = deque(maxlen=memory_size)

sess = tf.Session()
env = gym.make('CartPole-v1')
learning_rate = 0.0001
model = 'IQN'
dqn = Distributional_RL(sess, model, learning_rate)
sess.run(tf.global_variables_initializer())
sess.run(dqn.assign_ops)

r = tf.placeholder(tf.float32) 
rr = tf.summary.scalar('reward', r)
merged = tf.summary.merge_all()  
writer = tf.summary.FileWriter('./board/'+model, sess.graph)

episode = 0

while True:
    episode += 1
    e = 1. / ((episode / 10) + 1)
    done = False
    state = env.reset()
    global_step = 0
    l = 0
    while not done:
        global_step += 1
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            action = dqn.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1
        else:
            reward = 0

        if len(memory) > 1000:
            _, loss = dqn.train(memory)
            l += loss
            if global_step % 5 == 0:
                sess.run(dqn.assign_ops)

        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1
        memory.append([state, next_state, action_one_hot, reward, done])
        state = next_state
        if done:
            summary = sess.run(merged, feed_dict={r: global_step})
            writer.add_summary(summary, episode)
            print('episode:', episode, 'reward:', global_step, 'expectation loss:', l)

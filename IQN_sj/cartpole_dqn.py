import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil
from timeit import default_timer as timer
from datetime import timedelta

from dqnagent import DQNAgent
from collections import deque

np.random.seed(1212)
tf.set_random_seed(1212)

env = gym.make('CartPole-v1')
env.seed(1212)

MINIBATCH_SIZE = 32
TRAIN_START = 1000
TARGET_UPDATE = 25
MEMORY_SIZE = 20000
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
LEARNING_RATE = 0.001
DISCOUNT = 0.99
LOAD_model = True
SAVE_model = True
TRAIN = False
RENDER = False

path = "./CartPole_dqn_test"


def train():
    with tf.Session() as sess:
        DQNbrain = DQNAgent(sess, OUTPUT, INPUT,
                            learning_rate=LEARNING_RATE,
                            gamma=DISCOUNT,
                            batch_size=MINIBATCH_SIZE,
                            buffer_size=MEMORY_SIZE,
                            target_update_step=TARGET_UPDATE,
                            e_greedy=not LOAD_model,
                            e_step=1000,
                            gradient_norm=None,
                            )

        if LOAD_model:
            DQNbrain.load_model(tf.train.latest_checkpoint(path))
        else:
            sess.run(tf.global_variables_initializer())

        all_rewards = []
        frame_rewards = []
        loss_list = []
        loss_frame = []
        recent_rlist = deque(maxlen=15)
        recent_rlist.append(0)
        episode, epoch, frame = 0, 0, 0
        start = timer()
        env.env.masspole = 0.05
        env.env.length = 2.
        #env.env.force_mag = 10.

        while np.mean(recent_rlist) < 499:
            episode += 1

            rall, count = 0, 0
            done = False
            s = env.reset()

            while not done:
                if RENDER:
                    env.render()

                frame += 1
                count += 1

                action, actions_value = DQNbrain.choose_action(s)

                s_, r, done, l = env.step(action)

                if done and count >= 500:
                    reward = 1
                elif done and count < 500:
                    reward = -10
                else:
                    reward = 0

                DQNbrain.memory_add(s, float(action), reward, s_, int(done))
                s = s_

                rall += r

                if frame > TRAIN_START and TRAIN:
                    loss = DQNbrain.learn()
                    loss_list.append(loss)
                    loss_frame.append(frame)

            recent_rlist.append(rall)
            all_rewards.append(rall)
            frame_rewards.append(frame)

            print("Episode:{} | Frames:{} | Reward:{} | Recent reward:{}".format(episode, frame, rall,
                                                                                              np.mean(recent_rlist)))

        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
        ckpt_path = os.path.join(path, 'DQN.ckpt')
        if SAVE_model:
            DQNbrain.save_model(ckpt_path)

        plt.figure(figsize=(10, 8))
        plt.subplot(211)
        plt.title('Episode %s. Recent_reward: %s. Time: %s' % (
            len(all_rewards), np.mean(recent_rlist), timedelta(seconds=int(timer() - start))))
        plt.plot(frame_rewards, all_rewards)
        plt.ylim(0, 510)
        plt.subplot(212)
        plt.title('Loss')
        plt.plot(loss_frame, loss_list)
        #plt.ylim(0, 20)
        plt.show()
        plt.close()


def test():
    with tf.Session() as sess:
        DQNbrain = DQNAgent(sess, OUTPUT, INPUT,
                            learning_rate=LEARNING_RATE,
                            gamma=DISCOUNT,
                            batch_size=MINIBATCH_SIZE,
                            buffer_size=MEMORY_SIZE,
                            target_update_step=TARGET_UPDATE,
                            e_greedy=not LOAD_model,
                            e_step=1000,
                            gradient_norm=None,
                            )

        DQNbrain.load_model(tf.train.latest_checkpoint(path))

        masspole_list = np.arange(0.01, 0.21, 0.025)
        length_list = np.arange(0.5, 3, 0.25)

        performance_mtx = np.zeros([masspole_list.shape[0], length_list.shape[0]])

        for im in range(masspole_list.shape[0]):
            for il in range(length_list.shape[0]):
                env.env.masspole = masspole_list[im]
                env.env.length = length_list[il]

                all_rewards = []

                for episode in range(5):

                    rall, count = 0, 0
                    done = False
                    s = env.reset()

                    while not done:
                        if RENDER:
                            env.render()

                        action, actions_value = DQNbrain.choose_action(s)

                        s_, r, done, _ = env.step(action)

                        s = s_

                        rall += r

                    all_rewards.append(rall)

                    print("Episode:{} | Reward:{} ".format(episode, rall))

                performance_mtx[im, il] = np.mean(all_rewards)

        fig, ax = plt.subplots()
        ims = ax.imshow(performance_mtx, cmap=cm.gray, interpolation=None, vmin=0, vmax=500)
        ax.set_xticks(np.arange(0, length_list.shape[0], length_list.shape[0] - 1))
        ax.set_xticklabels(['0.5', '3'])
        ax.set_yticks(np.arange(0, masspole_list.shape[0], masspole_list.shape[0] - 1))
        ax.set_yticklabels(['0.01', '0.20'])
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Pole mass')
        ax.set_title('Robustness test: DQN')
        fig.colorbar(ims, ax=ax)
        plt.show()
        plt.close()


def main():
    if TRAIN:
        train()
    else:
        test()


if __name__ == "__main__":
    main()

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from timeit import default_timer as timer
from datetime import timedelta

from iqnagent import IQNAgent
from collections import deque

np.random.seed(1234)
tf.set_random_seed(1234)

env = gym.make('CartPole-v0')
env.seed(1234)

MINIBATCH_SIZE = 32
TRAIN_START = 1000
TARGET_UPDATE = 25
MEMORY_SIZE = 20000
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
LEARNING_RATE = 0.001
DISCOUNT = 0.95

path = "./CartPole_test"


def main():
    with tf.Session() as sess:
        IQNbrain = IQNAgent(sess, OUTPUT, INPUT,
                            learning_rate=LEARNING_RATE,
                            gamma=DISCOUNT,
                            batch_size=MINIBATCH_SIZE,
                            buffer_size=MEMORY_SIZE,
                            target_update_step=TARGET_UPDATE,
                            e_step=1000,
                            gradient_norm=None,
                            )

        all_rewards = []
        frame_rewards = []
        loss_list = []
        loss_frame = []
        recent_rlist = deque(maxlen=15)
        recent_rlist.append(0)
        episode, epoch, frame = 0, 0, 0
        start = timer()

        while episode < 500:
            episode += 1

            rall, count = 0, 0
            done = False
            s = env.reset()

            while not done:
                frame += 1
                count += 1

                action = IQNbrain.choose_action(s)

                s_, r, done, l = env.step(action)

                IQNbrain.memory_add(s, float(action), r, s_, int(done))
                s = s_

                rall += r

                if frame > TRAIN_START:
                    loss = IQNbrain.learn()
                    loss_list.append(loss)
                    loss_frame.append(frame)

            recent_rlist.append(rall)
            all_rewards.append(rall)
            frame_rewards.append(frame)

            print("Episode:{} | Frames:{} | Reward:{} | Recent reward:{}".format(episode, frame, rall,
                                                                                              np.mean(recent_rlist)))

        if os.path.isdir(path): shutil.rmtree(path)
        os.mkdir(path)
        ckpt_path = os.path.join(path, 'IQN.ckpt')
        IQNbrain.save_model(ckpt_path)

        plt.figure(figsize=(10, 8))
        plt.subplot(211)
        plt.title('Episode %s. Recent_reward: %s. Time: %s' % (
            len(all_rewards), np.mean(recent_rlist), timedelta(seconds=int(timer() - start))))
        plt.plot(frame_rewards, all_rewards)
        plt.ylim(0, 210)
        plt.subplot(212)
        plt.title('Loss')
        plt.plot(loss_frame, loss_list)
        plt.ylim(0, 20)
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()

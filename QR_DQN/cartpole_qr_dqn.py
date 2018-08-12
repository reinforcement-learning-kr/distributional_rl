import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from timeit import default_timer as timer
from datetime import timedelta

from qrdqnagent import QRduelingDQNAgent
from collections import deque


np.random.seed(1212)
tf.set_random_seed(1212)

env = gym.make('CartPole-v0')
env.seed(1212)

# 하이퍼 파라미터
MINIBATCH_SIZE = 64
TRAIN_START = 1000
TARGET_UPDATE = 20
MEMORY_SIZE = 10000
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
LEARNING_RATE = 0.001
DISCOUNT = 0.99

path = "./CartPole_test"


def main():
    with tf.Session() as sess:
        QRbrain = QRduelingDQNAgent(sess, OUTPUT, INPUT,
                                    learning_rate=LEARNING_RATE,
                                    gamma=DISCOUNT,
                                    batch_size=MINIBATCH_SIZE,
                                    buffer_size=MEMORY_SIZE,
                                    gradient_norm=None
                                    )

        all_rewards = []
        recent_rlist = deque(maxlen=15)
        recent_rlist.append(0)
        episode, epoch, frame = 0, 0, 0
        start = timer()

        while np.mean(recent_rlist) <= 198:
            episode += 1

            rall, count = 0, 0
            done = False
            s = env.reset()

            while not done:
                frame += 1
                count += 1

                action = QRbrain.choose_action(s)

                s_, r, done, l = env.step(action)

                if done:
                    reward = -1
                else:
                    reward = r

                QRbrain.memory_add(s, action.astype(float), reward, s_, int(done))
                s = s_

                rall += r

                if frame > TRAIN_START:
                    QRbrain.learn()

            recent_rlist.append(rall)
            all_rewards.append(rall)

            print("Episode:{} | Steps:{} | Reward:{} |Recent reward:{}".format(episode, count, rall, np.mean(recent_rlist)))

        if os.path.isdir(path): shutil.rmtree(path)
        os.mkdir(path)
        ckpt_path = os.path.join(path, 'QR_duel_DQN.ckpt')
        QRbrain.save_model(ckpt_path)

        plt.figure(figsize=(15, 5))
        plt.title('Episode %s. Recent_reward: %s. Time: %s' % (len(all_rewards), np.mean(recent_rlist), timedelta(seconds=int(timer()-start))))
        plt.plot(all_rewards)
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
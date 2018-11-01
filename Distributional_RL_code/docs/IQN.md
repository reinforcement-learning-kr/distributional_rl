본 readme는 Distributional_RL.py 내부의 클래스를 설명하는 문서입니다.

그 중에서 model이 IQN일 경우에 대한 설명입니다. IQN을 학습하는데 필요한 파라미터만 설명합니다.

# Distributional_RL.py

## import library

``` python
import gym
import tensorflow as tf
import numpy as np
import random
```

필요한 library들을 import합니다.

## __init__(self, sess, model, learning_rate)

에이전트를 생성하고 학습하는데 필요한 파라미터들을 설정합니다.

``` python
def __init__(self, sess, model, learning_rate):
        self.learning_rate = learning_rate
        self.state_size = 4
        self.action_size = 2
        self.model = model
        self.sess = sess
        self.batch_size = 8
        self.gamma = 0.99
        self.quantile_embedding_dim = 64

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.Y = tf.placeholder(tf.float32, [None, self.num_support])
        self.tau = tf.placeholder(tf.float32, [None, self.num_support])
```

self.learning_rate는 학습률이며, self.state_size는 state의 크기, self.action_size는 행동의 개수입니다. 여기서는 CartPole에 적용하였으므로 self.state_size = 4, self.action_size = 2입니다. self.model은 사용할 모델을 정의합니다. IQN으로 정의됩니다. 한번에 학습하기 위해 메모리에서 뽑은 데이터의 개수는 self.batch_size = 8개입니다. self.quantile_embedding_dim은 무작위의 tau를 특정 layer로 embedding하는 작업을 진행하는데에 얼마나의 정보를 가지도록 embedding을 할지 정하는 인수입니다. self.Y는 T * theta_j를 받는 인수입니다. Class 내부에 정의되어 있는 함수 _build_network를 통해 main network와 target network를 생성합니다. _build_network는 학습에 사용할 network, inference에 사용할 network, 그리고 network를 구성하고 있는 parameter를 출력으로 합니다. self.tau는 무작위로 생성된 tau를 인수로 받기 위한 placeholder입니다.

``` python
if self.model == 'IQN':
        expand_dim_action = tf.expand_dims(self.action, -1)
        main_support = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)

        theta_loss_tile = tf.tile(tf.expand_dims(main_support, axis=2), [1, 1, self.num_support])
        logit_valid_tile = tf.tile(tf.expand_dims(self.Y, axis=1), [1, self.num_support, 1])
        Huber_loss = tf.losses.huber_loss(logit_valid_tile, theta_loss_tile, reduction=tf.losses.Reduction.NONE)
        tau = self.tau
        inv_tau = 1 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.num_support, 1])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.num_support, 1])
        error_loss = logit_valid_tile - theta_loss_tile

        Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(Loss, axis=2), axis=1))

        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
```

[QRDQN의 학습 방법](QRDQN.md)과 같습니다. 링크를 타고 들어가서 확인하실 수 있습니다.

# train(self, memory)

``` python
def train(self, memory):
        minibatch = random.sample(memory, self.batch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]
        done_stack = [int(i) for i in done_stack]
        if self.model == 'IQN':
            t = np.random.rand(self.batch_size, self.num_support)
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack, self.tau: t})
            next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
            Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
            T_theta = [reward + (1-done)*self.gamma*Q for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss], feed_dict={self.state: state_stack, self.action: action_stack, self.tau:t, self.Y: T_theta})
```

[QRDQN의 학습 방법](QRDQN.md)과 같습니다. 링크를 타고 들어가서 확인하실 수 있습니다.

# _build_network(self, name)

네트워크를 만드는 함수입니다.

``` python
def _build_network(self, name):
        elif self.model == 'IQN':
                state_tile = tf.tile(self.state, [1, self.num_support])
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

                net_action = tf.transpose(tf.split(net, 1, axis=0), perm=[0, 2, 1])

                net = tf.transpose(tf.split(net, self.batch_size, axis=0), perm=[0, 2, 1])

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return net, net_action, params
```
 
QRDQN과 IQN이 다른 부분은 model의 구성 부분입니다. self.state 받아 embedding된 tau와 concatenate되기 전까지는 일반적인 layer 구성법과 같습니다. self.tau를 받아 학습이 가능한 상태로 reshape한 다음 tau에 정의합니다. pi_mtx는 0 ~ 63의 정수와 pi를 곱한 값을 나타냅니다. tau와 pi_mtx를 matrix multiply함으로써 cos(pi*i*tau)를 만들어내고 이를 cos_tau로 정의합니다. 다음 cos_tau를 Relu연산을 함으로써 phi에 정의내립니다. 이 식에 대한 이해가 필요하다면 [다음](https://github.com/reinforcement-learning-kr/distributional_rl/tree/master/3_CartPole_IQN)의 링크로 가서 2.Network 부분을 확인하세요.

앞의 DQN, C51, QRDQN과 달리 IQN은 무작위로 생성된 tau 모두에 대해 network를 가집니다. 하지만 실제 action을 inference할 경우에는 무작위로 생성한 모든 tau가 아닌 그 중 하나만에 대해 inference하면 되기에 net(training에 필요한 네트워크)와 net_action(inference에 필요한 네트워크)를 분리하여 return합니다. 이를 통해 inference할 시 불필요한 연산을 줄여 시간을 줄일 수 있습니다.

# choose_action(self, state)

network을 이용하여 action을 inference하는 함수입니다.

``` python
def _build_network(self, name):
        elif self.model == 'IQN':
            t = np.random.rand(1, self.num_support)
            Q_s_a = self.sess.run(self.main_action_support, feed_dict={self.state: [state], self.tau: t})
            Q_s_a = Q_s_a[0]
            Q_a = np.sum(Q_s_a, axis=1)
            action = np.argmax(Q_a)
        return action
```

무작위로 0과 1사이에 support의 개수만큼 tau를 생성합니다. tau와 state를 입력으로 하여 inference용 network의 출력을 Q_s_a로 가집니다. 각 action에 대한 support의 기대값을 계산하고 이 중 max인 action을 선택하여 return 합니다.
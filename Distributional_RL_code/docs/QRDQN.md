본 readme는 Distributional_RL.py 내부의 클래스를 설명하는 문서입니다.

그 중에서 model이 QRDQN일 경우에 대한 설명입니다. QRDQN을 학습하는데 필요한 파라미터만 설명합니다. 용어에 대해서 애매모호한 부분은 [다음 링크](https://github.com/reinforcement-learning-kr/distributional_rl/tree/master/2_CartPole_QR-DQN)에서 참조하세요

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

        self.num_support = 8

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.Y = tf.placeholder(tf.float32, [None, self.num_support])

        self.main_network, self.main_action_support, self.main_params = self._build_network('main')
        self.target_network, self.target_action_support, self.target_params = self._build_network('target')
```

self.learning_rate는 학습률이며, self.state_size는 state의 크기, self.action_size는 행동의 개수입니다. 여기서는 CartPole에 적용하였으므로 self.state_size = 4, self.action_size = 2입니다. self.model은 사용할 모델을 정의합니다. QRDQN으로 정의됩니다. 한번에 학습하기 위해 메모리에서 뽑은 데이터의 개수는 self.batch_size = 8개이며, support의 개수는 8개로 합니다. self.Y는 T * theta_j를 받는 인수입니다. Class 내부에 정의되어 있는 함수 _build_network를 통해 main network와 target network를 생성합니다. _build_network는 학습에 사용할 network, inference에 사용할 network, 그리고 network를 구성하고 있는 parameter를 출력으로 합니다.

``` python
elif self.model == 'QRDQN':
            self.theta_s_a = self.main_network
            expand_dim_action = tf.expand_dims(self.action, -1)
            theta_s_a = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)

            theta_loss_tile = tf.tile(tf.expand_dims(theta_s_a, axis=2), [1, 1, self.num_support])
            logit_valid_tile = tf.tile(tf.expand_dims(self.Y, axis=1), [1, self.num_support, 1])

            Huber_loss = tf.losses.huber_loss(logit_valid_tile, theta_loss_tile, reduction=tf.losses.Reduction.NONE)
            tau = tf.reshape(tf.range(1e-10, 1, 1 / self.num_support), [1, self.num_support])
            inv_tau = 1.0 - tau

            tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.num_support, 1])
            inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.num_support, 1])

            error_loss = logit_valid_tile - theta_loss_tile
            Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(Loss, axis=2), axis=1))

            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
```

self.theta_s_a는 [여기](https://github.com/reinforcement-learning-kr/distributional_rl/tree/master/2_CartPole_QR-DQN)에서 theta_i(s)를 뜻합니다. 그리고 실제 선택한 action(self.action)을 학습이 가능한 차원으로 맞춰 expand_dim_action으로 지정합니다. self.theta_s_a와 expand_dim_action을 이용하여 theta_i(s, a)를 구하고 theta_s_a에 지정합니다. 그리고 theta_loss_tile은 2번째 차원에 대해서 support개수만큼 복사를 한 후 저장한 것이고, logit_valid_tile은 1번째 차원에 대해서 support개수만큼 복사를 한 후 저장한 것입니다. theta_loss_tile과 logit_valid_tile에 대해서 huber loss를 구하고 Huber_loss에 지정합니다. 다음 support의 개수만큼 0에서 1사이에 균일하게 샘플링을 한 후 차원을 구한 후 tau에 저장합니다. 그리고 huber loss가 음수일 때에 곱해줄 1-tau를 계산하고 inv_tau에 지정합니다. 그리고 학습을 할 수 있는 차원으로 expand_dim을 한 후 tile합니다. 그리고 logit_valid_tile(T * theta_j) 와 theta_loss_tile(theta_i(s,a)) 간의 차이를 구합니다. 텐서플로우에서 제공하는 if문인 tf.where을 이용하여 error_loss가 0보다 작을 경우 inv_tau(1-tau)를 곱하고 0보다 클 경우 tau를 곱합니다. 마지막으로 self.loss가 뜻하는 것은 p(T*theta_j - theta_i(s,a))가 됩니다. 마지막으로 self.train_op은 self.loss를 줄이도록 학습하는 optimizer를 뜻합니다.

* theta_loss_tile과 logit_valid_tile을 각 축에 대해서 support의 개수만큼 복사를 하는 이유는 다음과 같습니다. target의 모든 theta와 main의 모든 theta가 서로 엮여서 계산을 해야되기 때문입니다. (target theta, main theta)라고 한다면 (1,1),(1,2),(1,3)...(1,8),(2,1),(2,2)...(2,8)...(8,8)에 대해 모두 계산해야 되기 때문입니다.(본 코드에서 support의 개수는 8개이기 때문)

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
        elif self.model == 'QRDQN':
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
            Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
            Q_next_state_next_action = np.sort(Q_next_state_next_action)
            T_theta = [np.ones(self.num_support) * reward if done else reward + self.gamma * Q for reward, Q, done in
                       zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.Y: T_theta})
```

먼저 self.batch_size만큼 메모리에서 샘플들을 뽑습니다. 그 후 state_stack, next_state_stack, action_stack. reward_stack, done_stack을 새로 지정한 후 배치합니다. Q_next_state는 next_state에 대한 분포를 뽑아내며 각 action에 대한 support들의 집합을 뜻합니다. 수식으로 나타내면 아래의 수식입니다.

<p align= "center">

<img src="https://github.com/reinforcement-learning-kr/distributional_rl/blob/master/2_CartPole_QR-DQN/img/quantile_regression.png" alt="paper" style="width: 600;"/>

 </p>

각 z들에 대한 평균값을 구하여 max값을 가지는 action을 next_action에 지정합니다. next_action을 이용하여 next_state에 대한 next_action의 support들을 구합니다. 다음 gamma를 곱하고 r을 더하여 T_theta에 지정합니다. 결국 T * theta_j를 수행한 것 입니다. 그리고 state와 action을 입력으로 가져 training을 진행합니다.


# _build_network(self, name)

네트워크를 만드는 함수입니다.

``` python
def _build_network(self, name):
        elif self.model == 'QRDQN':
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu,
                                          trainable=True)
                layer_4 = tf.layers.dense(inputs=layer_3, units=self.action_size * self.num_support, activation=None,
                                          trainable=True)
                net = tf.reshape(layer_4, [-1, self.action_size, self.num_support])
                net_action = net

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return net, net_action, params
```

네트워크를 만듭니다. 그리고 net은 학습을 위한 network, net_action은 inference를 위한 network이며 이를 구성하고 있는 params를 return합니다. QRDQN에서는 학습을 위한 network와 inference를 위한 network가 분리될 필요가 없기에 net_action = net을 사용하여 같게 두었습니다. net과 net_action의 크기는 [batch_size, action_size, support의 개수]로 본 예제에서는 [8, 2, 8]입니다. 

# choose_action(self, state)

network을 이용하여 action을 inference하는 함수입니다.

``` python
def _build_network(self, name):
        elif self.model == 'QRDQN':
            Q = self.sess.run(self.main_network, feed_dict={self.state: [state]})
            Q_s_a = np.mean(Q[0], axis=1)
            action = np.argmax(Q_s_a)

        return action
```

result는 main_network에서 state을 입력으로하여 Q(s)를 구합니다. 이 histogram에 대해 평균을 구하여 Q_s_a를 비교한 후 이 중에 큰 action을 선택합니다.
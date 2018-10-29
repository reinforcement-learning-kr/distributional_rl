본 readme는 Distributional_RL.py 내부의 클래스를 설명하는 문서입니다.

그 중에서 model이 DQN일 경우에 대한 설명입니다. DQN을 학습하는데 필요한 파라미터만 설명합니다.

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

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.dqn_Y = tf.placeholder(tf.float32, [None, 1])

        self.main_network, self.main_action_support, self.main_params = self._build_network('main')
        self.target_network, self.target_action_support, self.target_params = self._build_network('target')
```

self.learning_rate는 학습률이며, self.state_size는 state의 크기, self.action_size는 행동의 개수입니다. 여기서는 CartPole에 적용하였으므로 self.state_size = 4, self.action_size = 2입니다. self.model은 사용할 모델을 정의합니다. DQN으로 정의됩니다. self.sess는 tensorflow Session(tf.Session())입니다. self.batch_size는 실제 학습할 때 샘플링할 메모리의 개수입니다. Class 내부에 정의되어 있는 함수 _build_network를 통해 main network와 target network를 생성합니다. _build_network는 학습에 사용할 network, inference에 사용할 network, 그리고 network를 구성하고 있는 parameter를 출력으로 합니다.

``` python
elif self.model == 'DQN':
        self.Q_s_a = self.main_network * self.action
        self.Q_s_a = tf.expand_dims(tf.reduce_sum(self.Q_s_a, axis=1), -1)
        self.loss = tf.losses.mean_squared_error(self.dqn_Y, self.Q_s_a)
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
```

 학습에 필요한 변수들을 지정하는 부분입니다. self.Q_s_a는 main network에서 나온 Q값과 실제 선택한 action값을 곱함으로써 Q(s,a)가 됩니다. 이것을 target network에서 계산한 Qtarget(s',a')차이를 self.loss로 지정합니다. self.train_op는 self.train_op를 감소시키는 방향으로 학습을 시키도록 하는 변수입니다.

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
        elif self.model == 'DQN':
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(Q_next_state, axis=1)
            Q_next_state_next_action = [s[a] for s, a in zip(Q_next_state, next_action)]
            T_theta = [[reward + (1-done)*self.gamma * Q] for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.dqn_Y: T_theta})
```

먼저 self.batch_size만큼 메모리에서 샘플들을 뽑습니다. 그 후 state_stack, next_state_stack, action_stack. reward_stack, done_stack을 새로 지정한 후 배치합니다. target network에 next_state_stack을 입력으로 하여 Qtarget(s')을 구한 후 Q_next_state로 정의합니다. Q_next_state에서 큰 값을 선택하여 next_action을 계산한 후 그에 맞게 Qtarget(s',a')을 구하고 Q_next_state_next_action에 지정합니다. 그 후 done_stack에 지정되어 있는 값을 이용하여 bellman equation을 수행하고 T_theta에 정의합니다. 결국 T_theta는 Qtarget(s',a')*gamma + r이 됩니다. 이 값을 self.dqn_Y에 배치하고 training을 합니다.

# _build_network(self, name)

네트워크를 만드는 함수입니다.

``` python
def _build_network(self, name):
        with tf.variable_scope(name):
            if self.model == 'DQN':
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu,
                                          trainable=True)
                net = tf.layers.dense(inputs=layer_3, units=self.action_size, activation=None)
                net_action = net
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return net, net_action, params
```

네트워크를 만듭니다. 그리고 net은 학습을 위한 network, net_action은 inference를 위한 network이며 이를 구성하고 있는 params를 return합니다. DQN에서는 학습을 위한 network와 inference를 위한 network가 분리될 필요가 없기에 net_action = net을 사용하여 같게 두었습니다.

# choose_action(self, state)

network을 이용하여 action을 inference하는 함수입니다.

``` python
def _build_network(self, name):
        if self.model == 'DQN':
            result = self.sess.run(self.main_network, feed_dict={self.state: [state]})[0]
            action = np.argmax(result)
        return action
```

result는 main_network에서 state을 입력으로하여 Q(s)를 구합니다. 그 중 큰 값에 해당하는 action을 선택합니다.
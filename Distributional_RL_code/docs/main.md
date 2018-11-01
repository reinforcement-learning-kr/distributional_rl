본 readme는 Distributional_RL.py 내부의 클래스를 설명하고 이를 CartPole에 구현하는 파일 main.py를 설명하는 문서입니다. 

먼저 Distributional_RL 클래스를 이용하여 CartPole을 학습하는 에이전트를 구현하는 main.py를 설명합니다.

# main.py

## import library
``` python
import gym
from Distributional_RL import Distributional_RL
import tensorflow as tf
import numpy as np
import random
from collections import deque
```

필요한 library들을 import하고 정의한 Distributional_RL 클래스를 import합니다.

## setting parameter & define model

``` python
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
```

데이터가 쌓일 max의 개수를 지정합니다. 사용할 환경인 CartPole-v1을 지정합니다. 학습에 사용할 model과 learning_rate를 지정합니다. model은 `DQN`, `C51`, `QR-DQN`, `IQN` 네가지입니다. 다음 Distributional_RL의 클래스를 이용하여 dqn이라는 객체를 지정하고 sess.run(tf.global_variables_initializer())를 이용하여 초기 파라미터들을 셋팅합니다. sess.run(dqn.assign_ops)를 이용하여 초기 target network와 main network의 파라미터들을 동일하게 셋팅합니다. r = tf.placeholder(tf.float32)이하의 코드는 tensorboard 파일을 만들기 위해 추가된 코드입니다.

## run episode with step

전체 코드는 아래와 같습니다.

``` python
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
```

전체 코드를 나눠서 설명하겠습니다.

``` python
episode = 0

while True:
    episode += 1
    e = 1. / ((episode / 10) + 1)
    done = False
    state = env.reset()
    global_step = 0
    l = 0
```

에피소드를 무한으로 돌리지만 몇번째 에피소드인지 알기 위해 episode=0으로 지정하고 while에 한번 들어갈때마다 1씩 더합니다. e의 값에 따라 exploration을 하기 위해 다음과 같이 지정합니다. episode가 증가할때마다 e가 조금씩 감소합니다. state = env.reset()를 이용하여 환경을 리셋하고 그 때의 state를 받습니다.

``` python
episode = 0

while True:
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
```
스텝별로 global_step을 1씩 증가시킵니다. 장대를 오래 세우고 있는 것이 목적이며 이를 global_step으로 한 에피소드에서 얼마나 잘 수행했느냐를 판단합니다. 0 ~ 1 사이의 무작위한 숫자를 생성하고 그 수가 e보다 적으면 랜덤한 action을 선택하고 그 외에는 에이전트가 선택한 action을 선택합니다. 에피소드가 점점 진행할수록 e가 적어지기 때문에 exploration하는 비율이 적어집니다.

exploration에 의하거나 에이전트가 선택한 action을 env.step()의 input으로 하고 output으로 next_state, reward, done를 받습니다. 에피소드가 끝났을 경우 reward가 -1이며 그 외에는 0입니다. 에피소드가 끝나지 않도록 장대를 계속 세우고 있는 것이 목적이므로 이와 같이 reward을 설정합니다.

## train, memory append

``` python
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
```

memory에 1000개 이상의 데이터가 모였을 경우에 학습을 진행합니다. dqn.train(memory)를 이용하여 memory내에 있는 데이터를 토대로 에이전트를 학습합니다. 그리고 5 step마다 target network에 main network의 파라미터를 씌웁니다.

action는 one-hot encoding의 형태로 바꾸어준 후 memory에 state, next_state, action_one_hot, reward, done을 저장합니다. state에 next_state를 덮어씌웁니다.

마지막으로 에피소드가 종료되었을 때마다 에피소드의 수마다 global_step을 저장합니다.
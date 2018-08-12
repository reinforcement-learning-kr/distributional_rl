# Cartpole
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Import modules
import tensorflow as tf
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
import time
import gym

env = gym.make('CartPole-v0')
game_name = 'CartPole'
algorithm = 'C51'

# Parameter setting
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.01

Num_replay_memory = 10000
Num_start_training = 10000
Num_training = 15000
Num_testing  = 10000
Num_update = 150
Num_batch = 32
Num_episode_plot = 20

# Categorical Parameters
Num_atom = 51
V_min = -25.0
V_max = 25.0
delta_z = (V_max - V_min) / (Num_atom - 1)

first_fc  = [4, 512]
second_fc = [512, 128]
third_fc  = [128, Num_action * Num_atom]

Is_render = False 

# Initialize weights and bias
def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

def bias_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

# Assigning network variables to target network variables
def assign_network_to_target():
    # Get trainable variables
    trainable_variables = tf.trainable_variables()
    # network variables
    trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

    # target variables
    trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

    for i in range(len(trainable_variables_network)):
        sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

# Input
x = tf.placeholder(tf.float32, shape = [None, 4])

# Set z
z = tf.reshape ( tf.range(V_min, V_max + delta_z, delta_z), [1, Num_atom])

# Densely connect layer variables
with tf.variable_scope('network'): 
    w_fc1 = weight_variable('_w_fc1',first_fc)
    b_fc1 = bias_variable('_b_fc1',[first_fc[1]])

    w_fc2 = weight_variable('_w_fc2',second_fc)
    b_fc2 = bias_variable('_b_fc2',[second_fc[1]])

    w_fc3 = weight_variable('_w_fc3',third_fc)
    b_fc3 = bias_variable('_b_fc3',[third_fc[1]])

h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1)+b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

# Get Q value for each action
logits = tf.matmul(h_fc2, w_fc3) + b_fc3
logits_reshape = tf.reshape(logits, [-1, Num_action, Num_atom])
p_action = tf.nn.softmax(logits_reshape)
Q_action = tf.reduce_sum(tf.multiply(z, p_action), axis = 2)

# Densely connect layer variables target
with tf.variable_scope('target'): 
    w_fc1_target = weight_variable('_w_fc1',first_fc)
    b_fc1_target = bias_variable('_b_fc1',[first_fc[1]])

    w_fc2_target = weight_variable('_w_fc2',second_fc)
    b_fc2_target = bias_variable('_b_fc2',[second_fc[1]])

    w_fc3_target = weight_variable('_w_fc3',third_fc)
    b_fc3_target = bias_variable('_b_fc3',[third_fc[1]])

h_fc1_target = tf.nn.relu(tf.matmul(x, w_fc1_target)+b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)

# p value (target network)
logits_target = tf.matmul(h_fc2, w_fc3) + b_fc3
logits_reshape_target = tf.reshape(logits_target, [-1, Num_action, Num_atom])
p_action_target = tf.nn.softmax(logits_reshape_target)

# Loss function and Train
m_loss = tf.placeholder(tf.float32, shape = [Num_batch, Num_atom])
action_binary_loss = tf.placeholder(tf.float32, shape = [None, Num_action * Num_atom])

logit_valid = tf.multiply(logits, action_binary_loss)
logit_valid_reshape = tf.reshape(logit_valid, [-1, Num_action, Num_atom])
logit_valid_nonzero = tf.reduce_sum(logit_valid_reshape, axis = 1)

p_loss = tf.nn.softmax(logit_valid_nonzero)

Loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(m_loss, tf.log(p_loss + 1e-8)), axis = 1))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Initial parameters
Replay_memory = []

step = 1
score = 0
episode = 0

plot_y_loss = []
plot_y_maxQ = []
loss_list = []
maxQ_list = []

data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

state = env.reset()

# Figure and figure data setting
plot_x = []
plot_y = []

# f, ax = plt.subplots(3,2, sharex=False)
# f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(3,2, sharex=True)
# Making replay memory
while True:
    if Is_render:
        # Rendering
        env.render()

    if step <= Num_start_training:
        progress = 'Exploring'
    elif step <= Num_start_training + Num_training:
        progress = 'Training'
    elif step < Num_start_training + Num_training + Num_testing:
        progress = 'Testing'
    else:
        # Test is finished
        print('Test is finished!!')
        plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')
        break

    # Select Action (Epsilon Greedy)
    if random.random() < Epsilon:       
        action = np.zeros([Num_action])
        action[random.randint(0, Num_action - 1)] = 1.0
        action_step = np.argmax(action)
    else:
        Q_value = Q_action.eval(feed_dict={x: [state]})[0]
        action = np.zeros([Num_action])
        action[np.argmax(Q_value)] = 1
        action_step = np.argmax(action)

    state_next, reward, terminal, info = env.step(action_step)

    if progress != 'Testing':
        # Training to stay at the center 
        reward -= 5 * abs(state_next[0])

    # Save experience to the Replay memory
    if len(Replay_memory) > Num_replay_memory:
        del Replay_memory[0]

    Replay_memory.append([state, action, reward, state_next, terminal])

    if progress == 'Training':
        minibatch =  random.sample(Replay_memory, Num_batch)

        # Save the each batch data
        state_batch      = [batch[0] for batch in minibatch]
        action_batch     = [batch[1] for batch in minibatch]
        reward_batch     = [batch[2] for batch in minibatch]
        state_next_batch = [batch[3] for batch in minibatch]
        terminal_batch 	 = [batch[4] for batch in minibatch]

        y_batch = []

        # Update target network according to the Num_update value
        if step % Num_update == 0:
            assign_network_to_target()

        # Training
        Q_batch = Q_action.eval(feed_dict = {x: state_next_batch})
        p_batch = p_action_target.eval(feed_dict = {x: state_next_batch})
        z_batch = z.eval()

        m_batch = np.zeros([Num_batch, Num_atom])
        for i in range(len(minibatch)):
            action_max = np.argmax(Q_batch[i, :])
            if terminal_batch[i]:
                for j in range(Num_atom):
                    Tz = reward_batch[i]

                    # Bounding Tz
                    if Tz >= V_max:
                        Tz = V_max
                    elif Tz <= V_min:
                        Tz = V_min

                    b = (Tz - V_min) / delta_z
                    l = np.int32(np.floor(b))
                    u = np.int32(np.ceil(b))

                    m_batch[i, l] += (u - b)
                    m_batch[i, u] += (b - l)
            else:
                for j in range(Num_atom):
                    Tz = reward_batch[i] + Gamma * z_batch[0,j]

                    # Bounding Tz
                    if Tz >= V_max:
                        Tz = V_max
                    elif Tz <= V_min:
                        Tz = V_min

                    b = (Tz - V_min) / delta_z
                    l = np.int32(np.floor(b))
                    u = np.int32(np.ceil(b))

                    m_batch[i, l] += p_batch[i, action_max, j] * (u - b)
                    m_batch[i, u] += p_batch[i, action_max, j] * (b - l)

            # Normalize m (target distribution)
            sum_m_batch = np.sum(m_batch[i,:])
            for j in range(Num_atom):
                m_batch[i,j] = m_batch[i,j] / sum_m_batch

        # Calculate action binary
        action_binary = np.zeros([Num_batch, Num_action * Num_atom])

        for i in range(len(action_batch)):
        	action_batch_max = np.argmax(action_batch[i])
        	action_binary[i, Num_atom * action_batch_max : Num_atom * (action_batch_max + 1)] = 1

        _, loss, p_test = sess.run([train_step, Loss, p_loss],
                                    feed_dict = {x:state_batch, 
                                                 m_loss: m_batch, 
                                                 action_binary_loss: action_binary})

        loss_list.append(loss)
        maxQ_list.append(np.max(Q_batch))

        # Reduce epsilon at training mode
        if Epsilon > Final_epsilon:
            Epsilon -= 1.0/Num_training

    if progress == 'Testing':
        Epsilon = 0

    # Update parameters at every iteration
    step += 1
    score += reward
    state = state_next

    # Plot average score
    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and progress != 'Exploring':
        plt.subplot(3, 2, 1)
        plt.hold(True)
        plt.ylabel('Mean Loss')
        plt.plot(np.average(plot_x), np.average(plot_y_loss), '*')

        plt.subplot(3, 2, 3)
        plt.hold(True)
        plt.ylabel('Mean score')
        plt.plot(np.average(plot_x), np.average(plot_y),'*')

        plt.subplot(3, 2, 5)
        plt.hold(True)
        plt.xlabel('episode')
        plt.ylabel('Mean Max Q')
        plt.plot(np.average(plot_x), np.average(plot_y_maxQ),'*')

        plt.subplot(3, 2, 2)
        plt.cla()
        plt.hold(True)
        plt.plot(z_batch[0,:], p_test[0,:],'b', label='Prediction')
        plt.plot(z_batch[0,:], m_batch[0,:],'r', label='Target')
        plt.legend(loc='upper left')
        plt.title('Distributions')

        plt.subplot(3, 2, 4)
        plt.cla()
        plt.hold(True)
        plt.plot(z_batch[0,:], p_test[1,:],'b')
        plt.plot(z_batch[0,:], m_batch[1,:],'r')

        plt.subplot(3, 2, 6)
        plt.cla()
        plt.hold(True)
        plt.plot(z_batch[0,:], p_test[2,:],'b')
        plt.plot(z_batch[0,:], m_batch[2,:],'r')
        plt.xlabel('supports')

        plt.draw()
        plt.pause(0.000001)

        plot_x = []
        plot_y = []
        plot_y_loss = []
        plot_y_maxQ = []

    # Terminal
    if terminal == True:
        print('step: ' + str(step) + ' / '  + 
              'episode: ' + str(episode) + ' / ' +
              'progess: ' + progress  + ' / '  + 
              'epsilon: ' + str(Epsilon) + ' / '  + 
              'score: ' + str(score))

        if progress != 'Exploring':
            # add data for plotting
            plot_x.append(episode)
            plot_y.append(score)
            plot_y_loss.append(np.mean(loss_list))
            plot_y_maxQ.append(np.mean(maxQ_list))

        score = 0
        loss_list = []
        maxQ_list = []
        episode += 1

        state = env.reset()
import numpy as np
import tensorflow as tf
import random

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy

class DQNAgent:
    def __init__(self, sess, a_dim, s_dim,
                 learning_rate=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 buffer_size=10000,
                 target_update_step=25,
                 e_greedy=True,
                 e_step=1000,
                 eps_max=0.9,
                 eps_min=0.01,
                 gradient_norm=None
                 ):
        self.memory = []
        self.iter = 0
        self.sess = sess
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.e_greedy = e_greedy
        self.e_step = e_step
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps = 0 if not e_greedy else self.eps_max
        self.target_update_step = target_update_step
        self.gradient_norm = gradient_norm

        self.a_dim, self.s_dim, = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.int32, [None, 1], 'd')

        self.q = self._build_net(self.S, scope='eval', trainable=True)
        q_next = self._build_net(self.S_, scope='target', trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_eval_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.q_eval_a = tf.gather_nd(params=self.q, indices=a_eval_indices)

        y = tf.squeeze(self.R) + tf.squeeze(1 - tf.to_float(self.D)) * self.gamma * tf.reduce_max(q_next, axis=1)
        self.loss = tf.losses.mean_squared_error(labels=y, predictions=self.q_eval_a)

        if self.gradient_norm is not None:
            q_optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01/self.batch_size)
            q_gradients = q_optimizer.compute_gradients(self.loss, var_list=self.qe_params)
            for i, (grad, var) in enumerate(q_gradients):
                if grad is not None:
                    q_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm), var)
            self.train_op = q_optimizer.apply_gradients(q_gradients)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01/self.batch_size).minimize(self.loss, var_list=self.qe_params)

        self.saver = tf.train.Saver()

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 32, activation=tf.nn.selu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 32, activation=tf.nn.selu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            q = tf.layers.dense(net, self.a_dim, activation=None, name='a', trainable=trainable)
            return q

    def update_target_net(self):
        self.sess.run(self.params_replace)

    def memory_add(self, state, action, reward, next_state, done):
        self.memory += [(state, action, reward, next_state, done)]
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:]

    def learn(self):
        if self.iter % self.target_update_step == 0:
            self.update_target_net()

        minibatch = np.vstack(random.sample(self.memory, self.batch_size))
        bs = np.vstack(minibatch[:, 0])
        ba = np.vstack(minibatch[:, 1])
        br = np.vstack(minibatch[:, 2])
        bs_ = np.vstack(minibatch[:, 3])
        bd = np.vstack(minibatch[:, 4])

        loss, _ = self.sess.run([self.loss, self.train_op], {self.S: bs, self.A: ba, self.R: br, self.S_: bs_, self.D: bd})

        self.iter += 1

        if self.eps > self.eps_min:
            self.eps -= self.eps_max / self.e_step

        return loss

    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() > self.eps:
            actions_value = self.sess.run(self.q, feed_dict={self.S: state})
            action = np.argmax(actions_value)

        else:
            action = np.random.randint(0, self.a_dim)
            actions_value = 0

        return action, actions_value

    def save_model(self, model_path):
        print("Model saved in : ", self.saver.save(self.sess, model_path))

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)
        print("Model restored")












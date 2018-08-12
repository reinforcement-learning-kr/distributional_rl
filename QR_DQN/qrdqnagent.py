import numpy as np
import tensorflow as tf
import random

np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)

class QRDQNAgent:
    def __init__(self, sess, a_dim, s_dim,
                 num_q=50,
                 learning_rate=1e-3,
                 gamma=0.99,
                 batch_size=64,
                 buffer_size=10000,
                 gradient_norm=None
                 ):
        self.memory = []
        self.iter = 0
        self.sess = sess
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.num_q = num_q
        self.tau = tf.reshape(tf.lin_space(0.5/self.num_q, 1-0.5/self.num_q, self.num_q), [1, -1, 1])
        self.gradient_norm = gradient_norm

        self.a_dim, self.s_dim, = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.int32, [None, 1], 'd')

        self.q_theta_eval, self.q_mean_eval = self._build_net(self.S, scope='eval_params', trainable=True)
        self.q_theta_next, self.q_mean_next = self._build_net(self.S_, scope='target_params', trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_params')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_params')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.q_theta_eval_a = tf.gather_nd(params=self.q_theta_eval, indices=a_indices)
        a_next_max_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                       tf.squeeze(tf.to_int32(tf.argmax(self.q_mean_next, axis=1)))], axis=1)
        self.q_theta_next_a = tf.gather_nd(params=self.q_theta_next, indices=a_next_max_indices)

        self.loss = self.quantile_huber_loss()

        if self.gradient_norm is not None:
            q_optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.001/self.batch_size)
            q_gradients = q_optimizer.compute_gradients(self.loss, var_list=self.qe_params)
            for i, (grad, var) in enumerate(q_gradients):
                if grad is not None:
                    q_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm), var)
            self.train_op = q_optimizer.apply_gradients(q_gradients)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01/self.batch_size).minimize(self.loss, var_list=self.qe_params)

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, s, scope, trainable):

        def _noisy_dense(X, n_input, n_output, name_layer, trainable):
            W_mu = tf.get_variable("W_mu_" + name_layer, shape=[n_input, n_output],
                                   initializer=tf.random_uniform_initializer(-tf.sqrt(3 / n_input),
                                                                             tf.sqrt(3 / n_input)),
                                   trainable=trainable)

            W_sig = tf.get_variable("W_sig_" + name_layer, shape=[n_input, n_output],
                                    initializer=tf.constant_initializer(0.017), trainable=trainable)

            B_mu = tf.get_variable("B_mu_" + name_layer, shape=[1, n_output],
                                   initializer=tf.random_uniform_initializer(-tf.sqrt(3 / n_input),
                                                                             tf.sqrt(3 / n_input)),
                                   trainable=trainable)

            B_sig = tf.get_variable("B_sig_" + name_layer, shape=[1, n_output],
                                    initializer=tf.constant_initializer(0.017), trainable=trainable)

            W_fc = tf.add(W_mu, tf.multiply(W_sig, tf.random_normal(shape=[n_input, n_output])))

            B_fc = tf.add(B_mu, tf.multiply(B_sig, tf.random_normal(shape=[1, n_output])))

            pre_noisy_layer = tf.add(tf.matmul(X, W_fc), B_fc)

            return pre_noisy_layer

        with tf.variable_scope(scope):
            net = tf.nn.selu(_noisy_dense(s, self.s_dim, 512, "layer1", trainable=trainable))
            net = tf.nn.selu(_noisy_dense(net, 512, 128, "layer2", trainable=trainable))
            q_logits_flat = _noisy_dense(net, 128, self.a_dim * self.num_q, "layer3", trainable=trainable)
            q_theta = tf.reshape(q_logits_flat, [-1, self.a_dim, self.num_q], name="theta")
            q_theta_sort = tf.contrib.framework.sort(q_theta, axis=2)

            Q_mean = tf.reduce_mean(q_theta, axis=2)

        return q_theta_sort, Q_mean

    def update_target_net(self):
        self.sess.run(self.params_replace)

    def quantile_huber_loss(self):
        q_theta_expand = tf.tile(tf.expand_dims(self.q_theta_eval_a, axis=2), [1, 1, self.num_q])
        T_theta_expand = tf.tile(tf.expand_dims(self.T, axis=1), [1, self.num_q, 1])

        u_theta = T_theta_expand - q_theta_expand

        rho_u_tau = self._rho_tau(u_theta, tf.tile(self.tau, [1,1,self.num_q]))

        qr_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(rho_u_tau, axis=2), axis=1))

        return qr_loss

    def memory_add(self, state, action, reward, next_state, done):
        self.memory += [(state, action, reward, next_state, done)]
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:]

    def learn(self):
        if self.iter % 25:
            self.update_target_net()

        minibatch = np.vstack(random.sample(self.memory, self.batch_size))
        bs = np.vstack(minibatch[:, 0])
        ba = np.vstack(minibatch[:, 1])
        br = np.vstack(minibatch[:, 2])
        bs_ = np.vstack(minibatch[:, 3])
        bd = np.vstack(minibatch[:, 4])

        T_theta_ = self.sess.run(self.q_theta_next_a, feed_dict={self.S_: bs_})

        T_theta = br + (1-bd) * self.gamma * np.sort(T_theta_, axis=1)
        T_theta = T_theta.astype(np.float32)

        self.sess.run(self.train_op, {self.S: bs, self.A: ba, self.T: T_theta})

        self.iter += 1

    @staticmethod
    def _rho_tau(u, tau, kappa=1):
        delta = tf.cast(u < 0, 'float')
        if kappa == 0:
            return (tau - delta) * u
        else:
            return tf.abs(tau - delta) * tf.where(tf.abs(u) <= kappa, 0.5 * tf.square(u), kappa * (tf.abs(u) - kappa / 2))

    def choose_action(self, state):
        state = state[np.newaxis, :]
        actions_value = self.sess.run(self.q_mean_eval, feed_dict={self.S: state})
        action = np.argmax(actions_value)
        return action

    def save_model(self, model_path):
        print("Model saved in : ", self.saver.save(self.sess, model_path))

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)
        print("Model restored")


class QRduelingDQNAgent:
    def __init__(self, sess, a_dim, s_dim,
                 num_q=50,
                 learning_rate=1e-4,
                 gamma=0.99,
                 batch_size=64,
                 buffer_size=10000,
                 gradient_norm=None
                 ):
        self.memory = []
        self.iter = 0
        self.sess = sess
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.num_q = num_q
        self.tau = tf.reshape(tf.lin_space(0.5/self.num_q, 1-0.5/self.num_q, self.num_q), [1, -1, 1])
        self.gradient_norm = gradient_norm

        self.a_dim, self.s_dim, = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.int32, [None, 1], 'd')
        self.T = tf.placeholder(tf.float32, [None, num_q], 'theta_t')

        self.q_theta_eval, self.q_mean_eval = self._build_net(self.S, scope='eval_params', trainable=True)
        self.q_theta_next, self.q_mean_next = self._build_net(self.S_, scope='target_params', trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_params')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_params')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.q_theta_eval_a = tf.gather_nd(params=self.q_theta_eval, indices=a_indices)
        a_next_max_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                       tf.squeeze(tf.to_int32(tf.argmax(self.q_mean_next, axis=1)))], axis=1)
        self.q_theta_next_a = tf.gather_nd(params=self.q_theta_next, indices=a_next_max_indices)

        self.loss = self.quantile_huber_loss()

        if self.gradient_norm is not None:
            q_optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.001/self.batch_size)
            q_gradients = q_optimizer.compute_gradients(self.loss, var_list=self.qe_params)
            for i, (grad, var) in enumerate(q_gradients):
                if grad is not None:
                    q_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm), var)
            self.train_op = q_optimizer.apply_gradients(q_gradients)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01/self.batch_size).minimize(self.loss, var_list=self.qe_params)

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, s, scope, trainable):

        def _noisy_dense(X, n_input, n_output, name_layer, trainable):
            W_mu = tf.get_variable("W_mu_" + name_layer, shape=[n_input, n_output],
                                   initializer=tf.random_uniform_initializer(-tf.sqrt(3 / n_input),
                                                                             tf.sqrt(3 / n_input)),
                                   trainable=trainable)

            W_sig = tf.get_variable("W_sig_" + name_layer, shape=[n_input, n_output],
                                    initializer=tf.constant_initializer(0.017), trainable=trainable)

            B_mu = tf.get_variable("B_mu_" + name_layer, shape=[1, n_output],
                                   initializer=tf.random_uniform_initializer(-tf.sqrt(3 / n_input),
                                                                             tf.sqrt(3 / n_input)),
                                   trainable=trainable)

            B_sig = tf.get_variable("B_sig_" + name_layer, shape=[1, n_output],
                                    initializer=tf.constant_initializer(0.017), trainable=trainable)

            W_fc = tf.add(W_mu, tf.multiply(W_sig, tf.random_normal(shape=[n_input, n_output])))

            B_fc = tf.add(B_mu, tf.multiply(B_sig, tf.random_normal(shape=[1, n_output])))

            pre_noisy_layer = tf.add(tf.matmul(X, W_fc), B_fc)

            return pre_noisy_layer

        with tf.variable_scope(scope):
            net = tf.nn.selu(_noisy_dense(s, self.s_dim, 256, "layer1", trainable=trainable))
            net = tf.nn.selu(_noisy_dense(net, 256, 256, "layer2", trainable=trainable))

            v_logits_flat = _noisy_dense(net, 256, self.num_q, "layer3_v", trainable=trainable)
            v_logits_re = tf.reshape(v_logits_flat, [-1, 1, self.num_q])

            adv_logits_flat = _noisy_dense(net, 256, self.a_dim * self.num_q, "layer3_adv", trainable=trainable)
            adv_logits_re = tf.reshape(adv_logits_flat, [-1, self.a_dim, self.num_q])

            q_theta = v_logits_re + adv_logits_re - tf.reduce_mean(adv_logits_re, axis=1, keepdims=True)

            Q_mean = tf.reduce_mean(q_theta, axis=2)

        return q_theta, Q_mean

    def update_target_net(self):
        self.sess.run(self.params_replace)

    def quantile_huber_loss(self):
        q_theta_expand = tf.tile(tf.expand_dims(self.q_theta_eval_a, axis=2), [1, 1, self.num_q])
        T_theta_expand = tf.tile(tf.expand_dims(self.T, axis=1), [1, self.num_q, 1])

        u_theta = T_theta_expand - q_theta_expand

        rho_u_tau = self._rho_tau(u_theta, tf.tile(self.tau, [1,1,self.num_q]))

        qr_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(rho_u_tau, axis=2), axis=1))

        return qr_loss

    def memory_add(self, state, action, reward, next_state, done):
        self.memory += [(state, action, reward, next_state, done)]
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:]

    def learn(self):
        if self.iter % 25:
            self.update_target_net()

        minibatch = np.vstack(random.sample(self.memory, self.batch_size))
        bs = np.vstack(minibatch[:, 0])
        ba = np.vstack(minibatch[:, 1])
        br = np.vstack(minibatch[:, 2])
        bs_ = np.vstack(minibatch[:, 3])
        bd = np.vstack(minibatch[:, 4])

        T_theta_ = self.sess.run(self.q_theta_next_a, feed_dict={self.S_: bs_})

        T_theta = br + (1-bd) * self.gamma * np.sort(T_theta_, axis=1)
        T_theta = T_theta.astype(np.float32)

        self.sess.run(self.train_op, {self.S: bs, self.A: ba, self.T: T_theta})

        self.iter += 1

    @staticmethod
    def _rho_tau(u, tau, kappa=1):
        delta = tf.cast(u < 0, 'float')
        if kappa == 0:
            return (tau - delta) * u
        else:
            return tf.abs(tau - delta) * tf.where(tf.abs(u) <= kappa, 0.5 * tf.square(u), kappa * (tf.abs(u) - kappa / 2))

    def choose_action(self, state):
        state = state[np.newaxis, :]
        actions_value = self.sess.run(self.q_mean_eval, feed_dict={self.S: state})
        action = np.argmax(actions_value)
        return action

    def save_model(self, model_path):
        print("Model saved in : ", self.saver.save(self.sess, model_path))

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)
        print("Model restored")

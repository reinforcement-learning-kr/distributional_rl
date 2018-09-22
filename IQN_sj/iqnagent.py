import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)


class IQNAgent:
    def __init__(self, sess, a_dim, s_dim,
                 num_tau=32,
                 num_tau_prime=8,
                 learning_rate=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 buffer_size=10000,
                 target_update_step=25,
                 e_greedy=True,
                 e_step=1000,
                 eps_max=0.9,
                 eps_min=0.01,
                 eta=0.2,
                 gradient_norm=None,
                 view_dist=True
                 ):
        self.memory = []
        self.iter = 0
        self.sess = sess
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.num_tau = num_tau
        self.num_tau_prime = num_tau_prime
        self.e_greedy = e_greedy
        self.e_step = e_step
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps = 0.0 if not e_greedy else self.eps_max
        self.target_update_step = target_update_step
        self.gradient_norm = gradient_norm
        self.view_dist = view_dist
        self.eta = eta

        self.a_dim, self.s_dim, = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        self.A_ = tf.placeholder(tf.float32, [None, 1], 'a')
        self.T = tf.placeholder(tf.float32, [None, None], 'theta_t')
        self.tau = tf.placeholder(tf.float32, [None, None], 'tau')
        self.tau_ = tf.placeholder(tf.float32, [None, None], 'tau_')

        self.q_theta_eval_train, self.q_mean_eval_train, self.q_theta_eval_test, self.q_mean_eval_test = self._build_net(
            self.S, self.tau,
            scope='eval_params',
            trainable=True)
        self.q_theta_next_train, self.q_mean_next_train, _, _ = self._build_net(self.S_, self.tau_,
                                                                                scope='target_params',
                                                                                trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_params')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_params')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.q_theta_eval_a = tf.gather_nd(params=self.q_theta_eval_train, indices=a_indices)
        a_next_max_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                       tf.squeeze(tf.to_int32(self.A_))], axis=1)
        self.q_theta_next_a = tf.gather_nd(params=self.q_theta_next_train, indices=a_next_max_indices)

        self.loss = self.quantile_huber_loss()

        if self.gradient_norm is not None:
            q_optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01 / self.batch_size)
            q_gradients = q_optimizer.compute_gradients(self.loss, var_list=self.qe_params)
            for i, (grad, var) in enumerate(q_gradients):
                if grad is not None:
                    q_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm), var)
            self.train_op = q_optimizer.apply_gradients(q_gradients)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01 / self.batch_size).minimize(
                self.loss, var_list=self.qe_params)

        self.saver = tf.train.Saver()

        #self.sess.run(tf.global_variables_initializer())

    def _build_net(self, s, tau, scope, trainable):

        s_tiled = tf.tile(s, [1, tf.shape(tau)[1]])
        s_reshaped = tf.reshape(s_tiled, [-1, self.s_dim])
        tau_reshaped = tf.reshape(tau, [-1, 1])

        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)

            #net_psi = tf.layers.dense(s_reshaped, 16, activation=tf.nn.selu,
            #                          kernel_initializer=init_w, bias_initializer=init_b, name='psi',
            #                          trainable=trainable)
            cos_tau = tf.cos(tf.matmul(tau_reshaped, pi_mtx))
            net_phi = tf.layers.dense(cos_tau, 4, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='phi',
                                      trainable=trainable)

            #joint_term = tf.multiply(net_psi, net_phi)
            #joint_term = tf.multiply(s_reshaped, net_phi)
            joint_term = s_reshaped+tf.multiply(s_reshaped, net_phi)

            q_net = tf.layers.dense(joint_term, 32, activation=tf.nn.selu,
                                    kernel_initializer=init_w, bias_initializer=init_b, name="layer1",
                                    trainable=trainable)

            q_net = tf.layers.dense(q_net, 32, activation=tf.nn.selu,
                                    kernel_initializer=init_w, bias_initializer=init_b, name="layer2",
                                    trainable=trainable)

            q_flat = tf.layers.dense(q_net, self.a_dim, activation=None,
                                     kernel_initializer=init_w, bias_initializer=init_b, name="theta",
                                     trainable=trainable)

            q_re_train = tf.transpose(tf.split(q_flat, self.batch_size, axis=0), perm=[0, 2, 1])

            q_re_test = tf.transpose(tf.split(q_flat, 1, axis=0), perm=[0, 2, 1])

            q_mean_train = tf.reduce_mean(q_re_train, axis=2)

            q_mean_test = tf.reduce_mean(q_re_test, axis=2)

        return q_re_train, q_mean_train, q_re_test, q_mean_test

    def update_target_net(self):
        self.sess.run(self.params_replace)

    def quantile_huber_loss(self):
        q_theta_expand = tf.tile(tf.expand_dims(self.q_theta_eval_a, axis=2), [1, 1, self.num_tau_prime])
        T_theta_expand = tf.tile(tf.expand_dims(self.T, axis=1), [1, self.num_tau_prime, 1])

        u_theta = T_theta_expand - q_theta_expand

        rho_u_tau = self._rho_tau(u_theta, tf.tile(tf.expand_dims(self.tau, axis=2), [1, 1, self.num_tau_prime]))

        qr_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(rho_u_tau, axis=2), axis=1))

        return qr_loss

    def memory_add(self, state, action, reward, next_state, done):
        self.memory += [(state, action, reward, next_state, done)]
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:]

    def learn(self):
        if self.iter % self.target_update_step:
            self.update_target_net()

        minibatch = np.vstack(random.sample(self.memory, self.batch_size))
        bs = np.vstack(minibatch[:, 0])
        ba = np.vstack(minibatch[:, 1])
        br = np.vstack(minibatch[:, 2])
        bs_ = np.vstack(minibatch[:, 3])
        bd = np.vstack(minibatch[:, 4])

        tau = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_ = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_beta_ = self.conditional_value_at_risk(self.eta, np.random.rand(self.batch_size, self.num_tau))

        T_mean_K = self.sess.run(self.q_mean_next_train, feed_dict={self.S_: bs_, self.tau_: tau_beta_})
        ba_ = np.expand_dims(np.argmax(T_mean_K, axis=1), axis=1)

        T_theta_ = self.sess.run(self.q_theta_next_a, feed_dict={self.S_: bs_, self.A_: ba_, self.tau_: tau_})

        T_theta = br + (1 - bd) * self.gamma * T_theta_
        T_theta = T_theta.astype(np.float32)

        loss, _ = self.sess.run([self.loss, self.train_op], {self.S: bs, self.A: ba, self.T: T_theta, self.tau: tau})

        self.iter += 1

        if self.eps > self.eps_min:
            self.eps -= self.eps_max / self.e_step

        return loss

    @staticmethod
    def _rho_tau(u, tau, kappa=1):
        delta = tf.cast(u < 0, 'float')
        if kappa == 0:
            return (tau - delta) * u
        else:
            return tf.abs(tau - delta) * tf.where(tf.abs(u) <= kappa, 0.5 * tf.square(u),
                                                  kappa * (tf.abs(u) - kappa / 2))

    @staticmethod
    def conditional_value_at_risk(eta, tau):
        return eta * tau

    def choose_action(self, state):
        state = state[np.newaxis, :]
        tau_K = np.random.rand(1, self.num_tau)
        tau_beta = self.conditional_value_at_risk(self.eta, tau_K)
        if np.random.uniform() > self.eps:
            actions_value, q_dist = self.sess.run([self.q_mean_eval_test, self.q_theta_eval_test],
                                                  feed_dict={self.S: state, self.tau: tau_beta})
            action = np.argmax(actions_value)

        else:
            action = np.random.randint(0, self.a_dim)
            actions_value, q_dist = 0, 0

        return action, actions_value, q_dist, tau_beta

    def save_model(self, model_path):
        print("Model saved in : ", self.saver.save(self.sess, model_path))

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)
        print("Model restored")

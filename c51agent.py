import numpy as np
import tensorflow as tf
import random


class C51DQNAgent:
    def __init__(self, sess, a_dim, s_dim,
                 num_c=51,
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
        self.num_c = num_c
        self.V_min = -10.
        self.V_max = 10.
        self.dz = float(self.V_max - self.V_min) / (self.num_c - 1)
        self.z = [self.V_min + i * self.dz for i in range(self.num_c)]
        self.gradient_norm = gradient_norm

        self.a_dim, self.s_dim, = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.int32, [None, 1], 'd')
        self.M = tf.placeholder(tf.float32, [None, self.num_c], 'm_prob')

        self.dist_q_eval, self.q_mean_eval = self._build_net(self.S, scope='eval_params', trainable=True)
        self.dist_q_next, self.q_mean_next = self._build_net(self.S_, scope='target_params', trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_params')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_params')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.dist_q_eval_a = tf.gather_nd(params=self.dist_q_eval, indices=a_indices)
        a_next_max_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                       tf.squeeze(tf.to_int32(tf.argmax(self.q_mean_next, axis=1)))], axis=1)
        self.dist_q_next_a = tf.gather_nd(params=self.dist_q_next, indices=a_next_max_indices)

        self.loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(self.M, tf.log(self.dist_q_eval_a + 1e-20)), axis=1))

        if self.gradient_norm is not None:
            q_optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.001/self.batch_size)
            q_gradients = q_optimizer.compute_gradients(self.loss, var_list=self.qe_params)
            for i, (grad, var) in enumerate(q_gradients):
                if grad is not None:
                    q_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm), var)
            self.train_op = q_optimizer.apply_gradients(q_gradients)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.001/self.batch_size).minimize(self.loss, var_list=self.qe_params)

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
            net = tf.nn.relu(_noisy_dense(s, self.s_dim, 256, "layer1", trainable=trainable))
            net = tf.nn.relu(_noisy_dense(net, 256, 256, "layer2", trainable=trainable))
            q_logits_flat = _noisy_dense(net, 256, self.a_dim * self.num_c, "layer3", trainable=trainable)
            q_logits_re = tf.reshape(q_logits_flat, [-1, self.a_dim, self.num_c])
            dist_q = tf.nn.softmax(q_logits_re, axis=2, name='dist_q')

            z_space = tf.tile(tf.reshape(self.z, [1, 1, self.num_c]), [tf.shape(q_logits_re)[0], self.a_dim, 1])
            Q_mean = tf.reduce_sum(tf.multiply(z_space, dist_q), axis=2)

        return dist_q, Q_mean

    def update_target_net(self):
        self.sess.run(self.params_replace)

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

        m_prob = self.projection_dist(ba, br, bs_, bd)

        self.sess.run(self.train_op, {self.S: bs, self.A: ba, self.M: m_prob})

        self.iter += 1

    def projection_dist(self, ba, br, bs_, bd):
        m_prob = np.zeros([bs_.shape[0], self.num_c])

        p_next_a = self.sess.run(self.dist_q_next_a, feed_dict={self.A: ba, self.S_: bs_})

        for j in range(self.num_c):
            Tz = np.fmin(self.V_max, np.fmax(self.V_min, br + (1 - bd) * self.gamma * (self.V_min + j * self.dz)))
            bj = (Tz - self.V_min) / self.dz
            lj = np.floor(bj).astype(int)
            uj = np.ceil(bj).astype(int)
            blj = bj - lj
            buj = uj - bj
            m_prob[np.arange(bs_.shape[0]), lj[np.arange(bs_.shape[0]), 0]] += (bd[:, 0] + (1 - bd[:, 0]) * (p_next_a[:, j])) * buj[:, 0]
            m_prob[np.arange(bs_.shape[0]), uj[np.arange(bs_.shape[0]), 0]] += (bd[:, 0] + (1 - bd[:, 0]) * (p_next_a[:, j])) * blj[:, 0]

        m_prob = m_prob / m_prob.sum(axis=1, keepdims=1)

        return m_prob

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


class C51duelingDQNAgent:
    def __init__(self, sess, a_dim, s_dim,
                 num_c=51,
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
        self.num_c = num_c
        self.V_min = -10.
        self.V_max = 10.
        self.dz = float(self.V_max - self.V_min) / (self.num_c - 1)
        self.z = [self.V_min + i * self.dz for i in range(self.num_c)]
        self.gradient_norm = gradient_norm

        self.a_dim, self.s_dim, = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.int32, [None, 1], 'd')
        self.M = tf.placeholder(tf.float32, [None, self.num_c], 'm_prob')

        self.dist_q_eval, self.q_mean_eval = self._build_net(self.S, scope='eval_params', trainable=True)
        self.dist_q_next, self.q_mean_next = self._build_net(self.S_, scope='target_params', trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_params')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_params')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.dist_q_eval_a = tf.gather_nd(params=self.dist_q_eval, indices=a_indices)
        a_next_max_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                       tf.squeeze(tf.to_int32(tf.argmax(self.q_mean_next, axis=1)))], axis=1)
        self.dist_q_next_a = tf.gather_nd(params=self.dist_q_next, indices=a_next_max_indices)

        self.loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(self.M, tf.log(self.dist_q_eval_a + 1e-20)), axis=1))

        if self.gradient_norm is not None:
            q_optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.001/self.batch_size)
            q_gradients = q_optimizer.compute_gradients(self.loss, var_list=self.qe_params)
            for i, (grad, var) in enumerate(q_gradients):
                if grad is not None:
                    q_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm), var)
            self.train_op = q_optimizer.apply_gradients(q_gradients)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.001/self.batch_size).minimize(self.loss, var_list=self.qe_params)

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
            net = tf.nn.relu(_noisy_dense(s, self.s_dim, 512, "layer1", trainable=trainable))
            net = tf.nn.relu(_noisy_dense(net, 512, 256, "layer2", trainable=trainable))

            v_logits_flat = _noisy_dense(net, 256, self.num_c, "layer3_v", trainable=trainable)
            v_logits_re = tf.reshape(v_logits_flat, [-1, 1, self.num_c])

            adv_logits_flat = _noisy_dense(net, 256, self.a_dim * self.num_c, "layer3_adv", trainable=trainable)
            adv_logits_re = tf.reshape(adv_logits_flat, [-1, self.a_dim, self.num_c])

            q_logits = v_logits_re + adv_logits_re - tf.reduce_mean(adv_logits_re, axis=1, keepdims=True)

            dist_q = tf.nn.softmax(q_logits, axis=2, name='dist_q')

            z_space = tf.tile(tf.reshape(self.z, [1, 1, self.num_c]), [tf.shape(q_logits)[0], self.a_dim, 1])
            Q_mean = tf.reduce_sum(tf.multiply(z_space, dist_q), axis=2)

        return dist_q, Q_mean

    def update_target_net(self):
        self.sess.run(self.params_replace)

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

        m_prob = self.projection_dist(ba, br, bs_, bd)

        self.sess.run(self.train_op, {self.S: bs, self.A: ba, self.M: m_prob})

        self.iter += 1

    def projection_dist(self, ba, br, bs_, bd):
        m_prob = np.zeros([bs_.shape[0], self.num_c])

        p_next_a = self.sess.run(self.dist_q_next_a, feed_dict={self.A: ba, self.S_: bs_})

        for j in range(self.num_c):
            Tz = np.fmin(self.V_max, np.fmax(self.V_min, br + (1 - bd) * self.gamma * (self.V_min + j * self.dz)))
            bj = (Tz - self.V_min) / self.dz
            lj = np.floor(bj).astype(int)
            uj = np.ceil(bj).astype(int)
            blj = bj - lj
            buj = uj - bj
            m_prob[np.arange(bs_.shape[0]), lj[np.arange(bs_.shape[0]), 0]] += (bd[:, 0] + (1 - bd[:, 0]) * (p_next_a[:, j])) * buj[:, 0]
            m_prob[np.arange(bs_.shape[0]), uj[np.arange(bs_.shape[0]), 0]] += (bd[:, 0] + (1 - bd[:, 0]) * (p_next_a[:, j])) * blj[:, 0]

        m_prob = m_prob / m_prob.sum(axis=1, keepdims=1)

        return m_prob

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
import datetime
import tensorflow as tf
import numpy as np

from config import config

class PVNet:
    def __init__(self):
        self.BOARD_SHAPE       = config.board_shape
        self.ACTION_SHAPE      = config.action_shape
        self.BOARD_SIZE        = config.board_shape[1]
        self.NUM_STATE_LAYERS  = config.num_state_layers
        self.NUM_ACTION_LAYERS = config.num_action_layers

        self.data_format = config.data_format
        self.framework   = config.framework

        if self.data_format == "channels_last":
            self.BOARD_SHAPE = np.roll(self.BOARD_SHAPE, -1)
        
        # main function
        self.create_network()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def evaluate(self, board):
        prior, V = self.sess.run([self.policy, self.value], feed_dict={
            self.tf_X: board[None],
            self.is_training: False,
        })
        return prior, V
    
    def create_network(self):
        # training input placeholders
        self.tf_X        = tf.placeholder(tf.float32, (None, *self.BOARD_SHAPE), name='X')
        self.is_training = tf.placeholder(bool, name='is_training')
        self.tf_Z        = tf.placeholder(tf.float32, (None,), name='Z')
        self.tf_pi       = tf.placeholder(tf.float32, (None, *self.ACTION_SHAPE), name='pi')
        # possible actions
        self.tf_PAs      = tf.placeholder(tf.float32, (None, *self.ACTION_SHAPE), name='PAs')

        # Modified the network described in the AGZ paper:
        # https://deepmind.com/documents/119/agz_unformatted_nature.pdf
        print("tf_X: ", self.tf_X.shape)
        out = self.convolutional_block(self.tf_X, self.is_training)
        print("out: ", out.shape)

        for i in range(config.num_resblock):
            out = self.residual_block(out, self.is_training, i)
            print("ResBlock: ", out.shape)
        self.policy = self.policy_head(out, self.is_training)
        print("policy: ", self.policy.shape)
        self.value  = self.value_head(out, self.is_training)
        print("value: ", self.value.shape)


        # Defining losses
        error       = self.tf_Z - tf.reshape(self.value, (-1,))
        self.vloss  = tf.reduce_mean(tf.square(error), name='v_loss')
        dot         = tf.reduce_sum(self.tf_pi * tf.math.log(self.policy), axis=-1)
        self.ploss  = -tf.reduce_mean(dot, name='p_loss')
        self.loss   = self.vloss + self.ploss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(1e-4)
            self.train_step = optimizer.minimize(self.loss)


    def convolutional_block(self, x, is_training):
        with tf.variable_scope('ConvBlock'):
            conv = tf.layers.conv2d(x, 128, 3, data_format=self.data_format, padding='same', name='conv')
            bn   = tf.layers.batch_normalization(conv, training=is_training, name='bn')
            relu = tf.nn.relu(bn, name='relu')
        return relu
    
    def residual_block(self, x, is_training, i):
        with tf.variable_scope(f'ResBlock_{i}'):
            conv1 = tf.layers.conv2d(x, 128, 3, data_format=self.data_format, padding='same', name='conv1')
            bn1   = tf.layers.batch_normalization(conv1, training=is_training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')
            conv2 = tf.layers.conv2d(relu1, 128, 3, data_format=self.data_format, padding='same', name='conv2')
            bn2   = tf.layers.batch_normalization(conv2, training=is_training, name='bn2')
            skip  = tf.add(x, bn2, name='skip_connection')
            relu2 = tf.nn.relu(skip, name='relu2')
        return relu2

    def policy_head(self, x, is_training):
        fan_out = self.NUM_ACTION_LAYERS
        with tf.variable_scope('PolicyHead'):
            conv   = tf.layers.conv2d(x, fan_out, 1, data_format=self.data_format, padding='same', name='conv')
            flat   = tf.layers.flatten(conv, name='flat')
            policy = tf.reshape(tf.nn.softmax(flat, axis=-1, name='softmax'), tf.shape(conv), name='policy')
        return policy

    def value_head(self, x, is_training):
        with tf.variable_scope('ValueHead'):
            conv  = tf.layers.conv2d(x, 1, 1, data_format=self.data_format, padding='same', name='conv')
            bn    = tf.layers.batch_normalization(conv, training=is_training)
            relu1 = tf.nn.relu(bn, name='relu1')
            flat  = tf.layers.flatten(relu1, name='flat')
            fc1   = tf.layers.dense(flat, 64, name='fc1')
            relu2 = tf.nn.relu(fc1, name='relu2')
            fc2   = tf.layers.dense(relu2, 2, name='fc2')
            value = tf.math.subtract(tf.nn.softmax(fc2), 0.5, name='value')
        return value

    def save(self):
        now = str(datetime.datetime.now())[:-7]
        now = now.replace(' ', '_').replace(':', '-')
        save = f'./{config.num_resblock}B_model_{now}'

        with open('location.txt', 'w') as f:
            f.write(save)

        tf.saved_model.simple_save(
            self.sess,
            save,
            inputs = {'X': self.tf_X},
            outputs = {
                'value': self.value,
                'policy': self.policy
            }
        )
    
    def __call__(self, board):
        return self.evaluate(board)

    
if __name__ == '__main__':
    import numpy as np
    from model import Blokus

    BOARD_SIZE = 13
    env = Blokus()
    state = env.reset()
    print('Board shape:')
    print(state.board.shape)
    net = PVNet()
    net.evaluate(state.board)
    print('Everything alright. Now saving.')
    net.save()
    print('Saved.')
    # data = np.random.randn(10, *env.BOARD_SHAPE) # 10, 51, 13, 13
    # np_pi = np.random.random_sample((10, 13, 13, 91))
    # np_pi -= np.max(np_pi)
    # np_pi = np.exp(np_pi) / np.sum(np.exp(np_pi))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     import time
    #     t0 = time.time()
    #     value, loss, vloss, ploss, policy = sess.run([net.value, net.loss, net.vloss, net.ploss, net.policy],
    #                             feed_dict={
    #                                 net.tf_X: data,
    #                                 net.tf_Z: np.random.randn(10),
    #                                 net.tf_pi: np_pi,
    #                                 net.is_training: True})
    #     t1 = time.time()
    #     print(policy)
    #     print()
    #     print(value, loss, vloss, ploss)
    #     print(f'took: {t1- t0}')

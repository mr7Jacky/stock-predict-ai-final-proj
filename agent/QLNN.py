import tensorflow as tf
from tensorflow import keras


class QNetwork(tf.keras.Model):
    def __init__(self, state_shape, action_size, learning_rate):
        """
        Initialization of nn for Q-learning model
        :param state_shape: input shape of neural network, which changes as observation length changes
        :param action_size: size of the action space, output size of nn
        :param learning_rate: learning rate that controls how fast agent learns
        """
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.q_first = tf.keras.layers.Dense(units=128, input_shape=self.state_shape)
        self.q_second = tf.keras.layers.Dense(units=64, input_shape=(128,))
        self.q_state = tf.keras.layers.Dense(units=self.action_size, name='q_table', input_shape=(64,))

    def __call__(self, inputs):
        """
        Make predictions given an inputs
        :param inputs: input data of nn, here is the current state of price
        :return: predicted value, or actions
        """
        self.state_in, self.action_in, _ = inputs
        self.action = tf.one_hot(self.action_in, depth=self.action_size)
        self.q_first_layer = self.q_first(self.state_in)
        self.q_second_layer = self.q_second(self.q_first_layer)
        self.q_state_layer = self.q_state(self.q_second_layer)

        self.q_action = tf.reduce_sum(input_tensor=tf.multiply(self.q_state_layer, self.action), axis=1)

        return self.q_action

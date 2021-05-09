import tensorflow as tf
from tensorflow import keras

class QNetwork(tf.keras.Model):
    def __init__(self, state_shape, action_size, learning_rate):
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.q_first = tf.keras.layers.Dense(units=128, input_shape=self.state_shape)
        self.q_second = tf.keras.layers.Dense(units=64, input_shape=(128,))
        self.q_state = tf.keras.layers.Dense(units=self.action_size, name='q_table', input_shape=(64,))

    def __call__(self, inputs):
        self.state_in, self.action_in, _ = inputs
        self.action = tf.one_hot(self.action_in, depth=self.action_size)
        self.q_first_layer = self.q_first(self.state_in)
        self.q_second_layer = self.q_second(self.q_first_layer)
        self.q_state_layer = self.q_state(self.q_second_layer)

        self.q_action = tf.reduce_sum(input_tensor=tf.multiply(self.q_state_layer, self.action), axis=1)

        return self.q_action

class QLN:
    def __init__(self, state_shape, action_size, learning_rate):
        self.action_size = action_size
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(128, kernel_initializer='uniform', input_shape=(state_shape[0],)))
        self.model.add(keras.layers.Dense(64))
        self.model.add(keras.layers.Dense(self.action_size, name='q_table'))
        self.model.compile(loss='huber', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    def forward(self, inputs):
        #state_in, action_in, _ = inputs
        return self.model.predict(inputs)
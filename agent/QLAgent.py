import tensorflow as tf
import numpy as np

from QLNN import QLN
from QLNN import QNetwork

class QNAgent:
    def __init__(self, env, discount_rate=0.5, learning_rate=0.01):
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape

        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.model = QNetwork(self.state_size, self.action_size, self.learning_rate)#
#         self.model = QLN(self.state_size, self.action_size, self.learning_rate)

    def get_action(self, state, use_random=True):
        """Select action based on the q value corresponding to a given state. Best
        action will be the index of the highest q_value. Use np.argmax to take that."""
        # After training, state becomes the predicted value
#         print(state.reshape(1,self.state_size[0], self.state_size[1]).shape)
        q_state = self.model.q_state(self.model.q_second(self.model.q_first(state)))
#         q_state = self.model.forward(state.reshape(1,self.state_size[0], self.state_size[1]))
        action_greedy = np.argmax(q_state, axis=1)
        if use_random:
            action_random = [np.random.choice(range(self.action_size)) for i in range(len(state))]
            return action_random if np.random.random() < 0.250 else action_greedy
        else:
            return action_greedy

    def train(self, experience: tuple):
        state, action, next_state, reward, done = (exp for exp in experience)
        q_next = self.model.q_state(self.model.q_second(self.model.q_first(state)))
        q_target = reward + self.discount_rate * np.max(q_next, axis=1)
        with tf.GradientTape() as tape:
            q_action = self.model([state, action, q_target])
            loss = tf.reduce_sum(input_tensor=tf.square(q_target - q_action))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
#         self.model.fit(state, q_target, optimizer=self.optimizer, loss='huber')
    
    def load(self, path: str):
        self.model.load_weights(path)
import tensorflow as tf
import numpy as np

from QLNN import QLN
from QLNN import QNetwork

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculates the huber loss.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    err = tf.abs(y_true - y_pred, name='abs')
    mg = tf.constant(max_grad, name='max_grad')
    lin = mg*(err-.5*mg)
    quad=.5*err*err
    return tf.where(err < mg, quad, lin)

class QNAgent:
    def __init__(self, env, discount_rate=0.5, learning_rate=0.01, epsilon = 0.5):
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.model = QNetwork(self.state_size, self.action_size, self.learning_rate)

    def get_action(self, state, use_random=True):
        """Select action based on the q value corresponding to a given state. Best
        action will be the index of the highest q_value. Use np.argmax to take that."""
        # After training, state becomes the predicted value
        q_state = self.model.q_state(self.model.q_second(self.model.q_first(state)))
        action_greedy = np.argmax(q_state, axis=1)
        if use_random:
            action_random = np.floor(np.random.uniform(0, self.action_size, len(state))).astype('int')
            return action_random if np.random.uniform() < self.epsilon else action_greedy
        else:
            return action_greedy
    
    def train(self, experience: tuple):
        state, action, next_state, reward, done = (exp for exp in experience)
        q_next = self.model.q_state(self.model.q_second(self.model.q_first(state)))
        q_target = self.learning_rate * (reward + self.discount_rate * np.max(q_next, axis=1))
        with tf.GradientTape() as tape:
            q_action = self.model([state, action, q_target])
            #loss = tf.reduce_sum(input_tensor=tf.square(q_target - q_action))
            loss = tf.convert_to_tensor(huber_loss(q_target, q_action))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
    
    def load(self, path: str):
        self.model.load_weights(path)
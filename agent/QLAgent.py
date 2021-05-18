import tensorflow as tf
import numpy as np

from QLNN import QNetwork


def huber_loss(y_true, y_pred, max_grad=1.):
    """
    Calculates the huber loss.
    :param y_true: target value
    :param y_pred: predicted value
    :param max_grad: represents the maximum possible gradient magnitude
    :return: the huber loss as tf.Tensor
    """
    err = tf.abs(y_true - y_pred, name='abs')
    mg = tf.constant(max_grad, name='max_grad')
    lin = mg * (err - .5 * mg)
    quad = .5 * err * err
    return tf.where(err < mg, quad, lin)


class QNAgent:
    """
    Agent that decide which action to do, using q-learning model
    """

    def __init__(self, action_size, state_size, discount_rate=0.5, learning_rate=0.01, epsilon=0.5):
        """
        Initialization of agent:
            using a multiple layer neural network to represent q-table
            using q-learning method to train
        :param action_size: size of action
        :param state_size: shape of state
        :param discount_rate: gamma value for q learning equation
        :param learning_rate: alpha value for q learning equation
        :param epsilon: probability of taking a random action
        """
        self.action_size = action_size
        self.state_size = state_size
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.model = QNetwork(self.state_size, self.action_size, self.learning_rate)

    def get_action(self, state, use_random=True):
        """
        Select action based on the q value corresponding to a given state. Best
        action will be the index of the highest q_value. Use np.argmax to take that.
        :param state: current state as input to agent
        :param use_random: if we use the random action
        :return: action list for all batches, either random action is using random or greedy choice of actions
        """
        # After training, state becomes the predicted value
        q_state = self.model.q_state(self.model.q_second(self.model.q_first(state)))
        action_greedy = np.argmax(q_state, axis=1)
        if use_random:
            action_random = np.floor(np.random.uniform(0, self.action_size, len(state))).astype('int')
            return action_random if np.random.uniform() < self.epsilon else action_greedy
        else:
            return action_greedy

    def train(self, experience: tuple):
        """
        Training process of agent
        :param experience: A tuple includes all data required for training an agent, including:
            state: current state
            action: action to take
            next_state: next state
            reward: reward of state
            done: if finished
        """
        state, action, next_state, reward, done = (exp for exp in experience)
        q_next = self.model.q_state(self.model.q_second(self.model.q_first(state)))
        q_target = self.learning_rate * (reward + self.discount_rate * np.max(q_next, axis=1))
        with tf.GradientTape() as tape:
            q_action = self.model([state, action, q_target])
            # loss = tf.reduce_sum(input_tensor=tf.square(q_target - q_action))
            loss = tf.convert_to_tensor(huber_loss(q_target, q_action))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return np.mean(loss)

    def set_param(self, param):
        """
        Load the param from file. This function is used when try to use pretrain weight
        :param param: the path to weight file for agent
        """
        self.epsilon = param[2]
        self.discount_rate = param[3]
        self.learning_rate = param[4]
        self.optimizer = param[5]

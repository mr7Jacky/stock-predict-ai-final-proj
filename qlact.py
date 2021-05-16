import pickle
import os
import sys
sys.path.insert(1, './agent')
sys.path.insert(1, './price_pred')

import stock_env
from QLAgent import QNAgent
from helper import *

def load_ql_agent(filename):
    # Obtain parameter
    param = pickle.load(open(filename + '/param', 'rb'))
    # Initialize Agent
    a = QNAgent(param[0], param[1])
    # Set parameter
    a.set_param(param)
    # Load model weight
    a.model.load_weights(filename+'/model')
    return a


agent = load_ql_agent('ql_model')
print(agent.state_size)
state = np.zeros(agent.state_size) # shape = (1,n)
print(agent.get_action(np.transpose(state)))
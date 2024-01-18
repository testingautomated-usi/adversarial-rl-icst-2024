from exec_functions.train_agent_1 import train_agent_1
from utils.my_utils import callback_function_single

import utils.register
from config import config_11

e = 'highwayMA-v0'
n = 2000

for i in range(1):
    train_agent_1(env = e, agent_config= config_11, model_json_file ='../models/dqn.json', callback_fn=callback_function_single, number_of_episodes=n, maxlen_queue = 100, option = 'zero')

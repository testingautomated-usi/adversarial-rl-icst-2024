import highway_env
from gymnasium.envs.registration import register

from exec_functions.train_agent_2 import train_agent_2

from utils.my_utils import callback_function_multi, save_info_executions
import utils.register

from config import config_11, config_ma_2

import datetime

file_name = 'traing_agent_2_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

number_of_episodes = 2000
e = 'highwayMA-v0'


agent_file = './out/exp_1/run_20230914-170338_43658/checkpoint-best.tar'

for i in range(1):
    
    number_of_crashes, episode, directory, final_success_rate = train_agent_2(agent_file, env=e, agent_1_config = config_11, agent_2_config = config_ma_2, model_json_file='../models/dqn.json', number_of_episodes=number_of_episodes, maxlen_queue=100, option='zero', callback_fn=callback_function_multi, factor=0.5)
    save_info_executions(file_name, agent_file, i, directory, number_of_crashes,episode, final_success_rate)



import highway_env
from gymnasium.envs.registration import register

from exec_functions.retrain_agent_1 import retrain_agent_1
from utils.my_utils import callback_function_multi, save_info_executions
from exec_functions.test_agent_2 import test_agent_2

import utils.register
from config import config_11, config_ma_2

import datetime
import re
import os

pattern = r'run_(\d{8}-\d{6}_\d+)'

file_name = 'test_agent_2_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

number_of_episodes = 100
e = 'highwayMA-v0'

ego_file = './out/exp_1/run_20230914-170338_43658/checkpoint-best.tar'
'''
run = ['./out/HighwayEnvMA/DQNAgent/run_20231021-160638_11323/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-161053_11323/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-161602_11323/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-162213_11323/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-162916_11323/checkpoint-best.tar']
'''


'''
run = ['./out/HighwayEnvMA/DQNAgent/run_20231021-162213_11323/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-162916_11323/checkpoint-best.tar']
'''

run = ['out/exp_4/run_20231006-153212_36081/checkpoint-best.tar']


for adv_file in run:
    for j in range(1):
        match = re.search(pattern, adv_file) 
        if match:
            folder_name = os.path.join('v_analysis/test_agent_2_v2', match.group(0))
            print(folder_name)        
        else:
            folder_name = None
        
        number_of_crashes, episode, directory, final_success_rate = test_agent_2(ego_file, adv_file,env=e, agent_1_config=config_11, agent_2_config=config_ma_2, model_json_file='../models/dqn.json', number_of_episodes=number_of_episodes, callback_fn=callback_function_multi,maxlen_queue=100, option='zero', display=True, factor=0.5, folder=folder_name)
        save_info_executions(file_name, ego_file, j, directory, number_of_crashes, episode, final_success_rate, adv_file)




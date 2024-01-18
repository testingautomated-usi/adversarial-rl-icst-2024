import highway_env
from gymnasium.envs.registration import register
from utils.my_utils import callback_function_multi, save_info_executions, callback_function_single
from exec_functions.test_retrained import test_retrained_agent

import utils.register
from config import config_ma_2, config_rt, config_11

import datetime
import re
import os

pattern = r'run_(\d{8}-\d{6}_\d+)'

file_name = 'test_retrained_agent_1' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

number_of_episodes = 100
e = 'highwayMA-v0'


ego_files = ['./out/exp_5/run_20231015-142147_29233/checkpoint-best.tar']

adv_file = './out/exp_4/run_20231006-153212_36081/checkpoint-best.tar'



for ego_file in ego_files:
    for j in range(1):

        match = re.search(pattern, adv_file) 
        if match:
            folder_name = os.path.join('v_analysis/test_retrained_agent', match.group(0))
            print(folder_name)        
        else:
            folder_name = None
        print(folder_name)
        number_of_crashes, episode, directory, final_success_rate = test_retrained_agent(ego_file, adv_file, env=e, ego_config=config_rt, adversarial_config=config_ma_2, model_json_file='../models/dqn.json', number_of_episodes=number_of_episodes, callback_fn=callback_function_multi, maxlen_queue=100, option='zero', factor=0.5, folder = folder_name)
        #number_of_crashes, episode, directory, final_success_rate = test_retrained_agent(ego_file, adv_file, env=e, ego_config=config_11, adversarial_config=config_ma_2, model_json_file='../models/dqn.json', number_of_episodes=number_of_episodes, callback_fn=callback_function_single, maxlen_queue=100, option='zero', factor=0.5, folder=folder_name)               
        save_info_executions(file_name, ego_file, j, directory, number_of_crashes, episode, final_success_rate,adv_file)
from exec_functions.test_agent_1 import test_agent_1
from utils.my_utils import callback_function_single, save_info_executions, callback_function_multi

import utils.register
from config import config_11

import datetime
import re
import os

pattern = r'run_(\d{8}-\d{6}_\d+)'

file_name = 'test_agent_1_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

n = 100
e = 'highwayMA-v0'


run = ['./out/exp_1/run_20230914-170338_43658/checkpoint-best.tar']
 
for agent_file in run:
    for j in range(1):
        match = re.search(pattern, agent_file) 
        if match:
            folder_name = os.path.join('v_analysis/test_agent_1', match.group(0))
            print(folder_name)        
        else:
            folder_name = None
       
        
        print(agent_file)
        number_of_crashes, episode, directory, final_success_rate = test_agent_1(agent_file,  env = e, agent_config= config_11, model_json_file ='../models/dqn.json', callback_fn=callback_function_single, number_of_episodes=n, maxlen_queue = 100, option = 'zero', folder=folder_name)

        save_info_executions(file_name, agent_file, j, directory, number_of_crashes, episode, final_success_rate)

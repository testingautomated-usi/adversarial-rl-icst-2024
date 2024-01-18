import highway_env
from gymnasium.envs.registration import register

from exec_functions.retrain_agent_1 import retrain_agent_1
from utils.my_utils import callback_function_multi, save_info_executions
from exec_functions.test_agent_2 import test_agent_2

import utils.register
from config import config_11, config_ma_2

import datetime

file_name = 'test_agent_2_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

number_of_episodes = 2000
e = 'highwayMA-v0'

ego_file = './out/exp_1/run_20230914-170338_43658/checkpoint-best.tar'

run = ['./out/exp_5/run_20231010-153501_27414/checkpoint-best.tar',
'./out/exp_5/run_20231010-154135_27414/checkpoint-best.tar',
'./out/exp_5/run_20231010-154852_27414/checkpoint-best.tar',
'./out/exp_5/run_20231010-155631_27414/checkpoint-best.tar',
'./out/exp_5/run_20231010-160112_27414/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-233512_38776/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-234243_38776/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231021-235250_38776/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231022-000517_38776/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20231022-001922_38776/checkpoint-best.tar']


for adv_file in run:
    for j in range(10):
        agent_file = ''
        print('adv file: ',adv_file)
        
        number_of_crashes, episode, directory, final_success_rate = test_agent_2(ego_file, adv_file,env=e, agent_1_config=config_11, agent_2_config=config_ma_2, model_json_file='../models/dqn.json', number_of_episodes=number_of_episodes, callback_fn=callback_function_multi,maxlen_queue=100, option='zero', display=True, factor=0.5)
        save_info_executions(file_name, ego_file, j, directory, number_of_crashes, episode, final_success_rate, adv_file)




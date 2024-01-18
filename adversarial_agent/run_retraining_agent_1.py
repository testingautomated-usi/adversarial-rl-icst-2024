import highway_env
from exec_functions.retrain_agent_1 import retrain_agent_1
from utils.my_utils import callback_function_multi, save_info_executions
import utils.register

from config import config_11, config_rt, config_ma_2

import datetime

file_name = 'retrain_agent_1_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'


n = 1000
e = 'highwayMA-v0'


ego_file = './out/exp_1/run_20230914-170338_43658/checkpoint-best.tar'

'''
run = ['./out/HighwayEnvMA/DQNAgent/run_20230916-113606_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-113941_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-114310_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-114631_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-115001_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-115333_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-115656_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-120013_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-120338_39738/checkpoint-best.tar',
'./out/HighwayEnvMA/DQNAgent/run_20230916-120707_39738/checkpoint-best.tar']
'''
'''
run = ['out/HighwayEnvMA/DQNAgent/run_20230930-181629_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-182345_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-183409_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-184421_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-185257_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-185830_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-190620_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-191358_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-191945_34430/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20230930-192954_34430/checkpoint-best.tar']
'''
'''
run = ['out/HighwayEnvMA/DQNAgent/run_20231011-093244_2672/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20231011-093620_2672/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20231011-093953_2672/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20231011-094321_2672/checkpoint-best.tar',
'out/HighwayEnvMA/DQNAgent/run_20231011-094651_2672/checkpoint-best.tar']
'''

#run = ['out/HighwayEnvMA/DQNAgent/run_20231006-151017_36081/checkpoint-best.tar']
run = ['out/exp_4/run_20231006-153212_36081/checkpoint-best.tar']

for adv_file in run:
    for i in range(1):
        i = 1
        number_of_crashes, episode, directory, final_success_rate = retrain_agent_1(ego_file, adv_file, e, ego_config=config_rt, adversaril_config= config_ma_2, model_json_file='../models/dqn.json', number_of_episodes=n, callback_fn=callback_function_multi, maxlen_queue=100,option='zero', factor = 0.5)
        save_info_executions(file_name, ego_file, i, directory, number_of_crashes,episode, adv_file, final_success_rate)
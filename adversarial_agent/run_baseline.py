import highway_env
from gymnasium.envs.registration import register

from exec_functions.test_baseline import test_baseline 
from utils.my_utils import callback_function_multi

import utils.register

from config import config_11, config_ma_2


ego_file = './code/out/HighwayEnvMA/DQNAgent/run_20230810-181925_16224/checkpoint-final.tar'
number_of_episodes = 20000


test_baseline(ego_file, None, 'highwayMA-v0', agent_1_config=config_11, agent_2_config=config_ma_2, model_json_file='../models/dqn_2.json', number_of_episodes=number_of_episodes, callback_fn=callback_function_multi,maxlen_queue=100, option='zero', display=True)
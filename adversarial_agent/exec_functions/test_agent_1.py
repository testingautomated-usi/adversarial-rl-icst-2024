import gymnasium as gym
import highway_env

from rl_agents.agents.common.factory import agent_factory, load_agent_config, load_agent
from utils.our_evaluation import OurEvaluation

import copy 
import time
import pprint

#from my_utils import callback_function


def test_agent_1(agent_file, env='highway-v0', agent_config = None, model_json_file = 'models/dueling_ddqn.json', number_of_episodes = 2000, callback_fn=None, maxlen_queue = 5, option = 'zero', folder = None):

  env = gym.make(env)

  if agent_config:
    env.configure(agent_config)

  env.reset()

  # Make agent
  agent_config = load_agent_config(model_json_file)
  agent_1 = agent_factory(env, agent_config)
  agent_1.load(agent_file)

  evaluation = OurEvaluation(env, agent_1, num_episodes=number_of_episodes, display_env=True, display_agent=True, step_callback_fn=callback_fn, max_len = maxlen_queue, option = option, folder=folder)
  evaluation.test()

  number_of_crashes = evaluation.number_of_crashes
  directory = evaluation.run_directory  
  episode = evaluation.episode
  final_success_rate = evaluation.final_success_rate
  print('directory: ', evaluation.run_directory)
  print("number of crashes: ", number_of_crashes)
  print('episode: ', episode)
  print('final success rate: ', final_success_rate)
  
  pprint.pprint(env.config)

  return number_of_crashes, episode, directory, final_success_rate
  
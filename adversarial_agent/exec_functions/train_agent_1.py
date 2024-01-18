import copy 
import time
import pprint
import gymnasium as gym
import highway_env

from rl_agents.agents.common.factory import agent_factory, load_agent_config, load_agent

from utils.our_evaluation import OurEvaluation

def train_agent_1(env='highway-v0', agent_config = None, model_json_file = None, number_of_episodes=2000, callback_fn=None, maxlen_queue = 5, save_model = None, option = 'zero'):
  
  env = gym.make(env, render_mode="rgb_array")

  if agent_config:
    env.configure(agent_config)

  env.reset()

  # Make agent

  agent_config = load_agent_config(model_json_file)
  agent = agent_factory(env, agent_config)

  evaluation = OurEvaluation(env, agent, num_episodes=number_of_episodes, display_env=True, display_agent=True, step_callback_fn=callback_fn, max_len=maxlen_queue, option=option)
  evaluation.train()

  if save_model:
    agent.save(save_model+".tar")

  pprint.pprint(env.config)



import copy 
import time
import pprint
import gymnasium as gym
import highway_env

from rl_agents.agents.common.factory import agent_factory, load_agent_config, load_agent
from utils.our_evaluation import OurEvaluation

def test_retrained_agent(ego_file,adv_file,  env='highway-v0', ego_config = None, adversarial_config =None, model_json_file = 'models/dueling_ddqn.json', number_of_episodes=2000, callback_fn=None, maxlen_queue = 5, save_model = None, option = 'zero', factor= 1.0, folder = None):

    #make env for the ego
	env_1 = gym.make(env, render_mode="rgb_array")
	env_1.configure(ego_config)
	env_1.reset()
	
    #make env for the adversarial
	env_2 = gym.make(env, render_mode="rgb_array")
	env_2.configure(adversarial_config)
	env_2.reset()
	
    #make agents

	##ego
	a_c_1 = load_agent_config(model_json_file)
	agent_1 = agent_factory(env_1, a_c_1)
	agent_1.load(ego_file)
	
    ##adversarial
	a_c_2 = load_agent_config(model_json_file)
	agent_2 = agent_factory(env_2, a_c_2)
	agent_2.load(adv_file)
	evaluation = OurEvaluation(env_2, agent_2, agent_0 = agent_1, num_episodes=number_of_episodes, display_env=True, display_agent=True, step_callback_fn=callback_fn, max_len=maxlen_queue, option=option, factor=factor, folder = folder)
	#evaluation = OurEvaluation(env_1, agent_1, num_episodes=number_of_episodes, display_env=True, display_agent=True, step_callback_fn=callback_fn, max_len=maxlen_queue, option=option, factor=factor, folder = folder)
	#evaluation = OurEvaluation(env_1, agent_2, agent_0 = agent_1, num_episodes=number_of_episodes, display_env=True, display_agent=True, step_callback_fn=callback_fn, max_len=maxlen_queue, option=option, factor=factor)
	evaluation.test_agent_2()
	#evaluation.train_agent_2()
	#evaluation.test()
	number_of_crashes = evaluation.number_of_crashes
	directory = evaluation.run_directory
	episode = evaluation.episode
	final_success_rate = evaluation.final_success_rate
	print('directory: ', evaluation.run_directory)
	print("number of crashes: ", number_of_crashes)
	print('episode: ', episode)
	print("final success rate: ", final_success_rate)
	return number_of_crashes, episode, directory, final_success_rate
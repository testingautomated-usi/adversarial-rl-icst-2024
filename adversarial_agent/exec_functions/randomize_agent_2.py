import copy
import time
import pprint
import gymnasium as gym

from rl_agents.agents.common.factory import agent_factory, load_agent_config, load_agent
from utils.our_evaluation import OurEvaluation

def random_agent_2(agent_file, env='highway-v0', agent_1_config = None, agent_2_config =None, model_json_file = 'models/dueling_ddqn.json', number_of_episodes=2000, callback_fn=None, maxlen_queue = 5, save_model = None, option = 'zero', factor = 1.0, folder = None):
    
    #make env for the first agent
    env_1 = gym.make(env, render_mode = 'rgb_array')
    env_1.configure(agent_1_config)
    env_1.reset()

    #make env for the second agent
    env_2 = gym.make(env, render_mode = 'rgb_array')
    env_2.configure(agent_2_config)
    env_2.reset()

    #make agents
    ## ego
    a_c_1 = load_agent_config(model_json_file)
    agent_1 = agent_factory(env_1, a_c_1)
    agent_1.load(agent_file)

    #adversarial
    a_c_2 = load_agent_config(model_json_file)
    agent_2 = agent_factory(env_2, a_c_2)
    agent_2.load(agent_file)

    #setting random
    agent_2.config['exploration']['final_temperature'] = 1.0

    pprint.pprint(env_2.config)

    evaluation = OurEvaluation(env_2, agent_2, agent_0 = agent_1, num_episodes=number_of_episodes, display_env=True, display_agent=True, step_callback_fn=callback_fn, max_len=maxlen_queue,option=option, factor=factor, folder=folder)
    evaluation.train_agent_2()
    number_of_crashes = evaluation.number_of_crashes
    directory = evaluation.run_directory
    episode = evaluation.episode
    final_success_rate = evaluation.final_success_rate
    print('directory: ', evaluation.run_directory)
    print("number of crashes: ", number_of_crashes)
    print('episode: ', episode)

    return number_of_crashes, episode, directory, final_success_rate
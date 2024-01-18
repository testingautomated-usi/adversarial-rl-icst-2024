import copy 
import time
import pprint
import gymnasium as gym
import highway_env
import logging
import os
from matplotlib import pyplot as plt
import csv


from rl_agents.agents.common.factory import agent_factory, load_agent_config, load_agent
from rl_agents.trainer.evaluation import Evaluation
import rl_agents.trainer.logger
from gymnasium.wrappers import RecordVideo, capped_cubic_video_schedule
#from my_video_recorder import RewardTriggerRecordVideo
#from record_video import RecordVideo

#from utils import callback_function
#from agent_config import *

from collections import deque
import numpy as np
import datetime


logger = logging.getLogger(__name__)

class OurEvaluation(Evaluation):

	def __init__(self, env, agent, agent_0 = None, directory=None, run_directory=None, num_episodes=1000, training=True, sim_seed=None, recover=None, display_env=True, display_agent=True,
		 display_rewards=True, close_env=True, step_callback_fn=None, max_len=5, option = 'zero', factor = 0.1, record = False, record_thr=None, folder=None):
		
		super().__init__(env, agent, directory, run_directory, num_episodes, training, sim_seed, recover, display_env, display_agent,
			 display_rewards, close_env, step_callback_fn)

		self.agent_0 = agent_0 
		self.factor = factor
		self.number_of_crashes = 0
		self.final_success_rate = None

		self.total_reward_queue = deque(maxlen=max_len)
		self.return_queue = deque(maxlen=max_len)
		self.length_queue = deque(maxlen=max_len)
		self.success_queue = deque(maxlen=max_len)
		self.ttc_queue = deque(maxlen=max_len)
		self.col_queue = deque(maxlen=max_len)
		self.rl_queue = deque(maxlen=max_len)
		self.hs_queue = deque(maxlen=max_len)
		self.or_queue = deque(maxlen=max_len)
		self.adv_queue = deque(maxlen=max_len)

		self.option = option
		self.info = None

		self.mode = "single_agent"
		self.adv_rewards = []

		self.date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')	
		self.pid = os.getpid() 
		if folder:

			if not os.path.exists(folder):
				os.makedirs(folder)

			self.data_folder_name = os.path.join(folder,self.default_run_directory)
			video_folder_name = os.path.join(folder,self.default_run_directory)

			
		else:
			video_folder_name = './video/'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
			self.data_folder_name = './data/out/'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

		print('data folder: ', self.data_folder_name)		
		
		#if True:
		if display_env and record_thr is not None:
			#self.wrapped_env = RecordVideo(env,video_folder=video_folder_name, ttc_thr = record_thr)
			self.wrapped_env = RecordVideo(env,video_folder=video_folder_name, episode_trigger= lambda e: True)
			

			try:
				self.wrapped_env.unwrapped.set_record_video_wrapper(self.wrapped_env)
			except AttributeError:
				pass
		
	def multi_agent_step(self):
		"""
		Plan a sequence of actions for agent 2 while agent 1 acts as trained
		"""
		#print('run directory: ',self.run_directory)
		#print('directory: ',self.directory)
		#print('default directory: ', self.default_run_directory)
		print('obs: ', self.observation)
		#ego action
		obs_0 = self.modify_obs_vector(self.observation[0], option=self.option)
		action_0 = self.agent_0.plan(obs_0)

		#adversarial action
		obs_1 = self.modify_obs_vector(self.observation[1], option = self.option)
		action_1 = self.agent.plan(obs_1) 

		#print('adv obs:')
		#print(obs_1)
		
		if not ([action_0] and action_1):
			raise Exception("The agents did not plan any action")

		# Forward the actions to the environment viewer
		try:
			self.env.unwrapped.viewer.set_agent_action_sequence((action_0, action_1))
		except AttributeError:
			pass

		#step the environment
		previous_observation, action_1, action_0 = [obs_0,obs_1], action_1[0], action_0[0]
		transition = self.wrapped_env.step((action_0, action_1))
		self.observation, reward, done, truncated, info = transition
		ego_reward = info["total_rewards"][0]
		adv_reward = info["total_rewards"][1]
		obs_0 = self.modify_obs_vector(self.observation[0], option = self.option)
		
		obs_1 = self.modify_obs_vector(self.observation[1], option = self.option)
		self.info = info.copy()
		self.done = done
		self.truncated = truncated

		terminal = done or truncated

		print('calling callback')

		# Call callback function
		if self.step_callback_fn is not None:
			print('calling callback')
			self.number_of_crashes = self.step_callback_fn(self.episode, self.wrapped_env, self.agent, transition, self.writer, self.data_folder_name, self.num_episodes)
		
		info['total_rewards'] = info['total_rewards'][1]
		info['rewards'] = info['rewards'][1] 
		#info['action'] = info['action'][1]
		#info.pop('crashed_list')
		#info.pop('crashed')
		#info.pop('overtaken')
		
		i = {}
		if self.mode == "adversarial":
			print('adversarial mode')
			time_to_colision_reward = info['time_to_collision_reward']
			#total_adv_reward = time_to_colision_reward
			#total_adv_reward = time_to_colision_reward*self.factor + adv_reward
			#total_adv_reward = time_to_colision_reward + adv_reward
			total_adv_reward = adv_reward
			self.info['time_to_collision_reward'] = time_to_colision_reward
			
			self.adv_rewards.append(adv_reward)

			if terminal and info['crashed']:
				
				collision_reward = sum(self.adv_rewards)*0.1
				total_adv_reward = total_adv_reward + collision_reward
			'''
			if time_to_colision_reward > 0:
				self.adv_rewards.append(total_adv_reward)
			else:
				self.adv_rewards.append(adv_reward)
			
			

			if terminal and info['crashed']:
				
				collision_reward = sum(self.adv_rewards)*0.1
				total_adv_reward = total_adv_reward + collision_reward
			'''

			# Record the experience for adversarial agent
			i = {}
			try:
				
				#self.agent.record(previous_observation[1], action_1, time_to_colision_reward, obs_1, done, i)
				self.agent.record(previous_observation[1], action_1, total_adv_reward, obs_1, done, i)
				self.agent_0.record(previous_observation[0],action_0, ego_reward, obs_0, done, info)

			except NotImplementedError:
				
				pass

			#return total_adv_reward, terminal
			return time_to_colision_reward, terminal
		
		if self.mode == "baseline":
			print('baseline mode')
			
			try:
				#adv model
				self.agent.record(previous_observation[1], action_1, adv_reward, obs_1, done, i)
				#ego model
				self.agent_0.record(previous_observation[0],action_0, ego_reward, obs_0, done, i)
			except NotImplementedError:
				pass

			return adv_reward, terminal
		
		if self.mode == "retraining":
			print('retraining mode')

			#print('overtaken: ', info['overtaken'])
			#if terminal and info['overtaken']:
				
			#	overtaken_reward = 5
			#	ego_reward = ego_reward + overtaken_reward
		
			try:
				#adv model
				#self.agent.record(previous_observation[1], action_1, total_adv_reward, obs_1, done, i)
				#ego model
				self.agent_0.record(previous_observation[0],action_0, ego_reward, obs_0, done, i)
			except NotImplementedError:
				pass

			return adv_reward, terminal



	def step(self):

		custom_observation = self.modify_obs_vector(self.observation, option = self.option)
		actions = self.agent.plan(custom_observation)

		if not actions:
			raise Exception("The agent did not plan any action")

		# Forward the actions to the environment viewer
		try:
			self.env.unwrapped.viewer.set_agent_action_sequence(actions)
		except AttributeError:
			pass

		# Step the environment

		previous_observation, action = custom_observation, actions[0]
		transition = self.wrapped_env.step(action)
		self.observation, reward, done, truncated, info = transition
		self.info = info
		self.done = done
		self.truncated = truncated
		
		custom_observation = self.modify_obs_vector(self.observation, option = self.option)
		terminal = done or truncated

		# Call callback
		if self.step_callback_fn is not None:
			self.number_of_crashes = self.step_callback_fn(self.episode, self.wrapped_env, self.agent, transition, self.writer, self.data_folder_name, self.num_episodes)

		i = {}
		# Record the experience.
		try:
			self.agent.record(previous_observation, action, reward, custom_observation, done, i)
		except NotImplementedError:
			pass

		return reward, terminal

	def run_multi_agent_episodes(self, baseline = False, retraining=False):
		for self.episode in range(self.num_episodes):
			
			terminal = False 
			self.reset(seed=self.episode)
			rewards = []
			time_to_collision_rewards = []
			col_rewards  = []
			rl_rewards = []
			hs_rewards = []
			or_rewards = []
			adv_rewards = []
			start_time = time.time()
			
			while not terminal:

				if baseline:
					
					self.mode = "baseline"
					reward, terminal = self.multi_agent_step()
				elif retraining:
					
					self.mode = "retraining"
					reward, terminal = self.multi_agent_step()
				else:
					
					self.mode = "adversarial"
					reward, terminal = self.multi_agent_step()
					
				
				ttc_reward = self.info['time_to_collision_reward']
				col_reward = self.info['rewards'][1]['collision_reward']
				rl_reward = self.info['rewards'][1]['right_lane_reward']
				hs_reward = self.info['rewards'][1]['high_speed_reward']
				or_reward = self.info['rewards'][1]['on_road_reward']
				adv_reward = self.info["total_rewards"][1]

				rewards.append(reward)
				time_to_collision_rewards.append(ttc_reward)
				col_rewards.append(col_reward)
				rl_rewards.append(rl_reward)
				hs_rewards.append(hs_reward)
				or_rewards.append(or_reward)
				adv_rewards.append(adv_reward)

				all_rewards = [time_to_collision_rewards, col_rewards,rl_rewards,hs_rewards, or_rewards, adv_rewards]


				#Catch interruptions
				try:
					if self.env.unwrapped.done:
						break
				except AttributeError:
					pass

			#End of episode
			duration = time.time() - start_time
			self.after_all_episodes(self.episode,rewards,duration,all_rewards)
			self.after_some_episodes(self.episode,rewards)

	def after_all_episodes(self, episode, rewards, duration, all_rewards = None):

		rewards = np.array(rewards)
		gamma = self.agent.config.get("gamma", 1)
		#modification
		self.total_reward_queue.append(sum(rewards))
		self.return_queue.append(sum(r*gamma**t for t, r in enumerate(rewards)))
		self.length_queue.append(len(rewards))
		


		crashed = self.info['crashed']
		if crashed == False:
			self.success_queue.append(1)
		else:
			self.success_queue.append(0)
		
		self.writer.add_scalar('episode/length', len(rewards), episode)
		self.writer.add_scalar('episode/total_reward', sum(rewards), episode)
		self.writer.add_scalar('episode/return', sum(r*gamma**t for t, r in enumerate(rewards)), episode)
		self.writer.add_scalar('episode/fps', len(rewards) / max(duration, 1e-6), episode)
		self.writer.add_histogram('episode/rewards', rewards, episode)

		self.writer.add_scalar('mean/mean_total_reward', np.mean(self.total_reward_queue), episode)
		self.writer.add_scalar('mean/mean_return', np.mean(self.return_queue), episode)
		self.writer.add_scalar('mean/mean_length', np.mean(self.length_queue), episode)
		self.writer.add_scalar('mean/success_mean', np.mean(self.success_queue), episode)

		self.final_success_rate = np.mean(self.success_queue)

		if all_rewards:

			self.ttc_queue.append(sum(all_rewards[0]))
			self.col_queue.append(sum(all_rewards[1]))
			self.rl_queue.append(sum(all_rewards[2]))
			self.hs_queue.append(sum(all_rewards[3]))
			self.or_queue.append(sum(all_rewards[4]))
			self.adv_queue.append(sum(all_rewards[5]))

			self.writer.add_scalar('rewards/time_to_collision_reward', np.mean(self.ttc_queue), episode)
			self.writer.add_scalar('rewards/collision_reward', np.mean(self.col_queue), episode)
			self.writer.add_scalar('rewards/right_lane_reward', np.mean(self.rl_queue), episode)
			self.writer.add_scalar('rewards/high_speed_reward', np.mean(self.hs_queue), episode)
			self.writer.add_scalar('rewards/on_road_reward', np.mean(self.or_queue), episode)
			self.writer.add_scalar('rewards/adv_reward', np.mean(self.adv_queue), episode)
		
		####get statistics for plotting
		
		s = [np.mean(self.success_queue), 1-np.mean(self.success_queue), np.mean(self.total_reward_queue) ]
		
		file_name_s = os.path.join('.',self.date_time+'_'+str(self.pid)+'_statistics.csv')
		
		#file_name_s = str(self.directory)+'_statistics.csv' 
		print('filename s: ', file_name_s)
		with open(file_name_s, 'a') as s_object:
			writer_object = csv.writer(s_object)
			writer_object.writerow(s)
			s_object.close()

		logger.info("Episode {} score: {:.1f}".format(episode, sum(rewards)))

	def modify_obs_vector(self,obs_vector, option = 'zero'):

		custom_observation = np.delete(obs_vector,0,1)
		if option == 'zero':
			extra_column = np.array([0,0,0,0,0])


		if option == 'presence':
			extra_column = np.zeros(len(obs_vector))

			for i in range(len(obs_vector)):
				if obs_vector[i][1] != 0 or obs_vector[i][2] != 0:
					extra_column[i] = 1
		if option == 'negative_presence':
			extra_column = np.ones(len(obs_vector))
			for i in range(len(obs_vector)):
				x_dist = obs_vector[i][1] - obs_vector[0][1]
				if x_dist < 0:
					extra_column[i] = -1

		custom_observation = np.insert(custom_observation,0,extra_column,axis=1)
		return custom_observation
	
	def train_agent_2(self):
		self.training = True
		#change the policy of agent_0 (ego) to just act
		try:
			self.agent_0.eval()
		except AttributeError:
			pass

		self.run_multi_agent_episodes()
		self.close()
	
	def test_agent_2(self):
		self.training = False
		try:
			#changing the policy for both agents 
			self.agent_0.eval()
			self.agent.eval()
		except AttributeError:
			pass

		self.run_multi_agent_episodes(baseline=True)
		self.close()

	def retraining_mode(self):
		try:
			#changing the policy for adversarial agent
			self.agent.eval()
			#self.agent_0.eval()
		except AttributeError:
			pass

		self.run_multi_agent_episodes(retraining=True)
		self.close()
	
	




	

import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd

from rl_agents.agents.common.factory import agent_factory, load_agent_config, load_agent
from rl_agents.trainer.evaluation import Evaluation
from collections import deque
import csv
from datetime import datetime
import os

r = []
total_rewards = []
success = []
crashes = [[],[]]
total_crashes = []
success_queue = deque(maxlen=100)
total_overtaken = []

q = deque(maxlen=5)

obs_vector = []

dimension_obs_vector = 30 * 2 * 2

divisor = 20

velocities = []

#data = pd.DataFrame(columns = range(dimension_obs_vector+1))

def callback_function_multi(episode, env, agent, transition, log_writer, directory, num_episodes):
	obs, reward, terminated, truncated, info = transition
	
	print('enterring callback')
	collect_data(obs)
	#print('obs: ', obs)
	v = collect_velocities(obs)
	print('velocities: ', v)
	

	s =int(env.steps/env.config["simulation_frequency"])
	v.extend([episode,s])
	file_name_v = str(directory) + '_velocities.csv'

	print('filename v: ', file_name_v)

	with open(file_name_v,'a') as v_object:
			writer_object = csv.writer(v_object)
			writer_object.writerow(v)

	if truncated == True or terminated == True:	
		
		crashed = info['crashed_list']
		overtaken = info['overtaken']
		s = []
		
		

		for i in range(len(crashed)):
			if crashed[i] == True:
				crashes[i].append(1)
				s.append(True)
				
			else:
				crashes[i].append(0)
				s.append(False)
				
		if overtaken == True and crashed == False:
			total_overtaken.append(1)

		zero_quantity = dimension_obs_vector - len(obs_vector)
		for i in range(zero_quantity):
			obs_vector.append(0)
		
		if not(True in s):
			success.append(1)
			obs_vector.append('True')
		else:
			total_crashes.append(1)
			obs_vector.append('False')
		
		obs_vector.append(episode)
		

		file_name = str(directory) +'.csv'
		
		#file_name_v = str(directory) + 'velocities.csv'
		
		with open(file_name,'a') as f_object:
			writer_object = csv.writer(f_object)
			writer_object.writerow(obs_vector)

			f_object.close()
		'''
		with open(file_name_v,'a') as v_object:
			writer_object = csv.writer(v_object)
			writer_object.writerow(velocities)
		

		velocities.clear()
		'''
		obs_vector.clear()

		log_writer.add_scalar('success_rate/number_of_success' , len(success), episode)
		log_writer.add_scalar('success_rate/number_of_crashes' , len(total_crashes), episode)
		log_writer.add_scalar('success_rate/number_of_overtaken', len(total_overtaken), episode)
		number_crashes = len(total_crashes)
		for i in range(len(crashes)):
			log_writer.add_scalar('success_rate/number_of_crashes_vehicle_'+str(i), sum(crashes[i]), episode)
		#writer.add_scalar('success_rate/mean_success', ms , episode)
		if episode == num_episodes-1:
			success.clear()
			total_crashes.clear()
			for i in range(len(crashed)):
				crashes[i].clear()
		return number_crashes
	
	# plot graphs with all steps for an episode
	if (episode+1)%divisor == 0:
		
		ttc = info['time_to_collision_reward']
		
		
		log_writer.add_scalar('reward_per_episode/episode_'+str(episode),ttc,env.steps/5)

	reward_file = str(directory)+'_rewards.csv'
	ttc = info['time_to_collision_reward']
	#with open(reward_file,'a') as r_object:
	#	r_object.write(str(ttc) + ',')
	#	r_object.close()
	
	log_writer.add_histogram('step/ttc', ttc, env.steps/5)



def callback_function_single(episode, env, agent, transition, log_writer, directory, num_episodes):
	obs, reward, terminated, truncated, info = transition
	
	collect_velocity_single(obs,env,episode,directory)

	#metrics for the end of the episode
	if truncated == True or terminated == True:	
		crashed = info['crashed_list']
		s = []

		
		
		for i in range(len(crashed)):
			
			if crashed[i] == True:
				crashes[i].append(1)
				s.append(True)
				
			else:
				crashes[i].append(0)
				s.append(False)
		
		if not(True in s):
			success.append(1)	
		else:
			total_crashes.append(1)

		log_writer.add_scalar('success_rate/number_of_success' , len(success), episode)
		log_writer.add_scalar('success_rate/number_of_crashes' , len(total_crashes), episode)
		number_crashes = len(total_crashes)
		for i in range(len(crashes)):
			log_writer.add_scalar('success_rate/number_of_crashes_vehicle_'+str(i), sum(crashes[i]), episode)
		#writer.add_scalar('success_rate/mean_success', ms , episode)
		if episode == num_episodes-1:
			success.clear()
			total_crashes.clear()
			for i in range(len(crashed)):
				crashes[i].clear()
		return number_crashes

def collect_data(obs):
	for o in obs:
		x = o[0][1]
		y = o[0][2]
		obs_vector.extend([x,y])
	pass

def collect_velocities(obs):
	v = []
	for o in obs:
		vx = o[0][3]
		vy = o[0][4]
		v.extend([vx,vy])
	return(v)

def collect_velocity_single(obs,env,episode,directory):
	
	v = [obs[0][3],obs[0][4],0,0]

	s =int(env.steps/env.config["simulation_frequency"])
	v.extend([episode,s])
	file_name_v = str(directory) + '_velocities.csv'

	print('filename v: ', file_name_v)

	with open(file_name_v,'a') as v_object:
		writer_object = csv.writer(v_object)
		writer_object.writerow(v)

	print('v: ', v)



def add_day_and_hour_to_filename(filename):
    # Get the current date and time
    now = datetime.now()
    
    # Extract the day and hour from the current date and time
    day = now.strftime("%Y-%m-%d")
    #hour = now.strftime("%H-%M-%S")
    
    # Split the filename and extension (if any)
    filename, file_extension = os.path.splitext(filename)
    
    # Append the day and hour to the filename
    new_filename = f"{filename}_{day}_{hour}{file_extension}"

    return new_filename

def save_info_executions(file_name, agent_file, i, directory, number_of_crashes, episode,final_success_rate, adv_file =None):
	with open(file_name,'a') as object:
		writer_object = csv.writer(object)

		data = [agent_file, i, directory, number_of_crashes, episode+1, final_success_rate]

		if adv_file:
			data.append(adv_file)

		writer_object.writerow(data)

		object.close()

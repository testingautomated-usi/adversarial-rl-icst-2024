import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_name_1 = 'test_agent_2.csv'
file_name_2 = 'retrained_agent.csv'
df = pd.read_csv(file_name_1)
df_2 = pd.read_csv(file_name_2)

evx_mean = df['mean Evx'].values
avx_mean = df['mean Avx'].values
evx_std = df['std Evx'].values
steps = df['steps'].values

evx_mean = df['mean Evx'].values
avx_mean = df['mean Avx'].values
evx_std = df['std Evx'].values




plt.plot(steps, evx_mean)
plt.fill_between(steps,np.asarray(evx_mean) - np.asarray(evx_std),np.asarray(evx_mean) + np.asarray(evx_std), alpha=0.2)
plt.fill_between(steps,np.asarray(evx_mean) - np.asarray(evx_std),np.asarray(evx_mean) + np.asarray(evx_std), alpha=0.2)
plt.savefig('test_img.png', format='png')


'''

x = np.arange(len(wins_over_time_mean))
    plt.plot(x, wins_over_time_mean)
    plt.fill_between(x, np.asarray(wins_over_time_mean) - np.asarray(wins_over_time_std), np.asarray(wins_over_time_mean) + np.asarray(wins_over_time_std), alpha=0.2)
    plt.savefig("wins_over_time_uniform_0.1_1_new.png", format="png")
'''

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

def animate_trajectory(i):
    # Clear the previous plot
    plt.clf()

    # Plot the trajectory for Particle 1 and Particle 2
    plt.plot(data['Particle 1 (x)'][:i+1], data['Particle 1 (y)'][:i+1], label='Particle 1')
    plt.plot(data['Particle 2 (x)'][:i+1], data['Particle 2 (y)'][:i+1], label='Particle 2')


    # Set plot labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Trajectory at Time: {data["Time"][i]}')

    # Add legend
    plt.legend()

    # Adjust axis intervals
    plt.xticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])  # Customize the x-axis intervals
    plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # Customize the y-axis intervals

    # Adjust aspect ratio and axis scaling to capture small differences
    #plt.gca().set_aspect('equal')

datas = []

data_1 = {
    'Id': 1,
    'Cluster': 0,
    'Time': [1,2,3,4],
    'Particle 1 (x)': [-0.9421996,-0.9005391,-0.8588724,-0.8228457],
    'Particle 1 (y)': [0.34003487,0.011500538,0.00036298833,4.600523],
    'Particle 2 (x)': [-0.9206584,-0.8873809,-0.84998953,-0.81625885],
    'Particle 2 (y)': [4.0,0.40096095,0.017243141,-0.0022119572],    
}

datas.append(data_1)

data_2 = {
    'Id': 344,
    'Cluster': 0,
    'Time': [1,2,3,4],
    'Particle 1 (x)': [-0.35586348,-0.30639043,-0.25679973,-0.21016613],
    'Particle 1 (y)': [4.0,0.29623747,3.7066143,4.0108147],
    'Particle 2 (x)': [-0.34325293,-0.29703137,-0.24752896,-0.20182739],
    'Particle 2 (y)': [3.9930565,3.99989,4.0000005,4.0],    
}

datas.append(data_2)

data_3 = {
    'Id': 1658,
    'Cluster': 0,
    'Time': [1,2,3,4],
    'Particle 1 (x)': [-0.7888044,-0.73926646,-0.68966717,-0.64103836],
    'Particle 1 (y)': [-2.9449551E-05,3.7032242,0.29344285,0.0032354272],
    'Particle 2 (x)': [-0.7710934,-0.7215912,-0.6756485,-0.63258606],
    'Particle 2 (y)': [4.0,4.0,4.0,0.25136605],
}

datas.append(data_3)
data_4 = {
    'Id': 23,
    'Cluster': 1,
    'Time': [1,2,3,4],
    'Particle 1 (x)': [-0.38098437,-0.3309855,-0.28098565,-0.23169135],
    'Particle 1 (y)': [4.0,4.0,4.0,-0.70294183],
    'Particle 2 (x)': [-0.34600973,-0.30057192,-0.26240042,-0.22883391],
    'Particle 2 (y)': [4.0,4.0,4.0,1.2157182],
} 

datas.append(data_4)

data_5 = {
    'Id': 0,
    'Cluster': 1,
    'Time': [1,2,3,4],
    'Particle 1 (x)': [-0.2906587,-0.24066004,-0.19112098,-0.14717685],
    'Particle 1 (y)': [4.0,4.0,0.29678395,3.3691154],
    'Particle 2 (x)': [-0.24725315,-0.2144862,-0.18165822,-0.15224235],
    'Particle 2 (y)': [3.9989657,0.39685696,3.62816,5.244247],
} 

datas.append(data_5)

data_6 = {
    'Id': 1750,
    'Cluster': 1,
    'Time': [1,2,3,4],
    'Particle 1 (x)': [0.2098699,0.2598699,0.30986992,0.3587724],
    'Particle 1 (y)': [4.0,4.0,4.0,4.0],
    'Particle 2 (x)': [0.22000019,0.27,0.32,0.36710572],
    'Particle 2 (y)': [4.0,4.0,4.0,4.0],
} 

datas.append(data_6)



x_interval, y_interval = 1, 1  # Change these values to set your desired intervals

# Create the initial plot
fig, ax = plt.subplots()

# Create the animation
#animation = FuncAnimation(fig, animate_trajectory, frames=len(data['Time']), interval=1000, repeat=False)

# Display the animation
#plt.show()


for data in datas:
    for i in range(len(data['Time'])):
        plt.figure()
        plt.plot(data['Particle 1 (x)'][:i+1], data['Particle 1 (y)'][:i+1], label='Particle 1')
        plt.plot(data['Particle 2 (x)'][:i+1], data['Particle 2 (y)'][:i+1], label='Particle 2')
        plt.title(f'Trajectory at Time: {data["Time"][i]}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(min(min(data['Particle 1 (x)']), min(data['Particle 2 (x)'])) - 0.1,
                max(max(data['Particle 1 (x)']), max(data['Particle 2 (x)'])) + 0.1)
        plt.ylim(min(min(data['Particle 1 (y)']), min(data['Particle 2 (y)'])) - 0.1,
                max(max(data['Particle 1 (y)']), max(data['Particle 2 (y)'])) + 0.1)
        plt.legend()
        filename = f'trajectory_cluster_{data["Cluster"]}_id_{data["Id"]}_time_{data["Time"][i]}.png'
        plt.savefig(filename, dpi=100)
        plt.close()

#animation.save('cluster_0.mp4', writer='ffmpeg', fps=30)
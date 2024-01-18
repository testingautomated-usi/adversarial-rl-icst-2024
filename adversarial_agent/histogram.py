import matplotlib.pyplot as plt

ttc_list = []
with open('rewards.csv', 'r') as file:
    line = file.readline().strip()
    for value in line.split(','):
        print(value)
        try:
            ttc_list.append(float(value))
        except:
            pass


print(ttc_list)
print(len(ttc_list))

x_interval = 0.5
x_min = min(ttc_list)
print('min: ', x_min)
x_max = max(ttc_list)
print('max: ', x_max)
num_bins = int((x_max - x_min) / x_interval) + 1
print('num of bins: ', num_bins)

plt.hist(ttc_list, bins=num_bins, range=(x_min, x_max), edgecolor='black')
plt.xticks(range(int(x_min), int(x_max) + 1, 1))  # Set x-axis tick labels
#plt.yscale('log')
plt.xlabel('TTC Values')
plt.ylabel('Frequency')
plt.title('Histogram of TTC Values')
plt.show()  # Display the histogram


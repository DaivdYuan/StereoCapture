import matplotlib.pyplot as plt

# Data from the table
methods = ['StereoGesture (Ours)', 'Two-Camera Triangulation', 'Particle Filter']
noise_levels = [0.0, 0.1, 1.0, 5.0, 10.0]
mean_errors = {
    'StereoGesture (Ours)': [0.006, 0.008, 0.043, 0.169, 0.41],
    'Two-Camera Triangulation': [0.022, 0.027, 0.065, 0.270, 0.82],
    'Particle Filter': [0.76, 0.97, 1.409, 1.769, 1.910]
}

# Plotting the data
plt.figure(figsize=(10, 6))

for method in methods:
    plt.plot(noise_levels, mean_errors[method], marker='o', label=method)

# Adding titles and labels
plt.title('Mean Error vs. Noise Level for Different Methods')
plt.xlabel('Noise Level')
plt.ylabel('Mean Error')
plt.legend(loc='upper left')
# Show plot
plt.show()

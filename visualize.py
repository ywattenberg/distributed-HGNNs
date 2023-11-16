import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

N_THREADS = 20
SCRIPT_PATH = "torchtest"
CONFIG_PATH = "../config/test.yaml"
OUT_PATH = "../timing_info.txt"
RUN_SCRIPTS = False

# set working directory
os.chdir("build")

# record timing information
if RUN_SCRIPTS:
    for i in range(1, N_THREADS+1):
        with open(CONFIG_PATH, 'r') as file:
            data = yaml.safe_load(file)
            data["parallelization"]["num_threads"] = i
        with open(CONFIG_PATH, 'w') as file:
            yaml.dump(data, file)
        print(f"starting script using {i} threads")
        os.system(f"./{SCRIPT_PATH} -c {CONFIG_PATH}")

# Read the timings.txt file
with open(OUT_PATH, 'r') as file:
    lines = file.readlines()

# Parse the data and store it in the thread_runtimes dictionary
thread_runtimes = {}
current_thread_count = None

for line in lines:
    line = line.strip()
    if line[-2:] == "_t":
        current_thread_count = int(line[:-2])
        thread_runtimes[current_thread_count] = []
    elif current_thread_count is not None:
        thread_runtimes[current_thread_count].append(int(line))

# Plot histograms of runtimes for each thread count
n_runs = N_THREADS
x_size, y_size = int(n_runs/4), 4
fig, axs = plt.subplots(x_size, y_size, figsize=(8, 6))
x = np.arange(1000, 2000, 30)
for i, (thread_count, runtimes) in enumerate(thread_runtimes.items()):
    x_pos, y_pos = int(i/y_size), i%y_size
    axs[x_pos, y_pos].hist(runtimes, bins=x, edgecolor='black')
    axs[x_pos, y_pos].set_title(f'Runtimes for {thread_count} Threads')
    axs[x_pos, y_pos].set_xlabel('Runtime (ms)')
    axs[x_pos, y_pos].set_ylabel('Frequency')
    axs[x_pos, y_pos].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for ewma
import socket
import struct
import os
import logging
import time
import itertools

#15thJuly
#earlier version
port = 12341
#get the curr working directory
parent_dir = os.getcwd()
print(parent_dir)
np.random.seed(1)
# makes a directory named "plots"
directory_plots = "plots"
path_plots = os.path.join(parent_dir, directory_plots)
print(path_plots)
print(os.path.isdir(path_plots))
if(not os.path.isdir(path_plots)):
  os.mkdir(path_plots)

#makes a directory named "logs"
directory_logs = "logs"
path_logs = os.path.join(parent_dir, directory_logs)
print(os.path.isdir(path_logs))
if(not os.path.isdir(path_logs)):
    os.mkdir(path_logs)

#registering the start time of the simulation
start_time = time.time()

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
all_values_logger = logging.getLogger('ALL_VALUE_Logger')

# Create a CSV file handler for logging throuhgputs and sinr values at each iteration 
log_filename = os.path.join(path_logs, "all_values_log.csv")
all_values_csv_handler = logging.FileHandler(log_filename, mode='w')
all_values_logger.addHandler(all_values_csv_handler)

all_values_csv_handler.stream.write('Slot start, Queue Length, Reward, SINR1, SINR_URLLC, SINR2, Throughput EMBB1, Throughput EMBB2, Throughput DSNode, Fraction_0,Fraction_1,Fraction_2\n')

def log_all_values(all_values):
    all_values_logger.info(",".join(map(str, all_values)))

N = 300_000# Buffer size of 5MBytes
# P = 0.5  # Arrival probability for delay sensitive
# B = 30_000*8  # Batch size for delay sensitive

eta = 0.15 # Initial value for eta
# beta = 0.0001  # Step size for eta updates
# target_queue_length = 15000 # Define the target queue length for URLLC
# eta_values = []  # Store eta values for plotting
# eta_values.append(eta)


# TxPower = 40
# T = 308
# freq_GHz = 3.5
# freq_Hz = freq_GHz * 1e9
# B_MHz = 100
# B_Hz = B_MHz * 1e6
# eta_service = 3.5
# d = 200
# d0 = 1
# packet_size_bytes = 1500
# c = 3*(1e8)
# wavelength = c / freq_Hz
# thermal_noise = 10*np.log10(Boltzmann*T*B_Hz)
# thermal_noise = -93.8568

# Q-learning hyper parameters
gamma = 0.9
alpha = 0.1
episodes = (int)(input("Enter the number of episodes"))
iterations_per_episode = 2000  # Fixed number of iterations per episode
epsilon = 0.2

# Define the size of the state and action space
num_embb1_states = 4
num_embb2_states = 4
num_urllc_states = 4
num_queue_buckets = 40
num_actions = 6 
range1 = range(num_embb1_states)  # First three values can go from 0 to 3
range2 = range(num_embb2_states)
range3 = range(num_urllc_states)
range4 = range(num_queue_buckets)  # Last one can go from 0 to 39

# Initialize the Q-table
q_table = np.zeros((num_embb1_states, num_embb2_states, num_urllc_states, num_queue_buckets, num_actions))
average_rewards = []  # Store the average reward per episode

# def dB_to_linear(dBValue):
#     power = (dBValue/10)
#     linearValue = 10**(power)
#     return linearValue

# def log_pathloss():
#     const_term = 20*np.log10((4*np.pi*d0)/wavelength)
#     distance_term = 10*eta_service*np.log10(d/d0)
#     return const_term+distance_term

# def service_rate_model():
#     RxPower = TxPower - log_pathloss()
#     SINR_db = RxPower + 10*np.log10(-np.log(np.random.uniform(0,1))) - thermal_noise
#     SINR_linear = dB_to_linear(SINR_db)

#     R_bps = B_Hz * np.log2(1+SINR_linear)
#     R_Bps = R_bps/8
#     R_BpS = R_Bps*0.001
#     # R_pps = R_bps/(packet_size_bytes*8)

#     return R_BpS

# def simulation():
#     iterations = 10000
#     rates = []

#     for _ in range(iterations):
#         serv_rate = service_rate_model()
#         rates.append(serv_rate)

#     return rates

# rates = simulation()
# percentiles = np.percentile(rates, [20, 40, 60, 80, 100])  # Calculate percentiles
# # Define bucket boundaries based on percentiles
# bucket_boundaries = [percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]]
bucket_boundaries = [-4,0,4]

# Simulate binomial arrivals for delay sensitive node
# def simulate_arrival():
#     return np.random.binomial(B, P)

action_to_allocation = {
    0: (1, 0, 0),
    1: (0, 1, 0),
    2: (0, 0, 1),
    3: (0.5, 0.5, 0),
    4: (0.5, 0, 0.5),
    5: (0, 0.5, 0.5)
}

def discretize_queue_length(queue):
    # Define the number of buckets
    num_buckets = 30
    # Calculate the new bucket boundaries for N = 250,000
    start = int(N/num_buckets)
    bucket_boundaries = [i for i in range(start, N, start)]

    # Find the bucket index
    for i, boundary in enumerate(bucket_boundaries):
        if queue <= boundary:
            return i
    # If the queue length exceeds the maximum boundary, return the last bucket index
    return num_buckets - 1


def discretize_service_rates(sinr):
    if sinr<=bucket_boundaries[0]:
        return 0
    elif sinr>bucket_boundaries[0] and sinr<=bucket_boundaries[1]:
        return 1
    elif sinr>bucket_boundaries[1] and sinr<=bucket_boundaries[2]:
        return 2
    else:
        return 3

# def  update_queue_and_calculate_reward(embb1_state, embb2_state, urllc_state, urllc_queue, action):
#     # Determine resource allocation based on action
#     allocation = action_to_allocation[action]    
    
#     #fractinoal allocations not allowed in NetSim
#     packets_served_embb1 = allocation[0] * embb1_state 
#     packets_served_embb2 = allocation[1] * embb2_state
#     packets_served_urllc = allocation[2] * urllc_state

#     arrivals_urllc = simulate_arrival()
#     urllc_queue1 = min(max(urllc_queue + arrivals_urllc - packets_served_urllc, 0), N-1)
    
#     # Discount the reward based on the iteration number
#     reward = ((packets_served_embb1 + packets_served_embb2) - eta * urllc_queue1)
    
#     return reward, urllc_queue1, packets_served_embb1, packets_served_embb2

# Function to choose an action using epsilon-greedy policy
def choose_action(state, q_table):
    if np.random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        action = np.random.choice(num_actions)
    else:
        # Exploit: choose the best action from Q-table
        action = np.argmax(q_table[state])
    return action

def NETSIM_interface_state():
    msgType = 0
    request_value = 0
    packed_value = struct.pack(">II", msgType, request_value)

    try:
        client_socket.send(packed_value)
    except:
        return 0,[0,0,0]
    
    try:
        data = client_socket.recv(24)
    # print("Received Rates")
    except:
        return 0,[0,0,0]
    
    unpacked_data = struct.unpack(">ddd", data)
    new_states = [unpacked_data[i] for i in range(3)]

    return 1, new_states

def NETSIM_interface_queue_throughputs(action):

    msgType = 1
    packed_data = struct.pack('>II', msgType, action)

    try:
        client_socket.send(packed_data)
    # print(f"Sent action {action}")
    except:
        return 0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    
    try:
        data = client_socket.recv(4)
    except:
        return 0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    
    unpacked_data = struct.unpack('>I', data)

    msgType = 0
    request_value = 2
    packed_value = struct.pack(">II", msgType, request_value)

    try:
        client_socket.send(packed_value)
    except:
        return 0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    
    try:
        data = client_socket.recv(256)
    except:
        return 0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

    unpacked_data = struct.unpack('>dddddddd', data)
    # print(f"Received queue length {unpacked_data[0]}, reward {unpacked_data[1]}\n")

    
    return 1, unpacked_data[0], unpacked_data[1], unpacked_data[2], unpacked_data[3], unpacked_data[4], unpacked_data[5], unpacked_data[6], unpacked_data[7]

def episode_Start():
    request_value = 2
    msgType = 0
    packed_value = struct.pack(">II",msgType, request_value)

    bytes_Sent = client_socket.send(packed_value)

    data = client_socket.recv(64)
    print("Received state at episode start")

    return data

# action = 0

# Section 1: Q-learning algorithm
for episode in range(episodes):

    total_reward = 0  # Reset total reward for each episode
    
    while True:
        try:
            server_address = (socket.gethostbyname(socket.gethostname()), port)
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(server_address)
            print("Connected to the server")
            break
        except:
            continue
    
    data = episode_Start()
    unpacked_data = struct.unpack(">dddddddd", data)
    urllc_queue = unpacked_data[0]
    # new_states = [unpacked_data[i] for i in range(1,4)]
    # new_states = [(B_Hz * np.log2(1+dB_to_linear(i))) for i in new_states]

    embb1_state = (unpacked_data[5])
    urllc_state = (unpacked_data[6])
    embb2_state = (unpacked_data[7])
    urllc_queue = int(min(max(urllc_queue, 0), N - 1))

    for iteration in range(iterations_per_episode):

        embb1_state_discretized = discretize_service_rates(embb1_state)
        embb2_state_discretized = discretize_service_rates(embb2_state)
        urllc_state_discretized = discretize_service_rates(urllc_state)
        urllc_queue_discretized = discretize_queue_length(urllc_queue)
        
        # print(f"Start of iteration {iteration+1} \n ========================================================================== \n")

        # get states here
        # flag, new_states = NETSIM_interface_state()
        # print(f"New Rates for 3 nodes are, {new_states[0]}, {new_states[1]}, {new_states[2]}\n")
        
        # if flag == 0:
            # print("flag is 0_state")
            # continue
        # eta += beta * (urllc_queue - target_queue_length)/10000
        # eta = max(0, min(eta, 6))
        # eta_values.append(eta)  # Store eta value for plotting        
        
        # action_prev = action
        # Choose action based on current state using epsilon-greedy policy
        action = choose_action((embb1_state_discretized, embb2_state_discretized, urllc_state_discretized, urllc_queue_discretized), q_table)

        # eta += beta * (urllc_queue - target_queue_length)
        # eta = max(0, min(eta, 6))
        # eta_values.append(eta)  # Store eta value for plotting

        # reward, new_urllc_queue, _, _ = update_queue_and_calculate_reward(embb1_state, embb2_state, urllc_state, urllc_queue, action)
        flag, new_urllc_queue, reward, throughput1, throughput2, throughputDS, new_embb1_state, new_urllc_state, new_embb2_state = NETSIM_interface_queue_throughputs(action)
        
        if flag == 0:
            # new_urllc_queue = urllc_queue
            print("flag is 0_queue")
            continue

        new_embb1_state_discretized = discretize_service_rates(new_embb1_state)
        new_embb2_state_discretized = discretize_service_rates(new_embb2_state)
        new_urllc_state_discretized = discretize_service_rates(new_urllc_state)
        
        # log_all_values([new_urllc_queue, reward, throughput1, throughput2, throughputDS, action_to_allocation[action_prev][0], action_to_allocation[action_prev][1], action_to_allocation[action_prev][2]])
        log_all_values([iteration, new_urllc_queue, reward, new_embb1_state, new_urllc_state, new_embb2_state, throughput1, throughput2, throughputDS, action_to_allocation[action][0], action_to_allocation[action][1], action_to_allocation[action][2]])
        
        new_urllc_queue = int(min(max(new_urllc_queue, 0), N - 1))
        new_urllc_queue_discretized = discretize_queue_length(new_urllc_queue)

        q_table[embb1_state_discretized, embb2_state_discretized, urllc_state_discretized, urllc_queue_discretized, action] = \
        (1 - alpha) * q_table[embb1_state_discretized, embb2_state_discretized, urllc_state_discretized, urllc_queue_discretized, action] \
        + alpha * (reward + gamma*np.max(q_table[new_embb1_state_discretized, new_embb2_state_discretized, new_urllc_state_discretized, new_urllc_queue_discretized]))

        # Update URLLC queue state
        urllc_queue = new_urllc_queue
        embb1_state = new_embb1_state
        embb2_state = new_embb2_state
        urllc_state = new_urllc_state

        total_reward += reward
        print(f"Training | Episode {episode} | Iteration {iteration}: UE1 Throughput = {throughput1}, UE2 Throughput = {throughput2}")
        # print(f"Episode {episode}, Time_Step {iteration}")
        # print(f" ========================================================================== \nEnd of iteration {iteration+1}\n")
    
    average_reward = total_reward / iterations_per_episode
    average_rewards.append(average_reward)
    print(f"Episode {episode}: Completed")
    client_socket.close()

# Extract the policy: for each state, choose the action with the highest Q-value
policy = np.argmax(q_table, axis=4)
# df = pd.DataFrame(policy)
# df.to_csv("Arthur.csv")
# arr_length = len(eta_values)
# index = 0.9*arr_length
# eta = np.mean(eta_values[(int)(index):])
# Generate all combinations
combinations = list(itertools.product(range1, range2, range3, range4))
# Generate the final list by combining each combination with the corresponding Q-table values
final_list = [list(combo) + list(q_table[combo[0], combo[1], combo[2], combo[3]]) for combo in combinations]
df = pd.DataFrame(final_list, columns = ["URLLC1","URLLC2","DS","Queue","1","2","3","4","5","6",])
df.to_csv("Saved_table.csv")

# Section 2: Policy evaluation
# Simulation parameters
i = 0
episodes = 10
time_steps = episodes*iterations_per_episode
queue_lengths = np.zeros(time_steps)  # Store queue length at each time step
arrivals = np.zeros(time_steps)       # Store arrivals at each time step
packets_served_embb1_arr = np.zeros(time_steps)
packets_served_embb2_arr = np.zeros(time_steps)

for episode in range(episodes):

    while True:
        try:
            server_address = (socket.gethostbyname(socket.gethostname()), port)
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(server_address)
            print("Connected to the server for policy evaluation")
            break
        except:
            continue
     
    if episode == 0: 
        data = episode_Start()
        unpacked_data = struct.unpack(">dddddddd", data)
        urllc_queue = unpacked_data[0]
        # new_states = [unpacked_data[i] for i in range(1,4)]
        # new_states = [(B_Hz * np.log2(1+dB_to_linear(i))) for i in new_states]

        embb1_state = (unpacked_data[5])
        urllc_state = (unpacked_data[6])
        embb2_state = (unpacked_data[7])
        urllc_queue = int(min(max(urllc_queue, 0), N - 1))
    else:
        x = episode_Start()


    for t in range(iterations_per_episode):

        embb1_state_discretized = discretize_service_rates(embb1_state)
        embb2_state_discretized = discretize_service_rates(embb2_state)
        urllc_state_discretized = discretize_service_rates(urllc_state)
        urllc_queue_discretized = discretize_queue_length(urllc_queue)


        # flag, new_states = NETSIM_interface_state()
        
        # if flag == 0:
            # print("flag is 0_state")
            # continue

        # Choose the best action based on the current state and policy
        action = policy[embb1_state_discretized, embb2_state_discretized, urllc_state_discretized, urllc_queue_discretized]
        
        # Get reward and new queue length based on the chosen action
        flag, urllc_queue, reward, packets_served_embb1, packets_served_embb2, packets_served_DS,embb1_state,urllc_state,embb2_state = NETSIM_interface_queue_throughputs(action)
        
        if flag == 0:
            continue
        
        urllc_queue = int(min(max(urllc_queue, 0), N - 1))
        packets_served_embb1_arr[i] = packets_served_embb1
        packets_served_embb2_arr[i] = packets_served_embb2
        queue_lengths[i] = urllc_queue         
        log_all_values([t, urllc_queue, reward, embb1_state,urllc_state,embb2_state, packets_served_embb1, packets_served_embb2, packets_served_DS, action_to_allocation[action][0], action_to_allocation[action][1], action_to_allocation[action][2]])
        i+=1
        print(f"Evaluation | Episode {episode} | Time Step {t}: UE1 Throughput = {throughput1}, UE2 Throughput = {throughput2}")


# Calculate EWMA throughputs
ewma_embb1_throughputs = pd.Series(packets_served_embb1_arr).ewm(span=25).mean()
ewma_embb2_throughputs = pd.Series(packets_served_embb2_arr).ewm(span=25).mean()

# Calculate overall average throughput and average queue length
overall_avg_throughput_ue1 = np.mean(packets_served_embb1_arr)
overall_avg_throughput_ue2 = np.mean(packets_served_embb2_arr)
average_queue_length_urllc = np.mean(queue_lengths)

print(f"Overall average throughput of eMBB UE1: {overall_avg_throughput_ue1:.2f} Bytes/Slot")
print(f"Overall average throughput of eMBB UE2: {overall_avg_throughput_ue2:.2f} Bytes/Slot")
print(f"Average queue length of URLLC: {average_queue_length_urllc:.2f} Bytes")

# Adjusting figure size and layout to accommodate legend below the plot
# plt.figure(figsize=(10, 5))  # Increased vertical size
# ax1 = plt.gca()
# embb1_line, = ax1.plot(ewma_embb1_throughputs, label='eMBB1 EWMA Throughput', color='blue', linewidth=1)
# embb2_line, = ax1.plot(ewma_embb2_throughputs, label='eMBB2 EWMA Throughput', color='orange', linewidth=1)
# ax1.set_xlabel('Time (Slots)')
# ax1.set_ylabel('Throughput (Bytes/Slot)')
# ax1.tick_params(axis='y')
# # # Create a second Y-axis for queue length with adjusted parameters
# ax2 = ax1.twinx()
# queue_line, = ax2.plot(queue_lengths, label='Queue Length', color='orange', linewidth=1)
# ax2.set_ylabel('Queue Length (Bytes)')
# ax2.tick_params(axis='y')
# # Placing the legend below the plot, ensuring it is within the figure bounds
# lines = [embb1_line, embb2_line, queue_line]
# labels = [line.get_label() for line in lines]
# ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
# plt.title(f'Optimal policy; eMBB throughputs and DS queue length\nLagrange multiplier fixed at {eta}; DS node buffer is 300KB.')
# plt.grid(True)
# plt.tight_layout()  # Adjust layout to prevent overlap and ensure everything fits
# plot_filename = os.path.join(path_plots, "Throughputs_Queue_Lenghts.png")
# plt.savefig(plot_filename)
# plt.show()

plt.figure(figsize=(10, 5))  # Increased vertical size
ax1 = plt.gca()
embb1_line, = ax1.plot(ewma_embb1_throughputs, label='UE1  Throughput', color='blue', linewidth=1)
embb2_line, = ax1.plot(ewma_embb2_throughputs, label='UE2  Throughput', color='orange', linewidth=1)
ax1.set_xlabel('Time (Slots)')
ax1.set_ylabel('Throughput (Bytes/Slot)')
ax1.tick_params(axis='y')
lines = [embb1_line, embb2_line]
labels = [line.get_label() for line in lines]
# ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
# Set the legend inside the grid, in the top-right corner
plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0.1)

plt.title(f'Optimal policy; eMBB UE throughputs\nLagrange multiplier fixed at {eta}; URLLC node buffer is 300KB.')
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlap and ensure everything fits
plot_filename = os.path.join(path_plots, "Throughputs.png")
plt.savefig(plot_filename)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(average_rewards, label='Average Reward per Episode', color='green', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
title = f'Average reward per episode; scenario with 2 eMBB nodes and 1 low latency node.\nLagrange multiplier fixed at {eta}.'
plt.title(title)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
# Set the legend inside the grid, in the top-right corner
plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0.1)

# Adjust the layout to fit everything within the figure
plt.tight_layout()
plot_filename = os.path.join(path_plots, "average_rewards.png")
plt.savefig(plot_filename)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(queue_lengths, label='URLLC Queue Lengths', color='purple', linewidth=1)
plt.xlabel('Time (Slots)')
plt.ylabel('Queue length (Bytes)')
title = f'Optimal policy; URLLC queue length\nLagrange multiplier fixed at {eta}; URLLC node buffer is 300KB.'
plt.title(title)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
# Set the legend inside the grid, in the top-right corner
plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0.1)

# Adjust the layout to fit everything within the figure
plt.tight_layout()
plot_filename = os.path.join(path_plots, "Queue_lengths.png")
plt.savefig(plot_filename)
plt.show()

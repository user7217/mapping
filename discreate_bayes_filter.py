import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# 0 = Wall, 1 = Door
world_map = ['wall', 'door', 'wall', 'wall', 'door'] 
grid_size = len(world_map)

# Sensor Accuracy
p_hit = 0.8    # Correctly identifies door/wall
p_miss = 0.2   # False reading

# Motion Accuracy
p_exact = 0.8  # Robot moves exactly 1 step
p_overshoot = 0.1 # Robot moves 2 steps (slip)
p_undershoot = 0.1 # Robot moves 0 steps (stuck)

# Initial Belief: Uniform (I could be anywhere)
p = np.array([1.0/grid_size] * grid_size)

# --- 2. THE ALGORITHMS ---

def sense(p, measurement):
    """
    UPDATE STEP: Multiply belief by probability of measurement.
    Answers: "Given I saw a door, how likely is it I am in cell X?"
    """
    q = np.zeros_like(p)
    
    for i in range(len(p)):
        # Check if the map at this cell matches the measurement
        is_match = (world_map[i] == measurement)
        
        # Apply Bayes Rule
        hit = p_hit if is_match else p_miss
        q[i] = p[i] * hit
        
    # NORMALIZE (Probabilities must sum to 1.0)
    q = q / np.sum(q)
    return q

def move(p, steps=1):
    """
    PREDICT STEP: Convolution.
    Shifts the probability array to the right, adding 'motion blur' (noise).
    """
    q = np.zeros_like(p)
    
    for i in range(len(p)):
        # Exact move
        target = (i + steps) % len(p)
        q[target] += p[i] * p_exact
        
        # Overshoot
        target = (i + steps + 1) % len(p)
        q[target] += p[i] * p_overshoot
        
        # Undershoot
        target = (i + steps - 1) % len(p)
        q[target] += p[i] * p_undershoot
        
    return q

# --- 3. THE SIMULATION & VISUALIZATION ---

def plot_histogram(p, step_name, true_pos):
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(p)), p, color='blue', alpha=0.6, label='Belief')
    plt.axvline(x=true_pos, color='red', linestyle='--', linewidth=3, label='Actual Robot Pos')
    
    # Label the map on x-axis
    plt.xticks(range(len(world_map)), world_map)
    plt.ylim(0, 1.0)
    plt.title(f"Step: {step_name}")
    plt.legend()
    plt.show()

# Actual Robot State (Ground Truth)
robot_pos = 0 

# Show Initial State
plot_histogram(p, "Start (Uniform Belief)", robot_pos)

# SEQUENCE 1: SENSE "WALL"
# Robot is at 0 (Wall). Sensor says "Wall".
measurement = 'wall'
p = sense(p, measurement)
plot_histogram(p, "1. Sensed 'Wall' (Sharpen)", robot_pos)

# SEQUENCE 2: MOVE 1 STEP RIGHT
# Robot moves to index 1 (Door).
robot_pos = 1
p = move(p, 1)
plot_histogram(p, "2. Moved Right (Blur)", robot_pos)

# SEQUENCE 3: SENSE "DOOR"
# Robot is at 1 (Door). Sensor says "Door".
measurement = 'door'
p = sense(p, measurement)
plot_histogram(p, "3. Sensed 'Door' (Sharpen)", robot_pos)

# SEQUENCE 4: MOVE 1 STEP RIGHT
robot_pos = 2
p = move(p, 1)
plot_histogram(p, "4. Moved Right (Blur)", robot_pos)
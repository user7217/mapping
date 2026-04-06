import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. CONFIGURATION (THE PHYSICS) ---
GRID_SIZE = 50       # 50x50 cells
RESOLUTION = 1.0     # 1 meter per cell
WALL_POS = 40        # The actual wall is at x=40

# PROBABILITIES (The "Trust" Levels)
P_OCC = 0.9          # If laser hits, 90% sure it's a wall
P_FREE = 0.3         # If laser passes through, 30% chance it's a wall (70% free)
P_PRIOR = 0.5        # Initial belief (50/50 unknown)

# CONVERT TO LOG-ODDS (The Math Trick)
# Formula: l = log(p / (1-p))
L_OCC = np.log(P_OCC / (1 - P_OCC))
L_FREE = np.log(P_FREE / (1 - P_FREE))
L_PRIOR = np.log(P_PRIOR / (1 - P_PRIOR)) # This is usually 0.0

# --- 2. THE ALGORITHM ---
    
def inverse_sensor_model(robot_x, measured_dist):
    """
    Decides what to do with the map based on ONE single laser ray.
    Returns: A list of (index, log_odds_change) to apply.
    """
    updates = []
    
    # Calculate which cell the laser HIT
    # robot is at 0, hit is at measured_dist
    hit_cell = int(measured_dist / RESOLUTION)
    
    # ZONE 1: FREE SPACE (Ray passed through these cells)
    for cell_idx in range(robot_x + 1, hit_cell):
        # Safety: Don't update off the map
        if 0 <= cell_idx < GRID_SIZE:
            updates.append((cell_idx, L_FREE)) # Decrease score
            
    # ZONE 2: OBSTACLE (Ray stopped here)
    if 0 <= hit_cell < GRID_SIZE:
        updates.append((hit_cell, L_OCC))      # Increase score
        
    return updates

# --- 3. THE SIMULATION LOOP ---

# Initialize Map with 0.0 (Unknown)
# We act as a 1D line of cells for simplicity, but math works for 2D.
grid_map = np.zeros(GRID_SIZE) 

# Simulation: Robot stands at x=0 and shoots 100 lasers at the wall
print("Simulation starting... Robot at x=0. Wall at x=40.")

for i in range(100):
    # A. GENERATE REALITY (With Noise)
    # The sensor isn't perfect. It has +/- 2.0 meters error!
    noise = random.gauss(0, 1.0) 
    reading = WALL_POS + noise
    
    # B. CALCULATE UPDATE
    updates = inverse_sensor_model(0, reading)
    
    # C. UPDATE MAP (Bayes Filter)
    for index, value in updates:
        # Simple Addition! (The power of Log-Odds)
        grid_map[index] += value

# --- 4. VISUALIZATION ---

# Convert Log-Odds back to Probability for humans
# Formula: p = 1 - (1 / (1 + exp(l)))
probabilities = 1 - (1 / (1 + np.exp(grid_map)))

plt.figure(figsize=(10, 5))
plt.plot(probabilities, label="Belief (Probability)")
plt.axvline(x=WALL_POS, color='r', linestyle='--', label="True Wall Position")
plt.title(f"Occupancy Grid after 100 Noisy Scans")
plt.xlabel("Distance (Grid Cells)")
plt.ylabel("Probability of Obstacle (0=Free, 1=Wall)")
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION ---
WORLD_SIZE = 100.0
LANDMARKS = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]] # 4 Beacons
N_PARTICLES = 1000

# --- 1. THE PARTICLE CLASS ---
class Particle:
    def __init__(self):
        # Random position (Kidnapped!)
        self.x = random.random() * WORLD_SIZE
        self.y = random.random() * WORLD_SIZE
        self.orientation = random.random() * 2.0 * np.pi
        self.weight = 1.0 # Importance score

    def set(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation
        
    def move(self, turn, forward):
        """
        PREDICT STEP: Move the particle, but ADD NOISE.
        We can't trust the odometer 100%.
        """
        self.orientation = self.orientation + turn + random.gauss(0.0, 0.1)
        self.orientation %= 2 * np.pi
        
        dist = forward + random.gauss(0.0, 1.0) # Noisy movement
        self.x += np.cos(self.orientation) * dist
        self.y += np.sin(self.orientation) * dist
        
        # Cyclic World (Pacman style) for simplicity
        self.x %= WORLD_SIZE
        self.y %= WORLD_SIZE

    def measurement_prob(self, measurement):
        """
        UPDATE STEP: Calculate how likely this particle is accurate.
        Gaussian Probability: P(z | x) 
        """
        prob = 1.0
        sigma = 5.0 # Sensor Noise (Uncertainty)
        
        for i in range(len(LANDMARKS)):
            # 1. Calculate distance from THIS particle to the landmark
            lx, ly = LANDMARKS[i]
            dist_p = np.sqrt((self.x - lx)**2 + (self.y - ly)**2)
            
            # 2. Compare with the REAL robot's measurement
            dist_r = measurement[i]
            
            # 3. Gaussian: Peak at 0 error, drops off as error increases
            # Formula: e ^ (- (error^2) / (2 * sigma^2))
            prob *= np.exp(- ((dist_p - dist_r) ** 2) / (2 * (sigma ** 2)))
            
        self.weight = prob
        return prob

# --- 2. THE SIMULATION ---

# Create the Swarm
particles = [Particle() for i in range(N_PARTICLES)]

# Create the Actual Robot (Ground Truth)
my_robot = Particle()
my_robot.set(50, 50, 0) # Start in middle

print("Simulation Starting...")

for t in range(10): # Run for 10 steps
    
    # A. ROBOT MOVES (Hidden from filter)
    my_robot.move(0.1, 5.0) # Turn 0.1 rad, Move 5.0 units
    
    # B. ROBOT SENSES (Ground Truth + Noise)
    # The robot measures distance to all 4 landmarks
    Z = []
    for lx, ly in LANDMARKS:
        dist = np.sqrt((my_robot.x - lx)**2 + (my_robot.y - ly)**2)
        Z.append(dist + random.gauss(0.0, 1.0)) # Sensor noise
        
    # --- PARTICLE FILTER CYCLE ---
    
    # 1. PREDICT: Move all particles
    for p in particles:
        p.move(0.1, 5.0) 
        
    # 2. UPDATE: Weigh particles based on Sensor Z
    weights = []
    for p in particles:
        w = p.measurement_prob(Z)
        weights.append(w)
        
    # 3. RESAMPLE (The Wheel)
    # Pick particles with high weight more often
    # This is "Survival of the Fittest"
    new_particles = []
    index = int(random.random() * N_PARTICLES)
    beta = 0.0
    max_w = max(weights)
    
    for i in range(N_PARTICLES):
        beta += random.random() * 2.0 * max_w
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N_PARTICLES
        
        # Create a deep copy of the chosen particle
        p2 = Particle()
        p2.set(particles[index].x, particles[index].y, particles[index].orientation)
        new_particles.append(p2)
        
    particles = new_particles

    # --- VISUALIZATION ---
    # Plot every 2nd step
    if t % 2 == 0:
        plt.figure(figsize=(6, 6))
        # Draw Particles (Blue Dots)
        px = [p.x for p in particles]
        py = [p.y for p in particles]
        plt.scatter(px, py, color='blue', s=2, alpha=0.5, label='Particles')
        
        # Draw Real Robot (Red Star)
        plt.scatter([my_robot.x], [my_robot.y], color='red', marker='*', s=200, label='Robot')
        
        # Draw Landmarks (Green Squares)
        lx = [l[0] for l in LANDMARKS]
        ly = [l[1] for l in LANDMARKS]
        plt.scatter(lx, ly, color='green', marker='s', s=100, label='Landmarks')
        
        plt.xlim(0, 100); plt.ylim(0, 100)
        plt.legend(loc='upper right')
        plt.title(f"Step {t}: Localization")
        plt.show()

print("Robot Location:", my_robot.x, my_robot.y)
# Estimate is average of all particles
est_x = sum([p.x for p in particles]) / N_PARTICLES
est_y = sum([p.y for p in particles]) / N_PARTICLES
print("Filter Estimate:", est_x, est_y)
import numpy as np
import matplotlib.pyplot as plt
import random
import math

#config
WORLD_SIZE = 100.0
#map hidden from robot
REAL_LANDMARKS = {
    1: [20.0, 20.0], 
    2: [80.0, 80.0], 
    3: [20.0, 80.0], 
    4: [80.0, 20.0]
}
N_PARTICLES = 100

def normalize_angle(angle):
    """Keep angle between -pi and pi"""
    while angle > np.pi: angle -= 2.0 * np.pi
    while angle < -np.pi: angle += 2.0 * np.pi
    return angle

class Particle:
    def __init__(self, x=None, y=None, theta=None):
        # 1. LOCALIZATION: Where am I?
        if x is None:
            self.x = random.random() * WORLD_SIZE
            self.y = random.random() * WORLD_SIZE
            self.theta = random.random() * 2.0 * np.pi
        else:
            self.x = x; self.y = y; self.theta = theta
            
        self.weight = 1.0
        
        #mapping
        self.landmarks = {}

    def move(self, turn, forward):
        # add noise
        self.theta += turn + random.gauss(0.0, 0.1)
        self.theta = normalize_angle(self.theta)
        
        dist = forward + random.gauss(0.0, 1.0)
        self.x += np.cos(self.theta) * dist
        self.y += np.sin(self.theta) * dist
        self.x %= WORLD_SIZE
        self.y %= WORLD_SIZE

    def update_map_and_probability(self, measurement):
        
        dist_m, angle_m, lm_id = measurement
                # Calculate coordinate of the observed landmark in the world
        lx_obs = self.x + dist_m * np.cos(self.theta + angle_m)
        ly_obs = self.y + dist_m * np.sin(self.theta + angle_m)
        
        #first time seeing this landmark
        if lm_id not in self.landmarks:
            # Initialize it in this particle's personal map
            self.landmarks[lm_id] = [lx_obs, ly_obs, 1]
            return 1.0 
        
        #seen before check for consistency
        else:
            prev_lx, prev_ly, count = self.landmarks[lm_id]
            
            #calc error
            dist_error = np.sqrt((lx_obs - prev_lx)**2 + (ly_obs - prev_ly)**2)
            
            #update gaussian prob
            sigma = 2.0
            prob = np.exp(- (dist_error ** 2) / (2 * sigma ** 2))
            
            # update map (running average)
            new_count = count + 1
            avg_lx = (prev_lx * count + lx_obs) / new_count
            avg_ly = (prev_ly * count + ly_obs) / new_count
            
            self.landmarks[lm_id] = [avg_lx, avg_ly, new_count]
            
            return prob

#simulation
my_robot = Particle(50, 50, 0)

# random swarm
particles = [Particle() for i in range(N_PARTICLES)]

print("Running FastSLAM...")

for t in range(20):
    #move bot
    my_robot.move(0.1, 4.0)
    
    # robot senses the landmarks
    observations = []
    for lm_id, coords in REAL_LANDMARKS.items():
        # Calculate true distance/angle
        dx = coords[0] - my_robot.x
        dy = coords[1] - my_robot.y
        dist_true = np.sqrt(dx**2 + dy**2)
        angle_true = math.atan2(dy, dx) - my_robot.theta
        
        # nosie
        z_dist = dist_true + random.gauss(0, 1.0)
        z_angle = normalize_angle(angle_true + random.gauss(0, 0.05))
        
        # only add if in range
        if z_dist < 40.0:
            observations.append([z_dist, z_angle, lm_id])

    #slam    
    # predict
    for p in particles:
        p.move(0.1, 4.0)
        
    # update
    weights = []
    for p in particles:
        p_weight = 1.0
        for obs in observations:
            # update map and get prob
            p_weight *= p.update_map_and_probability(obs)
        
        p.weight = p_weight
        weights.append(p_weight)
        
    # resample
    if sum(weights) > 0:
        # norm weights
        weights = [w/sum(weights) for w in weights]
        
        # new particle set
        new_particles = []
        # indices based on weights
        indices = np.random.choice(N_PARTICLES, N_PARTICLES, p=weights)
        
        for i in indices:
            parent = particles[i]
            child = Particle(parent.x, parent.y, parent.theta)
            child.landmarks = parent.landmarks.copy() # Copy the personal map
            new_particles.append(child)
            
        particles = new_particles

    if t % 5 == 0:
        plt.figure(figsize=(6, 6))
        
        # plot robot
        plt.scatter(my_robot.x, my_robot.y, c='r', marker='*', s=150, label='True Pos')
        
        #plot best particle map
        best_p = particles[0]
        plt.scatter(best_p.x, best_p.y, c='b', label='Est Pos')
        
        # Plot the Landmarks estimated by this particle
        for l_id, coords in best_p.landmarks.items():
            plt.scatter(coords[0], coords[1], c='orange', marker='s', s=50)
            plt.text(coords[0]+2, coords[1], f"L{l_id}")
            
        # plot true landmarks
        for l_id, coords in REAL_LANDMARKS.items():
            plt.scatter(coords[0], coords[1], facecolors='none', edgecolors='k', marker='s', s=80, label='True Map' if l_id==1 else "")

        plt.xlim(0, 100); plt.ylim(0, 100)
        plt.title(f"FastSLAM Step {t}\n(Particle is building the map)")
        plt.legend()
        plt.show()
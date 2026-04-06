import numpy as np
import matplotlib.pyplot as plt
import math

#config
GRID_SIZE = 100
CENTER = (50, 50)
L_OCC = 0.9
L_FREE = -0.7

#initialize the map
grid_map = np.zeros((GRID_SIZE, GRID_SIZE))

#geometry engine (bresenham's line algorithm)
def bresenham(start, end):
    #returns x, y coords for a line between start and end
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    points = []
    while True:
        points.append((x0, y0))
        if x0==x1 and y0==y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

#scanner 
def get_laser_reading(angle_deg):
    #simulate a laser hitting a sqaure room
    angle_rad = math.radians(angle_deg)
    dist = 0
    while True:
        dist+= 0.5 #step size
        dx = dist * math.cos(angle_rad)
        dy = dist * math.sin(angle_rad)
        
        px = int(CENTER[0] + dx)
        py = int(CENTER[1] + dy)
        
        #stop if wall is hit
        if px <= 20 or px >= 80 or py <= 20 or py >= 80:
            return (px, py)
        
print("Scanning the room")

for angle in range(0, 360, 2):
    hit_x, hit_y = get_laser_reading(angle)
    
    #get path
    path = bresenham(CENTER, (hit_x, hit_y))
    
    #update map with bayes filter
    for (px, py) in path:
        if 0 <= px < GRID_SIZE and 0 <= py <GRID_SIZE:
            
            #if its the last cell its a wall
            if (px, py) == (hit_x, hit_y):
                grid_map[px][py] += L_OCC
            else:
                grid_map[px][py] += L_FREE
                
                
#visualization

prob_map = 1 - (1 / (1 + np.exp(grid_map)))

plt.figure(figsize=(8, 8))
plt.imshow(prob_map, cmap='gray', origin='lower')
plt.colorbar(label='Occupancy Probability')
plt.scatter([CENTER[0]], [CENTER[1]], color='red', s=100, label='Robot Position')
plt.title('2D Grid Map Occupancy')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
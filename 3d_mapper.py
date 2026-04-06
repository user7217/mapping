import numpy as np 
import matplotlib.pyplot as plt

#config 
GRID_SIZE = 30
CENTER = (15, 15, 15)
L_OCC = 0.9
L_FREE = -0.7

#map
voxel_map = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))

#3d ray tracing
def get_voxels_along_ray(start, end):
    x1, y1, z1 = start
    x2, y2, z2 = end
    
    #how far is the target
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    if dist == 0: return []
    
    steps = int(max(abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)))
    voxels = []
    
    for i in range(steps + 1):
        t = i / steps #interpolation factor
        
        #interpolate position
        ix = int(round(x1 + t * (x2 - x1)))
        iy = int(round(y1 + t * (y2 - y1)))
        iz = int(round(z1 + t * (z2 - z1)))
        
        if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE and 0 <= iz < GRID_SIZE:
            voxels.append((ix, iy, iz))
    return voxels

#scanner
def get_lidar_hit(azimuth, elevation):
    #simulates a laser hitting a sphere in a room
    
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    dx = np.cos(el_rad) * np.cos(az_rad)
    dy = np.cos(el_rad) * np.sin(az_rad)
    dz = np.sin(el_rad)
    
    #march teh ray forward from the center
    cx, cy, cz = CENTER
    
    for dist in range(1, 15):
        cx+= dx
        cy+= dy
        cz+= dz
        
        ix, iy, iz = int(round(cx)), int(round(cy)), int(round(cz))
        
        dist_from_center = np.sqrt((cx-15)**2 + (cy-15)**2 + (cz-15)**2)
        
        if 6.5 <= dist_from_center <= 7.5:
            return (ix, iy, iz) #hit
    return None #no hit within range

#the mapping loop

for az in range(0, 360, 10):
    for el in range(-90, 90, 10):
        
        #fire the laser
        hit_voxel = get_lidar_hit(az, el)
        if hit_voxel:
            path = get_voxels_along_ray(CENTER, hit_voxel)
            #update map
            for v in path:
                vx, vy, vz = v
                if v == hit_voxel:
                    voxel_map[vx][vy][vz] += L_OCC
                else:
                    voxel_map[vx][vy][vz] += L_FREE
#visualization
print("Plotting...")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Get coordinates of all voxels that are likely obstacles (> 0.5 log odds)
occupied = np.argwhere(voxel_map > 1.0)

# Plot them as red dots
ax.scatter(occupied[:, 2], occupied[:, 1], occupied[:, 0], c='red', marker='s')
ax.set_title("3D Voxel Map (Sphere)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(0, 30); ax.set_ylim(0, 30); ax.set_zlim(0, 30)

plt.show()
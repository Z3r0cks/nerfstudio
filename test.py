import open3d as o3d
import numpy as np
import csv
from numpy import genfromtxt

data = genfromtxt('C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/cube_interference.csv', delimiter=',')

#if distance in data = 0, delete row
# data = data[~(data == 0).all(1)]

print(data)

# lidar specs
opening_angle = 276  # in Grad
angle_resolution = 0.1  # in Grad
num_points = len(data)

# calc angular
angles = np.linspace(-opening_angle / 2, opening_angle / 2, num_points)
angles_rad = np.radians(angles)

x = data * np.cos(angles_rad)
y = data * np.sin(angles_rad)
z = np.zeros(num_points)

# point cloud
points = np.vstack((x, y, z)).T
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([point_cloud]) # type: ignore

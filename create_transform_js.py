import json
import os

import numpy as np

# This code is for creating a transforms.json with known camera parameter.

def create_transform_json(cameras):  
   transform_json = {
      "w": cameras["params"]["w"],
      "h": cameras["params"]["h"],
      "fl_x": cameras["params"]["fl_x"],
      "fl_y": cameras["params"]["fl_y"],    
      "cx": cameras["params"]["w"] / 2,
      "cy": cameras["params"]["h"] / 2,
      "k1": 0.001193459470544891,
      "k2": 0.0,
      "p1": 0.0,
      "p2": 0.0,
      "camera_model": "OPENCV",
      "frames": []
   }
   
   for camera in cameras["camera_list"]:
      pos = camera["pos"]
      final_rot_matrix = np.identity(3)
      
      if camera["rotation_list"][0] == None:
            rot_matrix = np.identity(3)
      else:
         for rotation in camera["rotation_list"]:
            rotation_axis = rotation["rotation_axis"]
            rotation_angle = rotation["rotation_angle"]

            if rotation_axis == "x":
               rot_matrix = rotate_x(rotation_angle)
            elif rotation_axis == "y":
               rot_matrix = rotate_y(rotation_angle)
            elif rotation_axis == "z":
               rot_matrix = rotate_z(rotation_angle)
            else:
               raise ValueError("Invalid rotation axis")
            
            final_rot_matrix = np.dot(rot_matrix, final_rot_matrix)
      
      transform_matrix = np.identity(4)
      transform_matrix[:3, :3] = final_rot_matrix  # Rotationsmatrix einfügen
      transform_matrix[:3, 3] = pos  # Translation einfügen
      
      transform_json["frames"].append({
         "file_path": f"{cameras['params']['path']}/{camera['name']}",
         "transform_matrix": transform_matrix.tolist()
      })
      
   return transform_json

def rotate_x(rotation_angle):
   rot_matrix = np.array([
      [1, 0, 0],
      [0, np.cos(np.radians(rotation_angle)), np.sin(np.radians(rotation_angle))], 
      [0, -np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))], 
   ])
   return rot_matrix

def rotate_y(rotation_angle):
   rot_matrix = np.array([
      [np.cos(np.radians(rotation_angle)), 0, -np.sin(np.radians(rotation_angle))],
      [0, 1, 0],
      [np.sin(np.radians(rotation_angle)), 0, np.cos(np.radians(rotation_angle))]
   ])
   return rot_matrix

def rotate_z(rotation_angle):
   rot_matrix = np.array([
      [np.cos(np.radians(rotation_angle)), np.sin(np.radians(rotation_angle)), 0],
      [-np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)), 0],
      [0, 0, 1]
   ])
   return rot_matrix

cameras = {
   "params": {
      "h": 1920,
      "w": 2560,
      "fl_x": 1213.1944165949544,
      "fl_y": 1213.1944165949544,
      "path": "images"
   },
   "camera_list": []
}

image_num = 47
levels = 3
if( image_num % levels != 0):
   raise ValueError("Image number is not divisible by levels")

images_per_level = int(image_num / levels)
position_degree = 360 / images_per_level

height_array = [.048, .474, 1.285]
factorize = 1

for j in range(levels):
   height = height_array[j]
   for i in range(int(images_per_level)):
      if j != 2:
         rotation_list = [
            {
               "rotation_axis": "y",
               "rotation_angle": (i * position_degree) - 90
            },
            {
               "rotation_axis": "x",
               "rotation_angle": 90
            }
         ]
      else:
         rotation_list = [
            {
               "rotation_axis": "y",
               "rotation_angle": (i * position_degree) - 90
            },
            {
               "rotation_axis": "x",
               "rotation_angle": 90
            },
            {
               "rotation_axis": "z",
               "rotation_angle": -135
            }
         ]
      cameras["camera_list"].append({
         "name": f"{((i + 1) + (images_per_level * j)):02d}.jpg",
         "pos": [np.cos(np.radians(i * position_degree))*factorize, np.sin(np.radians(i * position_degree))*factorize, height*factorize],
         "rotation_list": rotation_list
      })
   

matrix = create_transform_json(cameras)

out_dir = "C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/position_test/1m_chair/transforms_own.json"

if os.path.exists(out_dir):
   os.remove(out_dir)
   
with open(out_dir, "w") as f:
   json.dump(matrix, f, indent=4)

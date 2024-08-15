# create transform.json

# rot matrix x-axis
# 1 0 0
# 0 cos sin
# 0 -sin cos

# rot matrix y-axis
# cos 0 -sin
# 0 1 0
# sin 0 cos

# rot matrix z-axis
# cos sin 0
# -sin cos 0
# 0 0 1

# cos(0) = 1
# sin(0) = 0
#cos(45) = 0.866025067811865476
#sin(45) = 0.866025067811865475
# cos(90) = 0
# sin(90) = 1

import json
import numpy as np
import os


def create_transform_json(cameras):  
   transform_json = {
      "w": cameras["params"]["w"],
      "h": cameras["params"]["h"],
      "fl_x": cameras["params"]["fl_x"],
      "fl_y": cameras["params"]["fl_y"],    
      "cx": cameras["params"]["w"] / 2,
      "cy": cameras["params"]["h"] / 2,
      "k1": 0.017336012291600932,
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
      "fl_x": 1811,
      "fl_y": 1811,
      "path": "images"
   },
   "camera_list": []
}

# rotation_y = -90
# rotation_x = -90
# rotation_z = -135

image_num = 32
levels = 2
images_per_level = image_num / levels
position_degree = 360 / images_per_level

height_buttom = 0.0575
height_top = 0.427
factorize = 4

for j in range(levels):
   heigth = height_buttom if j == 0 else height_top
   for i in range(int(images_per_level)):
      cameras["camera_list"].append({
         "name": f"{i + 1:02}.jpg",
         "pos": [np.cos(np.radians(i * position_degree))*factorize, np.sin(np.radians(i * position_degree))*factorize, heigth*factorize],
         "rotation_list": [
            {
               "rotation_axis": "y",
               "rotation_angle": (i * position_degree) - 90
            },
            {
               "rotation_axis": "x",
               "rotation_angle": 90
            }
         ]
      })
   

matrix = create_transform_json(cameras)

out_dir = "C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/position_test/1m_chair/transforms.json"

if os.path.exists(out_dir):
   os.remove(out_dir)
   
with open(out_dir, "w") as f:
   json.dump(matrix, f, indent=4)

# cameras = {
#    "params": {
#       "h": 1920,
#       "w": 2560,
#       "fl_x": 1805.0912628601652,
#       "fl_y": 1805.0912628601652,
#       "path": "images"
#    }, 
#    "camera_list": [
#       {
#          "name": "01.jpg",
#          "pos": [1, 0, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             }
#          ]
#       }, 
#       {
#          "name": "02.jpg",
#          "pos": [0.866025, 0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -45         
#             }
#          ]
#       },
#       {
#          "name": "03.jpg",
#          "pos": [0, 1, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -90         
#             }
#          ]
#       },
#       {
#          "name": "04.jpg",
#          "pos": [-0.866025, 0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -135         
#             }
#          ]
#       },
#       {
#          "name": "05.jpg",
#          "pos": [-1, 0, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -180        
#             }
#          ]
#       },
#       {
#          "name": "06.jpg",
#          "pos": [-0.866025, -0.866025, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -225        
#             }
#          ]
#       },
#       {
#          "name": "07.jpg",
#          "pos": [0, -1, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -270       
#             }
#          ]
#       },
#       {
#          "name": "08.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
#       {
#          "name": "09.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
#       {
#          "name": "10.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
#       {
#          "name": "11.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
#       {
#          "name": "12.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
#       {
#          "name": "13.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
#       {
#          "name": "14.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
#       {
#          "name": "15.jpg",
#          "pos": [0.866025, -0.866025, 0.0575],
#           "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": rotation_x         
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -315      
#             }
#          ]
#       },
    
#    ]
# }

# cameras = {
#    "params": {
#       "h": 1920,
#       "w": 2560,
#       "fl_x": 24043.75700169994,
#       "fl_y": 24043.75700169994,
#       "path": "images"
#    }, 
#    "camera_list": [
#       {
#          "name": "01.jpg",
#          "pos": [1, 0, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": 90         
#             }
#          ]
#       }, 
#       {
#          "name": "02.jpg",
#          "pos": [0.866025, 0.866025, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y          
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -45         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": 90         
#             }
#          ]
#       },
#       {
#          "name": "03.jpg",
#          "pos": [1, 0, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y          
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": rotation_y         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": 90         
#             }
#          ]
#       },
#       {
#          "name": "04.jpg",
#          "pos": [-0.866025, 0.866025, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y          
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -135         
#             },
#             {
#                "rotation_axis": "x",
#                "rotation_angle": 90         
#             }
#          ]
#       },
#       {
#          "name": "02.jpg",
#          "pos": [0, 1, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y          
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -45         
#             }
#          ]
#       },
#       {
#          "name": "02.jpg",
#          "pos": [0, 1, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y          
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -45         
#             }
#          ]
#       },
#       {
#          "name": "02.jpg",
#          "pos": [0, 1, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y          
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -45         
#             }
#          ]
#       },
#       {
#          "name": "02.jpg",
#          "pos": [0, 1, 0.0575],
#          "rotation_list": [
#             {
#                "rotation_axis": "y",
#                "rotation_angle": rotation_y          
#             },
#             {
#                "rotation_axis": "z",
#                "rotation_angle": -45         
#             }
#          ]
#       },
#    ]
# }

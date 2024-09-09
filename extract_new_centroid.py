import json

def calculate_centroid(camera_positions):
   """
   calculate the centroid of camera positions in 3D space.

   camera_positions: List of tuples (x, y, z) representing the camera positions.
   return: Tuple (centroidX, centroidY, centroidZ) as the centroid of the camera positions.
   """
   if not camera_positions:
      return None

   # centroidX == (xa + xb + xc + ... + xn) / n
   
   n = len(camera_positions)
   sum_x = sum_y = sum_z = 0

   for position in camera_positions:
      sum_x += position[0]
      sum_y += position[1]
      sum_z += position[2]

   centroidX = sum_x / n
   centroidY = sum_y / n
   centroidZ = sum_z / n
   return (centroidX, centroidY, centroidZ)
 
def extract_camera_positions(json_file_path):
   """ extract camera positions from the json file.
   json_file_path: Path to the json file containing the camera positions.
   return: List of tuples (x, y, z) representing the camera positions.
   """
   with open(json_file_path, 'r') as file:
      data = json.load(file)

   positions = []
   
   for frame in data['frames']:
      transform_matrix = frame['transform_matrix']
      x = transform_matrix[0][3]
      y = transform_matrix[1][3]
      z = transform_matrix[2][3]
      positions.append((x, y, z))
      
   # for i in range(20, 70):
   #    transform_matrix = data['frames'][i]['transform_matrix']
   #    x = transform_matrix[0][3]
   #    y = transform_matrix[1][3]
   #    z = transform_matrix[2][3]
   #    positions.append((x, y, z))

   return calculate_centroid(positions)
 
json_file_path = 'D:/Masterthesis/omniverse/centroidcamera/transform.json'
positions = extract_camera_positions(json_file_path)
print(positions)
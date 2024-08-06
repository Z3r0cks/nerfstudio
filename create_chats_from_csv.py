import pandas as pd
import os
import matplotlib.pyplot as plt

# CSV-Datei laden
df = pd.read_csv('C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/single_ray_informations.csv')
grouped = df.groupby('ray_id')
save_dir = f'C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/Data/Plots/'

for ray_id, group in grouped:
   group = group.reset_index(drop=True) # reset index
   
   closest_idx = (group['distance'] - 1).abs().idxmin()
   closest_idx = int(closest_idx)

   max_range = min(closest_idx, len(group) - closest_idx - 1)
   # print("len: ",  len(group))
   # print("len - closes_index -1 : ",  len(group) - closest_idx - 1)
   # print("closest_idx: ", closest_idx)
   # print("max_range: ", max_range)
   # print("sum: ", closest_idx + max_range)
   # max_range = len(group) - 2
   step = 1

   plt.figure(figsize=(30, 12))

   plt.plot(group['distance'], group['density'], 'b-', marker='o', label='Density')
   plt.plot(group['distance'][closest_idx], group['density'][closest_idx], 'ro', label='Closest to Distance=1')

   plt.xlabel('Distance')
   plt.ylabel('Density')
   plt.title(f'Ray ID: {ray_id}')
   plt.legend()
   
   if not os.path.exists(f'{save_dir}/{ray_id}'):
      os.makedirs(f'{save_dir}/{ray_id}')
      
   plt.savefig(f'{save_dir}/{ray_id}/ray_id_{ray_id}.png')
   plt.close()
   
   for n in range(step, max_range + step, step):
      start = max(closest_idx - n, 0)
      end = min(closest_idx + n, len(group) - 1)
      plt.figure(figsize=(30, 12))
         
      for i in range(start, end + 1):
            plt.plot(group['distance'][i], group['density'][i], 'bo', label='Density' if i == start else "")
      
      if start <= closest_idx <= end:
            plt.plot(group['distance'][closest_idx], group['density'][closest_idx], 'ro', label='Closest to Distance=1')
      
      plt.plot(group['distance'][start:end + 1], group['density'][start:end + 1], color='blue')

      plt.xlabel('Distance')
      plt.ylabel('Density')
      plt.title(f'Ray ID: {ray_id}')
      plt.legend()

      plt.savefig(f'{save_dir}/{ray_id}/ray_id_{ray_id}_range_{n}.png')
      plt.close()

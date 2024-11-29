import open3d as o3d
import pandas as pd

with open("C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/cato.csv") as file:
    lines = [line.strip().replace("[", "").replace("]", "") for line in file]

data = [list(map(float, line.split())) for line in lines]
df = pd.DataFrame(data, columns=["x", "y", "z"])

df_filtered = df[(df['x'].abs() <= 3) & (df['y'].abs() <= 3) & (df['z'].abs() <= 3)]

points = df_filtered[['x', 'y', 'z']].values

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([point_cloud]) #type: ignore

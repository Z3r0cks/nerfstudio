import math
import os
import asyncio
import omni.kit.commands
import omni.kit.viewport.utility
import omni.usd
import json
import numpy as np
import omni.replicator.core as rep
import omni.ui.scene as scene
from pxr import UsdGeom, Gf, Usd
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from scipy.spatial.transform import Rotation
from omni.isaac.ui.element_wrappers import CollapsableFrame, StateButton, Button, CheckBox
from ..safety.lidar import Lidar

async def capture_image_from_camera(viewport, camera_path, image_path):
    viewport.camera_path = camera_path

    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(.8) 
    capture_viewport_to_file(viewport, image_path)

def create_transform_matrix(pos, rot):
    
    """Create a standard transformation matrix from the given position and rotation."""
    
    rotation = Rotation.from_euler("xyz", rot, degrees=True).as_matrix()
    x, y, z = rot
    
    rotation_matrix = np.array([
        [rotation[0, 0], rotation[0, 1], rotation[0, 2], pos[0]],
        [rotation[1, 0], rotation[1, 1], rotation[1, 2], pos[1]],
        [rotation[2, 0], rotation[2, 1], rotation[2, 2], pos[2]],
        [0, 0, 0, 1]
    ])
 
    return rotation_matrix.tolist()

async def generate_images_from_scene():
    # object_center = Gf.Vec3f(1.5, 1.7, 0)
    # camera_count = 4
    camera_positions = []
    camera_rotations = []
    frames = []

    # camera_heights = [3, 5, 7, 8.8]  # height of the camera
    # rotation_x_values = [70, 60, 47, 15]  # angle of the camera
    # radien = [7, 7, 6, 2]  # distance of the camera from the object center
    camera_heights = [0.3, 3, 5, 7, 8.8, 0, 0.3]  # height of the camera
    rotation_x_values = [98, 70, 60, 47, 15, 110, 90]  # angle of the camera
    radien = [5.3, 7, 7, 7, 2, 0.1, 0.1]  # distance of the camera from the object center
    object_center = [Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, 0)]
    image_count = [30, 30, 30, 30, 30, 25, 25]
            
    for z, rot_x, radius, center, length in zip(camera_heights, rotation_x_values, radien, object_center, image_count):
        for i in range(length):
            angle = 2 * math.pi * i / length
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            pos = Gf.Vec3f(x, y, z)
            direction = center - pos
            direction = direction.GetNormalized()

            # Berechnen Sie die Kamerarotation, so dass die Kamera auf center ausgerichtet ist
            y_rot = math.degrees(math.atan2(direction[1], direction[0])) - 90

            camera_positions.append(pos)
            camera_rotations.append((rot_x, 0, y_rot))
    
    stage = omni.usd.get_context().get_stage()
    viewport = get_active_viewport()

    for i, (pos, rot) in enumerate(zip(camera_positions, camera_rotations)):
        camera_name = f"camera_{i}"
        camera_path = f"/World/{camera_name}"
        if not stage.GetPrimAtPath(camera_path):
            camera_prim = UsdGeom.Camera.Define(stage, camera_path)
            
            # camera position and rotation
            UsdGeom.Xformable(camera_prim).AddTranslateOp().Set(Gf.Vec3d(pos))
            UsdGeom.Xformable(camera_prim).AddRotateXYZOp().Set(Gf.Vec3f(*rot))
            UsdGeom.Camera(camera_prim).GetFocalLengthAttr().Set(23)

            # change camera path
            viewport.camera_path = camera_path
            image_path = f"D:\Masterthesis\omniverse\yellow_grey_diffuse_floor\{camera_name}.png"
            await capture_image_from_camera(viewport, camera_path, image_path)

            transform_matrix = create_transform_matrix(pos, rot)
            frames.append({
                "file_path": image_path,
                "transform_matrix": transform_matrix,
                "colmap_im_id": i
            })

            stage.RemovePrim(camera_prim.GetPath())

    transform_data = {
        "w": 1280,
        "h": 720,
        "fl_x": 1406.2874242545224,
        "fl_y": 1403.5877771472647,
        "cx": 640,
        "cy": 360,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "camera_model": "OPENCV",
        "frames": frames,
        # "applied_transform": [
        #     [
        #         1.0,
        #         0.0,
        #         0.0,
        #         0.0
        #     ],
        #     [
        #         0.0,
        #         0.0,
        #         1.0,
        #         0.0
        #     ],
        #     [
        #         -0.0,
        #         -1.0,
        #         -0.0,
        #         -0.0
        #     ]
        # ],
        # "ply_file_path": "sparse_pc.ply"
    }

    with open("D:\Masterthesis\omniverse\yellow_grey_diffuse_floor/transform.json", "w") as f:
        json.dump(transform_data, f, indent=4)


lidar_run = False

async def start_lidar():
    lidar_prim_path = "/SICK_picoScan150"
    lidar = Lidar(lidar_prim_path)
    data = lidar.get_data()
    distances = data["distance"]
    position = np.array([2.0, -1.0, 0.2])
    rotation = np.array([0.0, 0.0, 0.2])
    pose = np.hstack((position, rotation))

    # save distances to file
    file_path_distance = "D:/Masterthesis/omniverse/lidar_data/distance.csv"
    file_path_pose = "D:/Masterthesis/omniverse/lidar_data/pose_data.csv"
    os.makedirs(os.path.dirname(file_path_distance), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_pose), exist_ok=True)
    # np.savetxt(file_path_distance, distances, delimiter=",")
    np.savetxt(file_path_pose, pose, delimiter=",", header="pos_x,pos_y,pos_z,rot_x,rot_y,rot_z")
    np.savetxt(file_path_distance, distances, delimiter=",", )

    global lidar_run
    render_product = rep.create.render_product(lidar_prim_path, [1, 1])
    writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
    writer.attach(render_product)
    try:
        rep.orchestrator.preview()
        lidar_run = True
        print(f"lidar is running")
    except Exception as e:
        print(f"Error starting lidar: {e}")

async def stop_lidar():
    print("placeholder")
    # global lidar_run
    # if lidar_run:
    #     try:
    #         print(f"Stopping Lidar")
    #         rep.orchestrator.stop()
    #         lidar_run = False
    #     except Exception as e:
    #         print(f"Error stopping lidar: {e}")
    # else:
    #     print("Lidar is not running")

# {'azimuth': array([], dtype=float64), 'beamId': array([], dtype=float64), 
#  'data': 
#     array([[-14.613132 , -13.157725 ,   0.       ],
#        [-14.547728 , -13.144879 ,   0.       ],
#        [-14.513968 , -13.16046  ,   0.       ],
#        ...,
#        [ -7.836207 ,   7.1054373,   0.       ],
#        [ -7.8675823,   7.1089053,   0.       ],
#        [ -7.890243 ,   7.104408 ,   0.       ]], dtype=float32), 
#     'distance': array([19.66391 , 19.606739, 19.592167, ..., 10.577967, 10.603555,
#        10.61737 ], dtype=float32), 
#     'elevation': array([], dtype=float64), 
#     'emitterId': array([], dtype=float64), 
#     'index': array([], dtype=float64), 
#     'intensity': array([0.02388821, 0.03522532, 0.05174344, ..., 0.03006477, 0.04155155,
#        0.05382654], dtype=float32), 
#     'materialId': array([], dtype=float64), 
#     'normal': array([], dtype=float64), 'objectId': array([], dtype=float64), 
#     'timestamp': array([], dtype=float64), 
#     'velocity': array([], dtype=float64), 
#     'info': {
#         'numChannels': 2761, 
#         'numEchos': 1, 
#         'numReturnsPerScan': 2761, 
#         'renderProductPath': '/Render/RenderProduct_Replicator', 
#         'ticksPerScan': 1, 
#         'transform': array([ 1. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  1. ,
#         0. ,  2. , -1. ,  0.2,  1. ])}}
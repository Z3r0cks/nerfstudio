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


dir = "D:/Masterthesis/omniverse/cube_interference"
sleep_time = 1.5
width = 1280
height = 720
focal_length = 23

async def capture_image_from_camera(viewport, camera_path, image_path):
    viewport.camera_path = camera_path

    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(sleep_time) 
    capture_viewport_to_file(viewport, image_path)

def create_transform_matrix(pos, rot):
    """Create a standard transformation matrix from the given position and rotation."""
    
    rotation = Rotation.from_euler("xyz", rot, degrees=True).as_matrix()
    rotation_matrix = np.array([
        [rotation[0, 0], rotation[0, 1], rotation[0, 2], pos[0]],
        [rotation[1, 0], rotation[1, 1], rotation[1, 2], pos[1]],
        [rotation[2, 0], rotation[2, 1], rotation[2, 2], pos[2]],
        [0, 0, 0, 1]
    ])
 
    return rotation_matrix.tolist()

def get_focal_length_x_y(camera_path):
    stage = omni.usd.get_context().get_stage()
    camera = stage.GetPrimAtPath(camera_path)
    focal_length = camera.GetAttribute("focalLength").Get()
    horiz_aperture = camera.GetAttribute("horizontalAperture").Get()
    vert_aperture = height/width * horiz_aperture
    return ((height * focal_length / vert_aperture),(width * focal_length / horiz_aperture))

async def generate_images_from_scene():
    camera_positions = []
    camera_rotations = []
    frames = []
    # object_center = [Gf.Vec3f(1, 0, 1), Gf.Vec3f(5, 2, 3)] cendroid 5
    # angle_of_camera = [83, 59]
    # radien = [7, 7]  # distance of the camera from the object center
    # object_center = [Gf.Vec3f(0, 0, 2), Gf.Vec3f(0, 0, 4)]
    # image_count = [2, 2]

    # angle_of_camera = [40, 60, 70, 87, 97]
    # radien = [6, 7, 7, 7, 5]  # distance of the camera from the object center
    # object_center = [Gf.Vec3f(1, 1, 8), Gf.Vec3f(1, 1, 5), Gf.Vec3f(1, 1, 3), Gf.Vec3f(1, 1, 1.5), Gf.Vec3f(1, 1, 0.2)]
    # image_count = [35, 35, 35, 35, 35]
    
    #cube 3
    # angle_of_camera = [52, 58, 66, 75, 50, 54, 66, 50, 54, 66]
    # radien = [16, 11, 10, 10, 15, 13, 12, 15, 13, 12]  # distance of the camera from the object center
    # object_center = [Gf.Vec3f(0, 0, 13), Gf.Vec3f(0, 0, 8), Gf.Vec3f(0, 0, 5), Gf.Vec3f(0, 0, 3), Gf.Vec3f(8, -6, 12), Gf.Vec3f(8, -6, 9), Gf.Vec3f(8, -6, 5), Gf.Vec3f(-4, 4, 12), Gf.Vec3f(-4, 4, 9), Gf.Vec3f(-4, 4, 5)]
    # image_count = [72, 72, 72, 72, 72, 72, 72, 72, 72, 72]
    
    #cube single
    angle_of_camera = [30, 52, 60, 68, 76]
    radien = [9, 15, 15, 15, 13, 13]  # distance of the camera from the object center
    object_center = [Gf.Vec3f(1, -2, 16), Gf.Vec3f(1, -2, 13), Gf.Vec3f(1, -2, 10), Gf.Vec3f(1, -2, 7), Gf.Vec3f(1, -2, 4)]
    image_count = [72, 72, 72, 72, 72]
            
    for rot_x, radius, center, length in zip(angle_of_camera, radien, object_center, image_count):
        for i in range(length):
            angle = 2 * math.pi * i / length
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2]
            pos = Gf.Vec3f(x, y, z)
            direction = center - pos
            direction = direction.GetNormalized()

            y_rot = math.degrees(math.atan2(direction[1], direction[0])) - 90

            camera_positions.append(pos)
            camera_rotations.append((rot_x, 0, y_rot))
    
    stage = omni.usd.get_context().get_stage()
    viewport = get_active_viewport()
    fly = 0
    flx = 0
    
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
            image_path = f"{dir}\{camera_name}.png"
            await capture_image_from_camera(viewport, camera_path, image_path)

            transform_matrix = create_transform_matrix(pos, rot)
            frames.append({
                "file_path": image_path,
                "transform_matrix": transform_matrix,
                "colmap_im_id": i
            })
        
            if camera_path == "/World/camera_0":
                fy, fx = get_focal_length_x_y(camera_path)
                flx, fly = fx, fy
            
            stage.RemovePrim(camera_prim.GetPath())

    transform_data = {
        "w": width,
        "h": height,
        "fl_x": fly,
        "fl_y": flx,
        "cx": width / 2,
        "cy": height / 2,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "camera_model": "OPENCV",
        "frames": frames,
    }

    with open(f"{dir}/transform.json", "w") as f:
        json.dump(transform_data, f, indent=4)


lidar_run = False
    
async def start_lidar():
    lidar_prim_path = "/SICK_picoScan150"
    lidar = Lidar(lidar_prim_path)
    data = lidar.get_data()
    distances = data["distance"]
    position = np.array([4.5, 0.5, 1])
    rotation = np.array([0.0, 90, 0])
    pose = np.hstack((position, rotation))

    # save distances to file
    file_path_distance = "D:/Masterthesis/omniverse/lidar_data/cube_interference.csv"
    file_path_pose = "D:/Masterthesis/omniverse/lidar_data/cube_interference.csv"
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
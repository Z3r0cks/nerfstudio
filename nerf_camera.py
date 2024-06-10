import math
import asyncio
import omni.kit.commands
import omni.kit.viewport.utility
import omni.usd
import json
import numpy as np
from pxr import UsdGeom, Gf
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from scipy.spatial.transform import Rotation

async def capture_image_from_camera(viewport, camera_path, image_path):
    viewport.camera_path = camera_path

    await omni.kit.app.get_app().next_update_async()
    await asyncio.sleep(.5) 
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

async def generate_images_from_scene():
    # object_center = Gf.Vec3f(1.5, 1.7, 0)
    # camera_count = 4
    camera_positions = []
    camera_rotations = []
    frames = []

    # camera_heights = [3, 5, 7, 8.8]  # height of the camera
    # rotation_x_values = [70, 60, 47, 15]  # angle of the camera
    # radien = [7, 7, 6, 2]  # distance of the camera from the object center
    camera_heights = [0.3, 3, 5, 7, 8.8, 0.1, 0.3]  # height of the camera
    rotation_x_values = [98, 70, 60, 47, 15, 110, 90]  # angle of the camera
    radien = [5.3, 7, 7, 7, 2, 0.1, 0.1]  # distance of the camera from the object center
    object_center = [Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(1.5, 1.7, 0), Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, 0)]
    image_count = [30, 30, 30, 30, 30, 30, 30]
            
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
            image_path = f"D:/Masterthesis/omniverse/images/{camera_name}.png"
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
        "cx": 639.0542252314708,
        "cy": 356.3752260152546,
        "k1": -0.002075426491080844,
        "k2": 0.002143635846669103,
        "p1": 0.0003473158440651779,
        "p2": -0.00022910263316722898,
        "camera_model": "OPENCV",
        "frames": frames,
        "applied_transform": [
            [
                1.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0
            ],
            [
                -0.0,
                -1.0,
                -0.0,
                -0.0
            ]
        ],
        # "ply_file_path": "sparse_pc.ply"
    }

    with open("D:/Masterthesis/omniverse/transform.json", "w") as f:
        json.dump(transform_data, f, indent=4)



# import math
# import asyncio
# import omni.kit.commands
# import omni.kit.viewport.utility
# import omni.usd
# from pxr import UsdGeom, Gf
# from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

# async def capture_image_from_camera(viewport, camera_path, image_path):
#     viewport.camera_path = camera_path

#     await omni.kit.app.get_app().next_update_async()
#     await asyncio.sleep(.3) 
#     capture_viewport_to_file(viewport, image_path)

# async def generate_images_from_scene():
#     object_center = Gf.Vec3f(1.5,1.7,0)
#     # radien = 10
#     camera_count = 50
#     camera_positions = []
#     camera_rotations = []

#     camera_heights = [7]  # Z-Werte für verschiedene Kamerahöhen
#     rotation_x_values = [38]  # Unterschiedliche X-Rotationswerte für jede Höhe
#     radien = [5]  # Unterschiedliche X-Rotationswerte für jede Höhe

#     # for z, rot_x, radius in zip(camera_heights, rotation_x_values, radien):
#     #     for i in range(camera_count):
#     #         angle = 2 * math.pi * i / camera_count
#     #         x = radius * math.cos(angle)
#     #         y = radius * math.sin(angle)
#     #         pos = Gf.Vec3f(x, y, z)
#     #         direction = object_center - pos
#     #         direction = direction.GetNormalized()
#     #         y_rot = math.degrees(math.atan2(direction[1], direction[0])) - 90

#     #         camera_positions.append(pos)
#     #         camera_rotations.append((rot_x, 0, y_rot))
            
#     for z, rot_x, radius in zip(camera_heights, rotation_x_values, radien):
#         for i in range(camera_count):
#             angle = 2 * math.pi * i / camera_count
#             x = object_center[0] + radius * math.cos(angle)
#             y = object_center[1] + radius * math.sin(angle)
#             pos = Gf.Vec3f(x, y, z)

#             direction = object_center - pos
#             direction = direction.GetNormalized()

#             # Berechnen Sie die Kamerarotation, so dass die Kamera auf das object_center ausgerichtet ist
#             y_rot = math.degrees(math.atan2(direction[1], direction[0])) - 90

#             camera_positions.append(pos)
#             camera_rotations.append((rot_x, 0, y_rot))
    
#     stage = omni.usd.get_context().get_stage()
#     viewport = get_active_viewport()

#     for i, (pos, rot) in enumerate(zip(camera_positions, camera_rotations)):
#         camera_name = f"Camera_{i}"
#         camera_path = f"/World/{camera_name}"
#         if not stage.GetPrimAtPath(camera_path):
#             camera_prim = UsdGeom.Camera.Define(stage, camera_path)
            
#             # camera position and rotation
#             UsdGeom.Xformable(camera_prim).AddTranslateOp().Set(pos)
#             UsdGeom.Xformable(camera_prim).AddRotateXYZOp().Set(Gf.Vec3f(*rot))
#             UsdGeom.Camera(camera_prim).GetFocalLengthAttr().Set(23)

#             # change camera path
#             viewport.camera_path = camera_path
#             image_path = f"D:/Omniverse/images/{camera_name}.png"
#             await capture_image_from_camera(viewport, camera_path, image_path)

#             stage.RemovePrim(camera_prim.GetPath())

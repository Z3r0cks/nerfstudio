import torch
import math

# def _compute_fov_rad(width_height, fx_fy):
#    fov = 2 * torch.atan(width_height / (2 * fx_fy))
#    fov_degrees = fov * (180.0 / math.pi)
#    fov_degrees = int(round(fov_degrees.item()))
#    if fov_degrees < 0:
#       fov_degrees = 360 + fov_degrees
#    return fov_degrees


# print(_compute_fov_rad(torch.tensor(270), -153.2645))
# print(_compute_fov_rad(torch.tensor(1), -0.555))

height = 1
FOV = 276


if FOV > 180:
   FOV = 360 - FOV

fy_value = height / (2 * math.tan(math.radians(FOV / 2)))

print(fy_value)
# Procces of the render pipeline to def get_density() in nerfacto_field.py:208 </h1>

## Pipeline Stack Call:

| Function                            | File                                                                                           | Line |
| ----------------------------------- | ---------------------------------------------------------------------------------------------- | ---- |
| `_bootstrap_inner`                  | file:///C:/Users/Patri/miniconda3/envs/nerfstudio/lib/threading.py                             | 932  |
| `_bootstrap`                        | file:///C:/Users/Patri/miniconda3/envs/nerfstudio/lib/threading.py                             | 890  |
| `run`                               | file:///D:/Masterthesis/nerfstudio/nerfstudio/viewer/render_state_machine.py                   | 228  |
| `render_img`                        | file:///D:/Masterthesis/nerfstudio/nerfstudio/viewer/render_state_machine.py                   | 173  |
| `decorate_context`                  | file:///C:/Users/Patri/miniconda3/envs/nerfstudio/lib/site-packages/torch/utils/_contextlib.py | 115  |
| `get_outputs_for_camera`            | file:///D:/Masterthesis/nerfstudio/nerfstudio/models/base_model.py                             | 173  |
| `decorate_context`                  | file:///C:/Users/Patri/miniconda3/envs/nerfstudio/lib/site-packages/torch/utils/_contextlib.py | 115  |
| `get_outputs_for_camera_ray_bundle` | file:///D:/Masterthesis/nerfstudio/nerfstudio/models/base_model.py                             | 195  |
| `forward`                           | file:///D:/Masterthesis/nerfstudio/nerfstudio/models/base_model.py                             | 143  |
| `get_outputs`                       | file:///D:/Masterthesis/nerfstudio/nerfstudio/models/nerfacto.py                               | 311  |
| `forward`                           | file:///D:/Masterthesis/nerfstudio/nerfstudio/fields/base_field.py                             | 128  |
| `get_density`                       | file:///D:/Masterthesis/nerfstudio/nerfstudio/fields/nerfacto_field.py                         | 211  |
 

## render_state_machine.run()
<pre>Main loop for the render thread</pre>

This methods reacts on different states and executes and handles the data between the components.
While self.running is true, the method will run in a loop and execute the different states.
It will checked if the viewer is ready, if not, the loop will pause for 0.1 seconds.
            
***self.render.trigger:***
self.render.trigger waits for 0.2 seconds for a render signal. If there is no signal, it returns a static action back to the viewer.

***process the next action:***
next action is called with self.next_action. If no action (action is None), the loop will continued without any action and self.render.trigger will be reseted.

***Check the current state and actions:
If the state is high and the action is static, no action is needed and the loop will continue.

***State changing:***
the state if the renderer will be updated with self.transition, which set the state depending on the resived action.

***Img rendering:***
the method ***self._render_img(action.camera_state)*** renders the image with the current camera state.

***Send the image to the viewer:
the rendered image will be send to the viewer with self._send_output_to_viewer(outputs, static_render=(action.action in ["static", "step"])). The parameter static_render specifies if its a static or step rendering.

***Summary:***
The method runs in a loop and checks the current state and action. Depending on the state and action, the method will execute the next action and send the rendered image to the viewer.

***outputs***
return from render_img() is outputs, which has following properties:

| Property         | Description                            |                           |
| ---------------- | -------------------------------------- | ------------------------- |
| `rgb`            | rendered image  (width x Height x RGB) | torch.Size([275, 512, 3]) |
| `accumulation`   | accumulation of the rendered image     | torch.Size([275, 512, 1]) |
| `expected_depth` | expected depth of the rendered image   | torch.Size([275, 512, 1]) |
| `prop_depth_0`   | prop depth 0 of the rendered image     | torch.Size([275, 512, 1]) |
| `prop_depth_1`   | prop depth 1 of the rendered image     | torch.Size([275, 512, 1]) |
| `gl_z_buf_depth` | gl z buf depth of the rendered image   | torch.Size([275, 512, 1]) |
|                  |

## render_state_machine.render_img()
<pre>Takes the current camera, generates rays, and renders the image
Args:
   camera_state: the current camera state</pre>

```python
CameraState(
   fov=1.3089969389957472, 
   aspect=1.860870829395944, 
   c2w=tensor(
      [[ 6.8639e-01, -3.0501e-01,  6.6018e-01,  2.1188e-01],
       [ 7.2724e-01,  2.8787e-01, -6.2310e-01, -2.5536e-01],
       [-2.2204e-16,  9.0780e-01,  4.1940e-01, -1.9855e-01]], dtype=torch.float64), 
   camera_type=<CameraType.PERSPECTIVE: 1>, time=0.0)
```

| Property      | Description                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `fov`         | radiant. 1.309 approx 75 degree                                                                                                |
| `aspect`      | aspect ratio of the image. 1.86 approx 16:9                                                                                    |
| `c2w`         | camera to world matrix. 4x3 matrix. The first 3x3 matrix is the rotation matrix and the last column is the translation vector. |
| `camera_type` | CameraType.PERSPECTIVE                                                                                                         |
| `time`        | time of the camera state. 0.0                                                                                                  |
|               |
```python
image_height 275
image_width 512

 Cameras(
   camera_to_worlds=tensor([[
      [ 6.8639e-01, -3.0501e-01,  6.6018e-01,  2.1188e-01],
      [ 7.2724e-01,  2.8787e-01, -6.2310e-01, -2.5536e-01],
      [-2.2204e-16,  9.0780e-01,  4.1940e-01, -1.9855e-01]]]), 
   fx=tensor([[179.1935]]), 
   fy=tensor([[179.1935]]), 
   cx=tensor([[256.]]), 
   cy=tensor([[137.5000]]), 
   width=tensor([[512]]), 
   height=tensor([[275]]), 
   distortion_params=None, 
   camera_type=tensor([[1]]), 
   times=tensor([[0.]]), 
   metadata=None)
```

| Property            | Description                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `camera_to_worlds`  | camera to world matrix. 4x3 matrix. The first 3x3 matrix is the rotation matrix and the last column is the translation vector. |
| `fx`                | focal length in x direction. 179.1935                                                                                          |
| `fy`                | focal length in y direction. 179.1935                                                                                          |
| `cx`                | principal point in x direction. 256.0                                                                                          |
| `cy`                | principal point in y direction. 137.5                                                                                          |
| `width`             | image width. 512                                                                                                               |
| `height`            | image height. 275                                                                                                              |
| `distortion_params` | None                                                                                                                           |
| `camera_type`       | CameraType.PERSPECTIVE                                                                                                         |
| `times`             | time of the camera state. 0.0                                                                                                  |
| `metadata`          | None                                                                                                                           |


Positions will be genrated in nerfacto.py with: ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns). Its the proposal sampler of the nerfacto model.
Erstes Diagramm ist ein allgemeines Digramm um den Prozess in Nerfstudio zu verstehen und wie an Density werte zugegriffen werden kann. Die Inhalte sollten auf englisch sein:

:start: (with input: ns-view-lidar in conda)

(unnecessary threading processes which can be skipped)

- in render_state_machine.py -> user create the GUI with a button
- after adjusting the the frustum in front end, the user can start the process to obtain the density values with a button by calling the method get_outputs_for_camera(camera, width=self.width, height=self.height) while camera:
camera = Cameras(
            camera_to_worlds=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=torch.tensor([[width]]),
            height=torch.tensor([[height]]),
            distortion_params=None,
            camera_type=10,
            times=torch.tensor([[0.]], device='cuda:0')
        )
- base_model.py will called, which icludes the main class "Model" where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.
- get_outputs_for_camera use the method camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box) als input to genrate a raybundle for the method: get_outputs_for_camera_ray_bundle(rayBundle) which takes in camera parameters and computes the output of the model
- in this class the method: get_outputs_for_camera(), which takes in a camera, generates the raybundle, and computes the output of the model. Assumes a ray-based model.
- following method in the same class will called: forward(), run forward starting with a ray bundle which use ray_samples as parameter.
- forward() calles nerfacto.py with the class NerfactoModel which is a subclass from Model. In this class the method get_outputs(raybundle) will be called (field_outputs, densities_locations = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)).
- which called the method forward() (yes, we have the same method name but not the same method) in base_field.py which includes the class Field: Base class for fields.
- the method forward() calles the get_outputs() method which then called the get_density() method in nerfacto_field() which includes the NerfactoField class. This method use raysamples to computes and returns the densities.

get_outputs:

setting selector and positions, then:
   h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
   density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
   self._density_before_activation = density_before_activation

and retuns the values: return density, base_mlp_out, ray_samples.frustums.get_positions()


flowchart LR
    %% Abschnitt 1: Fluss von links nach rechts
    subgraph left_to_right
        direction LR
        Start["Start with input: **ns-view-lidar** in Conda"] --> 
        GUI["User creates GUI with a button (in **render_state_machine.py**)"] --> 
        Adjust["User adjusts the frustum in the front end"] --> 
        PressButton["User presses the button to obtain density values"] --> 
        GetOutputs["Call **get_outputs_for_camera(camera, width, height)**"] --> 
        CreateCamera["Create **Camera** object with parameters"]
    end

    %% Verbindung zum nächsten Abschnitt
    CreateCamera --> BaseModel

    %% Abschnitt 2: Fluss von oben nach unten
    subgraph TB
        direction TB
        BaseModel["Call **base_model.py** (Model class is instantiated)"] --> 
        GetOutputsCamera["Call **get_outputs_for_camera()** method"] --> 
        GenerateRays["Generate ray bundle using **camera.generate_rays(...)**"] --> 
        GetOutputsRayBundle["Call **get_outputs_for_camera_ray_bundle(rayBundle)**"] --> 
        ForwardBase["Invoke **forward()** method in **base_model.py**"] --> 
        NerfactoModel["Call **NerfactoModel** class in **nerfacto.py**"] --> 
        GetOutputsNerfacto["Call **get_outputs(raybundle)** in **NerfactoModel**"] --> 
        FieldForward["Call **field.forward(ray_samples, compute_normals)**"] --> 
        ForwardField["Invoke **forward()** method in **Field** class in **base_field.py**"] --> 
        GetOutputsField["Call **get_outputs()** method"] --> 
        GetDensity["Call **get_density()** in **nerfacto_field.py** (**NerfactoField** class)"] --> 
        ComputeDensities["Compute densities using **ray_samples**"] --> 
        ReturnValues["Return **density**, **base_mlp_out**, and **positions**"] --> 
        End["End"]
    end


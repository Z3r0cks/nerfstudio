# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Model implementation which takes in RayBundles or Cameras
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import NearFarCollider

# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Model)
    """target class to instantiate"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    prompt: Optional[str] = None
    """A prompt to be used in text to NeRF models"""


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.render_aabb: Optional[SceneBox] = None  # the box that we want to render - should be a subset of scene_box
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.collider = None

        self.populate_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses

        if self.config.enable_collider:
            assert self.config.collider_params is not None
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def forward(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle)
    
    def test_origin(self):
        origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # Benutzerdefinierter Ursprungspunkt
        directions = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)  # Beispielhafte Richtungen

        outputs = self.get_outputs_for_custom_rays(origin, directions)
        print("outputs: ", outputs)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    def get_outputs_for_custom_rays(self, origin: torch.Tensor, directions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Berechnet die Ausgaben des Modells für benutzerdefinierte Rays.

        Args:
            origin: Ursprungspunkt der Rays.
            directions: Richtungen der Rays.
        """
        num_rays = directions.shape[0]
        ray_bundle = self.generate_custom_rays(origin, directions, num_rays)
        input_device = directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        outputs_lists = defaultdict(list)
        
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle_chunk = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle_chunk = ray_bundle_chunk.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle_chunk)
            density_locations = outputs["density_locations"]
            density = outputs["density"]
            
            for output_name, output in outputs.items():
                if not isinstance(output, torch.Tensor):
                    continue
                outputs_lists[output_name].append(output.to(input_device))
        
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(-1, outputs_list[0].shape[-1])
        
        outputs["density_locations"] = density_locations
        outputs["density"] = density
        
        return outputs
    
    def calculate_pixel_area(self, directions: torch.Tensor) -> torch.Tensor:
        # Placeholder: Pixelbereich-Berechnung basierend auf den Richtungen (hier einfach auf 1 gesetzt)
        return torch.ones(directions.shape[:-1] + (1,), device=directions.device)
    
    def generate_custom_rays(self, origin: torch.Tensor, directions: torch.Tensor, num_rays: int) -> RayBundle:
        """
        Generiert ein benutzerdefiniertes RayBundle basierend auf einem Ursprungspunkt und den Richtungen.

        Args:
            origin: Tensor der Form (3,), der den Ursprungspunkt der Rays repräsentiert.
            directions: Tensor der Form (num_rays, 3), der die Richtungen der Rays repräsentiert.
            num_rays: Anzahl der zu generierenden Rays.

        Returns:
            RayBundle: Ein RayBundle mit den generierten Rays.
        """
        # Normalisiere die Richtungen
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # Berechne die Pixelbereiche (hier als 1 angenommen)
        pixel_area = self.calculate_pixel_area(directions)
        
        # Erstelle das RayBundle
        raybundle = RayBundle(
            origins=origin.expand(num_rays, -1),  # Expandiert den Ursprung zu den Richtungen
            directions=directions,
            pixel_area=pixel_area
        )
    
        return raybundle
    
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        
        test = self.get_outputs_for_camera_ray_bundle(
            camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
        )
        
        from nerfstudio.utils.debugging import Debugging as db 
        
        db.log("weights", test["weights"].shape)
        
        return self.get_outputs_for_camera_ray_bundle(
            camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
        )
        
    

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            density_locations = outputs["density_locations"]
            density = outputs["density"]
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        outputs["density_locations"] = density_locations
        outputs["density"] = density
        return outputs

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
        """Returns the RGBA image from the outputs of the model.

        Args:
            outputs: Outputs of the model.

        Returns:
            RGBA image.
        """
        accumulation_name = output_name.replace("rgb", "accumulation")
        if (
            not hasattr(self, "renderer_rgb")
            or not hasattr(self.renderer_rgb, "background_color")
            or accumulation_name not in outputs
        ):
            raise NotImplementedError(f"get_rgba_image is not implemented for model {self.__class__.__name__}")
        rgb = outputs[output_name]
        if self.renderer_rgb.background_color == "random":  # type: ignore
            acc = outputs[accumulation_name]
            if acc.dim() < rgb.dim():
                acc = acc.unsqueeze(-1)
            return torch.cat((rgb / acc.clamp(min=1e-10), acc), dim=-1)
        return torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)

    @abstractmethod
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image. 
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """

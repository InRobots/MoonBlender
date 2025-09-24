import os
import json
import argparse
import warnings


import cv2
import hydra
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal


import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as tf
from omegaconf import DictConfig, OmegaConf
from einops import rearrange, repeat
from torch import Tensor


from io import BytesIO
from colorama import Fore
from jaxtyping import Float, UInt8
from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.visualization.camera_trajectory.interpolation import interpolate_extrinsics
    from src.dataset.shims.crop_shim import apply_crop_shim

# parse colmap data
from utils.colmap_utils import read_model, qvec2rotmat
from utils.interp import interpolate_extrinsics

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

def find_camera_groups_by_distance(images_info, min_baseline=1e-6, max_baseline=None):
    camera_ids = list(images_info.keys())
    positions = np.array([images_info[cam_id]['extrinsic'][:3, 3] for cam_id in camera_ids])
    
    distances = []
    for i in range(len(camera_ids)):
        for j in range(i + 1, len(camera_ids)):
            dist = np.linalg.norm(positions[i] - positions[j])
            
            # Apply baseline distance filtering
            if dist < min_baseline:
                print(f"Skipping camera pair {camera_ids[i]}-{camera_ids[j]}: baseline distance too small ({dist:.6f} < {min_baseline})")
                continue
                
            if max_baseline is not None and dist > max_baseline:
                print(f"Skipping camera pair {camera_ids[i]}-{camera_ids[j]}: baseline distance too large ({dist:.6f} > {max_baseline})")
                continue
                
            distances.append((dist, camera_ids[i], camera_ids[j]))

    if not distances:
        print(f"No camera pairs found satisfying baseline constraints (min_baseline={min_baseline}, max_baseline={max_baseline})")
        return []
    
    distances.sort(key=lambda x: x[0])
    
    dist, cam1_id, cam2_id = distances[0]
    print(f"Selected camera pair {cam1_id}-{cam2_id}, baseline distance: {dist:.6f}")
    
    return [(cam1_id, cam2_id)]

class PairDataParser:
    
    def __init__(self, cfg):
        self.cfg = cfg

        self.near = cfg.near
        self.far = cfg.far
        assert self.near is not None and self.far is not None

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]] | list[Float[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            if image.dtype in [torch.float32, torch.float64, torch.float16]:
                torch_images.append(image)
            else:
                image = Image.open(BytesIO(image.numpy().tobytes()))
                torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def parse_from_colmap(self, root_path: str):

        images_info = {}
        cameras, images, _ = read_model(os.path.join(root_path, "sparse"), ".bin")

        for img_id in images.keys():
            camera_id = images[img_id].camera_id
            camera_params = cameras[camera_id].params
            fx = cameras[camera_id].params[0]
            fy = cameras[camera_id].params[1]
            cx = cameras[camera_id].params[2]
            cy = cameras[camera_id].params[3]
            camera_h = cameras[camera_id].height
            camera_w = cameras[camera_id].width

            rr = qvec2rotmat(images[img_id].qvec); 
            tt = images[img_id].tvec
            w2c_mat = np.eye(4)
            w2c_mat[:3, :3] = rr
            w2c_mat[:3, 3] = tt
            
            images_info[img_id] = {
                'name': images[img_id].name,
                "poses": np.hstack([[fx / camera_w, fy / camera_h, cx / camera_w, cy / camera_h, 0, 0], w2c_mat[:3, :4].flatten()]),
                "intrinsics": np.array([fx / camera_w, fy / camera_h, cx / camera_w, cy / camera_h, 0, 0]),
                'extrinsic': np.linalg.inv(w2c_mat),
            }
        
        camera_groups = find_camera_groups_by_distance(images_info, self.cfg.baseline_epsilon)
        if not camera_groups:
            print("No camera groups found satisfying baseline constraints")
            return
        
        print(f"Found camera groups: {camera_groups}")
        
        first_group = camera_groups[0]
        cam1_id, cam2_id = first_group
        cam1_info = images_info[cam1_id]
        cam2_info = images_info[cam2_id]

        poses = torch.cat(
            [
                torch.from_numpy(cam1_info['poses']).unsqueeze(0), 
                torch.from_numpy(cam2_info['poses']).unsqueeze(0)
            ], dim=0)
        extrinsics, intrinsics = self.convert_poses(poses)

        num_interpolations = 20
        t = torch.linspace(0, 1, num_interpolations + 2)[1:-1]
        interpolated = interpolate_extrinsics(extrinsics[0:1], extrinsics[1:2], t)
        target_extrinsics = interpolated.squeeze(0)
        target_intrinsics = repeat(intrinsics[0], "i j -> t i j", t=num_interpolations)

        image1 = cv2.imread(os.path.join(root_path, "images", cam1_info['name']))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = torch.from_numpy(image1 / 255.)
        image1 = rearrange(image1, "h w c -> c h w")

        image2 = cv2.imread(os.path.join(root_path, "images", cam2_info['name']))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = torch.from_numpy(image2 / 255.)
        image2 = rearrange(image2, "h w c -> c h w")

        context_images = self.convert_images([image1, image2])
        target_images = repeat(torch.zeros_like(context_images[0]), "c h w -> b c h w", b=num_interpolations)
        
        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics
        context_intrinsics = intrinsics
        if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
            a, b = context_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_epsilon:
                print(
                    f"Skipped {root_path} because of insufficient baseline "
                    f"{scale:.6f}"
                )
                raise ValueError("Insufficient baseline")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0

        example = {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "image": context_images,
                "near": self.get_bound("near", 2) / nf_scale,
                "far": self.get_bound("far", 2) / nf_scale,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": self.get_bound("near", num_interpolations) / nf_scale,
                "far": self.get_bound("far", num_interpolations) / nf_scale,
            },
            "scene": root_path,
        }

        return apply_crop_shim(example, tuple(self.cfg.image_shape))

@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": StepTracker(),
    }

    model_wrapper = ModelWrapper.load_from_checkpoint(checkpoint_path, **model_kwargs, strict=True)
    model_wrapper.eval()
    model_wrapper = model_wrapper.to("cuda")
    print(cyan(f"Loaded weigths from {checkpoint_path}."))

    data_parser = PairDataParser(cfg.dataset)
    batch = data_parser.parse_from_colmap("path of colmap")

    if batch is None:
        return

    def add_batch_dimension(data):
        if isinstance(data, torch.Tensor):
            return data.unsqueeze(0).float().to("cuda")
        elif isinstance(data, dict):
            return {k: add_batch_dimension(v) for k, v in data.items()}
        elif isinstance(data, list):
            return data
        else:
            return (data, )

    with torch.no_grad():
        for key, value in batch.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"    {subkey}: {subvalue.shape}")
                    else:
                        print(f"    {subkey}: {type(subvalue)}")
            else:
                print(f"  {key}: {type(value)}")
        
        batch = add_batch_dimension(batch)
        
        for key, value in batch.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"    {subkey}: {subvalue.shape}")
                    else:
                        print(f"    {subkey}: {type(subvalue)}")
            else:
                print(f"  {key}: {type(value)}")
        
        model_wrapper.test_step_inference(batch)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    main()

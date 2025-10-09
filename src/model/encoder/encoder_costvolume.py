from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import numpy as np
from einops import rearrange, repeat
from jaxtyping import Float

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from .costvolume.ldm_unet.unet import UNetModel
from ...global_cfg import get_cfg

from dust3r.inference import make_batch_symmetric
from dust3r.model import AsymmetricCroCo3DStereoGaussian, inf  # noqa: F401, needed when loading the model

from models.dpt_block import FeatureFusionBlock_custom_own # type: ignore # noqa: F401, needed when loading the model

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        self.feature_fusion1 = FeatureFusionBlock_custom_own(256, nn.ReLU(False), deconv=False, bn=False, expand=True, align_corners=True)
        self.feature_fusion2 = FeatureFusionBlock_custom_own(256, nn.ReLU(False), deconv=False, bn=False, expand=True, align_corners=True)
        input_channels = 128
        channels = 128
        num_depth_candidates = 128
        modules = [
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=tuple(cfg.costvolume_unet_attn_res),
                channel_mult=tuple(cfg.costvolume_unet_channel_mult),
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=get_cfg().dataset.view_sampler.num_context_views,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
        ]
        self.corr_refine_net = nn.Sequential(*modules)
        # self.regressor_residual = nn.Conv2d(
        #     input_channels, num_depth_candidates, 1, 1, 0
        # )

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )

        self.model = AsymmetricCroCo3DStereoGaussian(
                        pos_embed='RoPE100', 
                        patch_embed_cls= 'ManyAR_PatchEmbed', 
                        img_size=(256, 256), 
                        freeze='encoder', # 'encoder_decoder_w_head'
                        head_type='dpt', 
                        output_mode='pts3d', 
                        depth_mode=('exp', -inf, inf), 
                        conf_mode=('exp', 1, inf), 
                        enc_embed_dim=1024, 
                        enc_depth=24, 
                        enc_num_heads=16, 
                        dec_embed_dim=768, 
                        dec_depth=12, 
                        dec_num_heads=12
                    )
            
        
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _extract_and_fuse_features(self, context: dict) -> torch.Tensor:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        
        # 构造batch数据
        view1 = dict()
        view2 = dict()
        view1["img"] = context["image"][:, 0]
        view1["org_img"] = context["image"][:, 0]
        view1["true_shape"] = repeat(torch.tensor(context["image"].shape[-2:]), "... -> b ...", b=b)
        view1["instance"] = ["v1"+str(i) for i in range(b)]

        view2["img"] = context["image"][:, 1]
        view2["org_img"] = context["image"][:, 1]
        view2["true_shape"] = repeat(torch.tensor(context["image"].shape[-2:]), "... -> b ...", b=b)
        view2["instance"] = ["v2"+str(i) for i in range(b)]

        batch = (view1, view2)
        
        # 将数据移动到设备上
        view1, view2 = batch
        ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
        for view in batch:
            for name in view.keys():  # pseudo_focal
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        
        # 创建对称batch并进行模型推理
        view1, view2 = make_batch_symmetric(batch)
        with torch.cuda.amp.autocast(enabled=False):
            pred1, pred2 = self.model(view1, view2)

        # 提取特征并重新排列
        features_00, features_10 = rearrange(pred1["features"][0].detach(), "(b v) c h w -> b v c h w", v=2)[:, 0], \
                        rearrange(pred1["features"][1].detach(), "(b v) c h w -> b v c h w", v=2)[:, 0]
        features_01, features_11 = rearrange(pred2["features"][0].detach(), "(b v) c h w -> b v c h w", v=2)[:, 1], \
                        rearrange(pred2["features"][1].detach(), "(b v) c h w -> b v c h w", v=2)[:, 1]

        # 特征融合 - 形状为 b 2 c h w
        trans_features = torch.stack((self.feature_fusion1(features_00, features_10), self.feature_fusion2(features_01, features_11)), dim=1)
        
        # 重新排列为 (b*v, c, h, w)
        trans_features = rearrange(trans_features, "b v c h w -> (b v) c h w")
        
        return trans_features

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        
        trans_features = self._extract_and_fuse_features(context)

        raw_correlation = self.corr_refine_net(trans_features)

        raw_correlation = trans_features + raw_correlation
        
        in_feats = rearrange(raw_correlation, "(b v) c h w -> b v c h w", v=2)
        
        extra_info = {}

        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel
        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=None,
        )
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel

        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )


    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )
            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None

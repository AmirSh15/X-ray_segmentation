# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List

import torch
import torch.nn.functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from captioning.utils.models.position_encoding import build_position_encoding
from captioning.utils.utils import NestedTensor
# from segmentation.data.dataloader import load_data
from segmentation.data.dataloader import load_data


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    # def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
    #     super().__init__()
    #     for name, parameter in backbone.named_parameters():
    #         if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
    #             parameter.requires_grad_(False)
    #     if return_interm_layers:
    #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    #     else:
    #         return_layers = {'layer4': "0"}
    #     self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    #     self.num_channels = num_channels

    def __init__(
        self,
        model: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        backbone = model.backbone
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "res2" not in name
                and "res3" not in name
                and "res4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"res1": "0", "res2": "1", "res3": "2", "res4": "3"}
        else:
            return_layers = {"res4": "0"}
        self.backbone = backbone
        self.model = model
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    #
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        backbone_cfg,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        # backbone2 = getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # num_channels2 = 512 if name in ('resnet18', 'resnet34') else 2048

        # input_shape = ShapeSpec(channels=len(backbone_cfg.MODEL.PIXEL_MEAN))
        # backbone = build_resnet_backbone(backbone_cfg, input_shape)

        # cfg = detectron_config()
        # all_cats = cfg.CATEGORIES

        for d in ["train", "val"]:
            DatasetCatalog.register(
                d,
                lambda d=d: load_data(
                    backbone_cfg, portion=d, pre_address="../segmentation/"
                ),
            )
            MetadataCatalog.get(d).set(thing_classes=backbone_cfg.CATEGORIES)
        metadata = MetadataCatalog.get("train")

        num_channels = (
            1024  # 1024 for mask_rcnn_R_101_C4_3x and 256 for mask_rcnn_R_50_C4_1x
        )
        trainer = DefaultTrainer(backbone_cfg)
        trainer.resume_or_load(resume=True)

        super().__init__(
            trainer.model, train_backbone, num_channels, return_interm_layers
        )


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(config):
    position_embedding = build_position_encoding(config)
    train_backbone = config.lr_backbone > 0
    return_interm_layers = False
    # if input_shape is None:

    backbone = Backbone(
        config.backbone_cfg,
        config.backbone,
        train_backbone,
        return_interm_layers,
        config.dilation,
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

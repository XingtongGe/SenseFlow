# MIT License
#
# Copyright (c) 2021 Intel ISL (Intel Intelligent Systems Lab)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Based on code from https://github.com/isl-org/DPT
#
# Utilities for VFMGAN discriminator with DINO ViT backbone.

import types
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from senseflow.utils import checkpoint


class AddReadout(nn.Module):
    """Add readout tokens (cls tokens) to patch embeddings."""
    def __init__(self, start_index: int = 1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class Transpose(nn.Module):
    """Transpose tensor dimensions."""
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()


# Global variable to store activations from forward hooks
activations = {}


def get_activation(name: str) -> Callable:
    """Create a forward hook to capture activations."""
    def hook(model, input, output):
        activations[name] = output
    return hook


def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int) -> torch.Tensor:
    """Resize position embeddings to match the input image size."""
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x: torch.Tensor, in_check=False) -> torch.Tensor:
    """
    Flexible forward pass that handles variable input sizes.
    Patch projection and dynamic position embedding resizing.
    """
    # patch proj and dynamically resize
    B, C, H, W = x.size()
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    pos_embed = self._resize_pos_embed(
        self.pos_embed, H // self.patch_size[1], W // self.patch_size[0]
    )

    # add cls token
    cls_tokens = self.cls_token.expand(
        x.size(0), -1, -1
    )
    x = torch.cat((cls_tokens, x), dim=1)

    # forward pass
    x = x + pos_embed
    x = self.pos_drop(x)
    
    for blk in self.blocks:
        x = checkpoint(blk, (x,), None, in_check)

    x = self.norm(x)
    return x


def make_vit_backbone(
    model: nn.Module,
    patch_size=[16, 16],
    hooks=[2, 5, 8, 11],
    hook_patch=True,
    start_index=1,
):
    """
    Create a Vision Transformer backbone with hooks to extract intermediate features.
    
    Args:
        model: timm VisionTransformer model
        patch_size: Patch size for the ViT model
        hooks: List of block indices to extract features from
        hook_patch: Whether to also hook the patch embedding layer
        start_index: Start index for readout tokens (cls tokens)
    
    Returns:
        pretrained: Module with hooked model and rearrange operations
    """
    pretrained = nn.Module()
    pretrained.model = model
    
    # Register forward hooks to capture activations at specified blocks
    for i in range(len(hooks)):
        pretrained.model.blocks[hooks[i]].register_forward_hook(get_activation('%d' % i))

    if hook_patch:
        pretrained.model.pos_drop.register_forward_hook(get_activation('%d' % (len(hooks))))
    
    # Configure readout operations
    pretrained.rearrange = nn.Sequential(AddReadout(start_index), Transpose(1, 2))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = patch_size

    # Inject flexible forward method to handle variable input sizes
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def forward_vit(pretrained: nn.Module, x: torch.Tensor, in_check=False, return_cls=False) -> dict:
    """
    Forward pass through ViT backbone and extract features.
    
    Args:
        pretrained: Module returned by make_vit_backbone
        x: Input tensor [B, C, H, W]
        in_check: Whether to use gradient checkpointing
        return_cls: Whether to return cls token only
    
    Returns:
        Dictionary of activations {hook_name: feature_tensor}
    """
    _, _, H, W = x.size()
    _ = pretrained.model.forward_flex(x, in_check)
    if return_cls:
        return {k: v[:, 0:1,].transpose(1, 2).contiguous() for k, v in activations.items()}
    return {k: pretrained.rearrange(v) for k, v in activations.items()}


"""
VFMGAN discriminator architecture.
This module contains the ProjectedDiscriminatorPlus class used in SenseFlow training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize, CenterCrop
import torchvision.transforms.functional as TF
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .vfmgan_utils import make_vit_backbone, forward_vit
from .diffaug import DiffAugment
from senseflow.utils import checkpoint
from torch_utils.ops import bias_act

from typing import Callable


class GANLoss(nn.Module):
    """Define GAN loss.
    
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(
        self,
        gan_type="vanilla",
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0,
    ):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":
            self.loss = self._wgan_loss
        elif self.gan_type == "wgan_softplus":
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    def _wgan_loss(self, input, target):
        """wgan loss.
        
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.
        
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        
        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """
        if self.gan_type in ["wgan", "wgan_softplus"]:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, keepdim=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == "hinge":
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                if keepdim:
                    loss = self.loss(1 + input).mean(dim=(1))
                else:
                    loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                if keepdim:
                    loss = -input.mean(dim=(1))
                else:
                    loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        return loss if is_disc else loss * self.loss_weight


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = 'linear',
        lr_multiplier: float = 1.0,
        weight_init: float = 1.0,
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class SpectralConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        if len(shape) == 4:  # b c h w -> b c l
            x = x.reshape(shape[0], shape[1], -1)

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


def make_block_2d(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHeadPlus(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64, useatt=False, ret_cls=False, downsample=0):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim
        self.useatt = useatt

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9 if not ret_cls else 1))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.refmapper = nn.Sequential(
                make_block(channels, kernel_size=1),
                ResidualBlock(make_block(channels, kernel_size=9 if not ret_cls else 1)),
                SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
            )
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)
        
        if self.useatt:
            raise NotImplementedError(
                "useatt=True requires BasicTransformerBlock which is not included in the open-source release. "
                "All default training configs use useatt=False."
            )
            cc = 64
            self.att_block = BasicTransformerBlock(dim=cc, d_head=16, n_heads=cc // 16, dropout=0., context_dim=cc, checkpoint=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor, refimg) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        assert self.c_dim > 0
        cmap = self.cmapper(c).unsqueeze(-1)
        refmap = self.refmapper(refimg)

        if self.useatt:
            out = self.att_block(out.permute(0, 2, 1), refmap.permute(0, 2, 1)).permute(0, 2, 1)

        out = (out * (cmap + refmap)).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out


class DiscHeadPlus_2d(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64, useatt=False, ret_cls=False, downsample=0):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim
        if useatt or ret_cls:
            assert NotImplemented

        self.main = nn.Sequential(
            make_block_2d(channels, kernel_size=1),
            SpectralConv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular', stride=2) if downsample > 0 else nn.Identity(),
            ResidualBlock(make_block_2d(channels, kernel_size=3)),
            SpectralConv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular', stride=2) if downsample > 1 else nn.Identity(),
            ResidualBlock(make_block_2d(channels, kernel_size=3)),
            SpectralConv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular', stride=2) if downsample > 2 else nn.Identity(),
        )

        self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
        self.refmapper = nn.Sequential(
            make_block_2d(channels, kernel_size=1),
            SpectralConv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular', stride=2) if downsample > 0 else nn.Identity(),
            ResidualBlock(make_block_2d(channels, kernel_size=3)),
            SpectralConv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular', stride=2) if downsample > 1 else nn.Identity(),
            ResidualBlock(make_block_2d(channels, kernel_size=3)),
            SpectralConv2d(channels, channels, kernel_size=3, padding=1, padding_mode='circular', stride=2) if downsample > 2 else nn.Identity(),
            SpectralConv2d(channels, cmap_dim, kernel_size=1, padding=0)
        )
        self.cls = SpectralConv2d(channels, cmap_dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor, refimg) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        cmap = self.cmapper(c).unsqueeze(-1).unsqueeze(-1)
        refmap = self.refmapper(refimg)
        out = (out * (cmap + refmap)).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class DINO(torch.nn.Module):
    def __init__(self, hooks=[2, 5, 8, 11], hook_patch=True, dino_name='vit_small_patch16_224_dino', patch_size=16, use_checkpoint=False, 
                 fix_res_dino=True, ret_cls=False, dino_pretrain=True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)
        self.fix_res_dino = fix_res_dino
        print('DINO pretraining', dino_pretrain)
        self.model = make_vit_backbone(
            timm.create_model(dino_name, pretrained=dino_pretrain),
            patch_size=[patch_size, patch_size], hooks=hooks, hook_patch=hook_patch,
        )
        self.model = self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        self.use_checkpoint = use_checkpoint
        self.ret_cls = ret_cls
        if self.use_checkpoint:
            print('** the passed use_checkpoint to DINO is no use')
        self.in_check = True
        print('self in check', self.in_check)

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: x in [0, 1]; output: dict of activations"""
        if self.fix_res_dino:
            x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        features = forward_vit(self.model, x, self.in_check, return_cls=self.ret_cls)
        return features


class ProjectedDiscriminatorPlus(nn.Module):
    """
    Projected discriminator for VFMGAN training.
    Based on StyleGAN-T architecture with DINO ViT backbone.
    """
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5, dino_name='vit_large_patch14_dinov2.lvd142m', hooks=None,
                 crop_plan='short', use_checkpoint=False, fix_res_dino=True, useatt=False, ret_cls=False, conv2d=False, downsample=0, dino_pretrain=True):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop
        self.crop_plan = crop_plan
        self.conv2d = conv2d
        self.dino_pretrain = dino_pretrain

        if hooks is None:
            hooks = [2, 5, 8, 11]
            if 'large' in dino_name:
                hooks = [item * 2 + 1 for item in hooks]
                hooks = [5, 10, 14, 19, 23]
        print('DINO hooks', hooks)
        ps = dino_name.split('_patch')[-1].split('_')[0]
        self.dino = DINO(dino_name=dino_name, patch_size=int(ps), hooks=hooks, use_checkpoint=use_checkpoint, 
                         fix_res_dino=fix_res_dino, ret_cls=ret_cls, dino_pretrain=dino_pretrain)

        if self.conv2d:
            headfunc = DiscHeadPlus_2d
        else:
            headfunc = DiscHeadPlus

        heads = []
        for i in range(self.dino.n_hooks):
            heads += [str(i), headfunc(self.dino.embed_dim, c_dim, useatt=useatt, ret_cls=ret_cls, downsample=downsample)],
        self.heads = nn.ModuleDict(heads)

        self.dino_fp16 = False

    def train(self, mode: bool = True):
        if self.dino_pretrain:
            print('DINO force to set train = False')
            self.dino = self.dino.train(False)
        else:
            print('DINO train mode', mode)
            self.dino = self.dino.train(mode)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor, ref, timesteps=None, return_ref=False) -> torch.Tensor:
        # Apply augmentation (x in [-1, 1])
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')
            assert 'ref image show do the same'

        # Transform to [0, 1]
        x = x.add(1).div(2)
        ref = ref.add(1).div(2)

        # Handle different crop plans
        if self.crop_plan in ['long', 'longms', 'longmms']:
            # Take crops with probability p_crop if the image is larger
            if np.random.random() < self.p_crop:
                short_len = min(list(x.size())[-2:])
                ratio = (self.dino.img_resolution + 1) / short_len
                if 'mms' in self.crop_plan:
                    ratio = ratio * np.random.choice([1.0, 1.25, 1.5, 1.75, 1.8, 2.0])
                x = F.interpolate(x, scale_factor=ratio, mode='area')
                ref = F.interpolate(ref, scale_factor=ratio, mode='area')
                i, j, h, w = RandomCrop.get_params(
                    x, output_size=(self.dino.img_resolution, self.dino.img_resolution))
                x = TF.crop(x, i, j, h, w)
                ref = TF.crop(ref, i, j, h, w)
            else:
                long_len = max(list(x.size())[-2:])
                ratio = self.dino.img_resolution / long_len
                if 'ms' in self.crop_plan:
                    ratio = ratio * np.random.choice([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25])
                x = F.interpolate(x, scale_factor=ratio, mode='area')
                ref = F.interpolate(ref, scale_factor=ratio, mode='area')
                _, _, H, W = x.shape
                pad = [0, 0, 0, 0]
                if H < self.dino.img_resolution:
                    pad[2] = (self.dino.img_resolution - H) // 2
                    pad[3] = self.dino.img_resolution - H - pad[2]
                if W < self.dino.img_resolution:
                    pad[0] = (self.dino.img_resolution - W) // 2
                    pad[1] = self.dino.img_resolution - W - pad[0]
                x = F.pad(x, pad, "constant", 0)
                ref = F.pad(ref, pad, "constant", 0)

        elif self.crop_plan == 'short':
            # Handle large ratio images
            short_len = min(list(x.size())[-2:])
            ratio = self.dino.img_resolution / short_len
            x = F.interpolate(x, scale_factor=ratio, mode='area')
            ref = F.interpolate(ref, scale_factor=ratio, mode='area')
            # Take crops with probability p_crop if the image is larger
            if np.random.random() < self.p_crop:
                i, j, h, w = RandomCrop.get_params(
                    x, output_size=(self.dino.img_resolution, self.dino.img_resolution))
                x = TF.crop(x, i, j, h, w)
                ref = TF.crop(ref, i, j, h, w)
            else:
                assert x.shape == ref.shape
                x = CenterCrop(self.dino.img_resolution)(x)
                ref = CenterCrop(self.dino.img_resolution)(ref)
        elif self.crop_plan.lower() == 'none':
            pass
        else:
            assert NotImplemented

        # Forward pass through DINO ViT
        if self.dino_fp16:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=False):
                features = self.dino(x)
                ref_features = self.dino(ref)
        else:
            features = self.dino(x)
            ref_features = self.dino(ref)
        
        _, _, HH, WW = x.shape
        
        if self.conv2d:  # transform to 2d features
            hh, ww = HH // 14, WW // 14
            trans = lambda x: x.reshape(x.shape[0], x.shape[1], hh, ww)
            for kk in features.keys():
                features[kk] = trans(features[kk])
                ref_features[kk] = trans(ref_features[kk])
        
        # Apply discriminator heads
        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[k], c, ref_features[k].requires_grad_(True)).view(x.size(0), -1))
        logits = torch.cat(logits, dim=1)
        
        if not return_ref:
            return logits
        else:
            return logits, ref, ref_features


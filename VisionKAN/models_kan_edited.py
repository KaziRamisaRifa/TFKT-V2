# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, Block, Attention
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from .fasterkan import FasterKAN as KAN

from torch.utils.checkpoint import checkpoint

__all__KAN = [
    'deit_base_patch16_224_KAN', 'deit_small_patch16_224_KAN',  
    'deit_base_patch16_384_KAN', 'deit_tiny_patch16_224_KAN', 
    'deit_tiny_distilled_patch16_224_KAN', 'deit_base_distilled_patch16_224_KAN', 
    'deit_small_distilled_patch16_224_KAN', 'deit_base_distilled_patch16_384_KAN']


__all__ViT = [
    'deit_base_patch16_224_ViT', 'deit_small_patch16_224_ViT',  
    'deit_base_patch16_384_ViT', 'deit_tiny_patch16_224_ViT', 
    'deit_tiny_distilled_patch16_224_ViT', 'deit_base_distilled_patch16_224_ViT', 
    'deit_small_distilled_patch16_224_ViT', 'deit_base_distilled_patch16_384_ViT']


class kanBlock(Block):
    def __init__(self, dim, num_heads=8, hdim_kan=192, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads)
        print(f"Initializing kanBlock with dim={dim} and num_heads={num_heads}")
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.norm1 = norm_layer(dim)  # Apply LayerNorm based on the input `dim`
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # Same norm, applied after reshaping
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.kan = nn.Sequential(
            nn.Linear(dim, hdim_kan),
            act_layer(),
            nn.Linear(hdim_kan, dim)
        )

    def forward(self, x):
        # Input is likely 4D (batch_size, channels, height, width), reshape it
        b, c, h, w = x.shape

        # Define the patch size (you can modify this as needed)
        patch_size = 16
        assert h % patch_size == 0 and w % patch_size == 0, 'Height and width must be divisible by patch_size'

        # Reshape the input into patches of size (patch_size, patch_size)
        patches_h = h // patch_size
        patches_w = w // patch_size
        x = x.reshape(b, c, patches_h, patch_size, patches_w, patch_size).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, patches_h * patches_w, c * patch_size * patch_size)

        print(f"kanBlock input shape after reshaping: {x.shape}")  # Now it's (batch_size, num_patches, dim)

        # Apply LayerNorm dynamically and the attention mechanism
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # Apply LayerNorm and the MLP/kan layer
        x = x + self.drop_path(self.kan(self.norm2(x)))

        # Reshape the tensor back to its original 4D format (batch_size, channels, height, width)
        x = x.view(b, patches_h, patches_w, c, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

        print(f"kanBlock output shape after reshaping back: {x.shape}")
        return x



class VisionKAN(VisionTransformer):
    def __init__(self, *args, num_heads=8, batch_size=16, **kwargs):
        print("Initializing VisionKAN")
        if 'hdim_kan' in kwargs:
            self.hdim_kan = kwargs['hdim_kan']
            del kwargs['hdim_kan']
        else:
            self.hdim_kan = 192
        
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        # For newer version timm they don't save the depth to self.depth, so we need to check it
        try:
            self.depth
        except AttributeError:
            if 'depth' in kwargs:
                self.depth = kwargs['depth']
            else:
                self.depth = 12

        block_list = [
            kanBlock(dim=self.embed_dim, num_heads=self.num_heads, hdim_kan=self.hdim_kan)
            for i in range(self.depth)
        ]
        # check the origin type of the block is torch.nn.modules.container.Sequential
        # if the origin type is torch.nn.modules.container.Sequential, then we need to convert it to a list
	
        # if the origin type is torch.nn.modules.container.Sequential, then we need to convert it to a list
        if isinstance(self.blocks, nn.Sequential):
            self.blocks = nn.Sequential(*block_list)
        elif isinstance(self.blocks, nn.ModuleList):
            self.blocks = nn.ModuleList(block_list)



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

def create_kan(model_name, pretrained, **kwargs):

    if model_name == 'deit_tiny_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_name == 'deit_small_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_384_KAN':
        model = VisionKAN(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_tiny_distilled_patch16_224_KAN':
        raise RuntimeError('Distilled models are not yet implmented in KAN')

    elif model_name == 'deit_small_distilled_patch16_224_KAN':
        raise RuntimeError('Distilled models are not yet implmented in KAN')

    elif model_name == 'deit_base_distilled_patch16_224_KAN':
        raise RuntimeError('Distilled models are not yet implmented in KAN')

def create_ViT(model_name, pretrained, **kwargs):
    if 'batch_size' in kwargs:
        del kwargs['batch_size']
    if model_name == 'deit_base_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_name == 'deit_small_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_384_ViT':
        model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_tiny_distilled_patch16_224_ViT':
        model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_name == 'deit_small_distilled_patch16_224_ViT':
        model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_distilled_patch16_224_ViT':
        model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_distilled_patch16_384_ViT':
        model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model


def create_model(model_name,**kwargs):
    pretrained = kwargs['pretrained'] if 'pretrained' in kwargs else False
    if 'pretrained' in kwargs:
        del kwargs['pretrained']
    print(kwargs)
    if model_name in __all__KAN:
        model = create_kan(model_name, pretrained, **kwargs)
        model.default_cfg = _cfg()
        return model
    elif model_name in __all__ViT:
        model = create_ViT(model_name, pretrained, **kwargs)
        model.default_cfg = _cfg()
        return model
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

if __name__ == '__main__':
    model = deit_tiny_patch16_224().cuda()
    img = torch.randn(5, 3, 224, 224).cuda()
    out = model(img)
    print(out.shape)


"""
vit implementation 
"""

import torch

import torch.nn as nn 

from learning.transformer.torch_arch import TransformerBlock


class TorchVisionTransformer(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, emb_dim: int, ffn_mult: int, num_classes: int, patch_size: int, img_size: int):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        self.blocks = nn.ModuleList(
            TransformerBlock(num_heads, emb_dim, ffn_mult)
            for _ in range(num_layers)
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.output = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) # (B, emb_dim, H', W')
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, emb_dim)
        

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        print(f"{x.shape=}\n {self.pos_embed.shape}")
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        x = self.output(x)

        return x


        



if __name__ == "__main__":
    vit_config= {
        "num_layers": 10,
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4,
        "num_classes": 10,
        "patch_size": 14,
        "img_size": 224,
    }


    model = TorchVisionTransformer(**vit_config)
    rand_tensor = torch.rand(10, 3, 224, 224)
    output = model(rand_tensor)


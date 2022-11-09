from src.config import Config
import torch.nn as nn

try:
    from vit_pytorch import ViT
except ImportError:
    print("'vit_pytorch' is not installed. Install with 'pip install vit-pytorch'")


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Transformer, self).__init__()
        self.dim = config.transformer_dim
        self.num_classes = config.num_classes

        self.model = ViT(
            image_size=config.resize_size[0],
            patch_size=config.transformer_patch_size,
            num_classes=self.num_classes,
            dim=self.dim,
            depth=config.transformer_depth,
            heads=config.transformer_heads,
            mlp_dim=config.transformer_mlp_dim,
            dropout=config.transformer_dropout,
            emb_dropout=config.transformer_emb_dropout,
            channels=1,
        )

        self.model.mlp_head = nn.Identity()
        self.heads = nn.ModuleDict()
        for head in config.heads:
            self.add_head(head)

    def add_head(self, name):
        self.heads.add_module(
            name,
            nn.Sequential(
                nn.LayerNorm(self.dim), nn.Linear(self.dim, self.num_classes)
            ),
        )

    def forward(self, x):
        x = self.model(x)
        heads = {name: head(x) for name, head in self.heads.items()}
        return heads

    def forward_single_head(self, x):
        x = self.model(x)
        return self.heads["n_counts"](x)

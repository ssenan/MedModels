import torch
import torch.nn as nn
import torch.nn.functional as F 

from modules import BaseGenBlock, AddGenBlock, AddDisBlock, FinalBlockClasses

def nf(stage, fmap_base, fmap_max):
            return min(int(fmap_base) / (2 ** stage), fmap_max)

class Generator(nn.Module):
    def __init__(self, depth, num_channels, latent_size):
        super().__init__()

        self.depth = depth
        self.latent_size = latent_size
        self.num_channels = num_channels

        self.layers = nn.ModuleList([BaseGenBlock(latent_size, nf(1))])

        for stage in range(1, depth - 1):
            self.layers.append(AddGenBlock(nf(stage), nf(stage+1)))

        self.rgb = nn.ModuleList(
            [nn.Conv3d(nf(stage), num_channels, kernel_size=(1, 1, 1))
            for stage in range(1, depth)]
        )
  
    def forward(self, x, depth, alpha):
        """
        Generator forward pass

        Args:
            x: latent noise
            depth: depth of input
            alpha: alpha value for fade-in
        """
        depth = self.depth if depth is None else depth

        if depth == 2:
            y = self.rgb[0](self.layers[0](x))

        else:
            y = x
            for layer_block in self.layers[: depth - 2]:
                y = layer_block(y)
            residual = nn.interpolate(self.rgb[depth - 3](y), scale_factor=2)
            straight = self.rgb_converters[depth - 2](self.layers[depth - 2](y))
            y = (alpha * straight) + ((1 - alpha) * residual)
        return y

    
    def create_save(self):
        return {
            "conf": {
                "depth": self.depth,
                "num_channels": self.num_channels,
                "latent_size": self.latent_size
            },
            "state_dict": self.state_dict()
        }

class Discriminator(nn.Module):
    def __init__(
        self,
        depth,
        num_channels,
        latent_size,
        num_classes,
        ):
        super().__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.conditional = self.num_classes is not None

        if self.conditional:
            self.layers = [FinalBlockClasses(nf(1), latent_size, num_classes)]

        for stage in range(1, depth - 1):
            self.layers.insert(
                0, AddDisBlock(nf(stage+1), nf(stage))
            )
        self.layers = nn.ModuleList(self.layers)
        self.from_rgb = nn.ModuleList([
            reversed(
                nn.Sequential(
                   nn.Conv3d(num_channels, nf(stage), kernel_size=(1, 1, 1)),
                   nn.LeakyReLU(0.2) 
                )
                for stage in range(1, depth)
            )
        ]
    )
    def forward(self, x, depth, alpha, labels):
        if depth > 2:
            residual = self.from_rgb[-(depth - 2)](
                nn.AvgPool3d(x, kernel_size=2, stride=2)
            )
            straight = self.layers[-(depth - 1)](self.from_rgb[-(depth - 1)](x))
            y = (alpha * straight) + ((1 - alpha) * residual)
            for layer_block in self.layers[-(depth - 2) : -1]:
                y = layer_block(y)
        else:
            y = self.from_rgb[-1](x)
        if self.conditional:
            y = self.layers[-1](y, labels)
        else:
            y = self.layers[-1](y)
        return y

    def create_save(self):
        return {
            "conf": {
                "depth": self.depth,
                "num_channels": self.num_channels,
                "latent_size": self.latent_size,
                "use_eql": self.use_eql,
                "num_classes": self.num_classes,
            },
            "state_dict": self.state_dict(),
        }

def create_generator(saved_model):
    loaded = torch.load(saved_model)
  
    # Create generator
    generator_info = (
        loaded['saved_generator']
        if "saved_generator" in loaded
        else loaded["generator"]
    )
    generator = Generator(**generator_info["conf"])
    generator.load_state_dict(generator_info["state_dict"])

    return generator
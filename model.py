import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def get_model(model_name : str, config):
    if model_name == "FeatureCubeMLP":
        return FeatureCubeMLP(config)
    elif model_name == "VoxelMLP":
        return VoxelMLP(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False):
        super().__init__()
        self.is_last = is_last
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU(inplace=True)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # ReLU Kaiming (He) Initialization
            nn.init.kaiming_normal_(self.linear.weight, a=0, mode='fan_in', nonlinearity='relu')
            
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        
        if self.is_last:
            return x
            
        return self.activation(x)

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False):
        super().__init__()
        self.in_features = in_features
        self.w0 = 30.0
        self.is_first = is_first
        self.is_last = is_last
        
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
        
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        
        if self.is_last:
            return x
            
        return torch.sin(self.w0 * x)

def positional_encoding(coords: torch.Tensor, num_frequencies: int, include_input: bool = True) -> torch.Tensor:
    """Sinusoidal positional encoding on UVW in [0,1]; returns [sin, cos] bands (and raw if enabled)."""
    if num_frequencies <= 0:
        return coords if include_input else coords.new_empty((coords.shape[0], 0))

    freq_bands = (2.0 ** torch.arange(num_frequencies, device=coords.device, dtype=coords.dtype)).view(1, -1, 1)
    scaled = coords.unsqueeze(1) * freq_bands * (2.0 * math.pi)
    sin_enc = torch.sin(scaled)
    cos_enc = torch.cos(scaled)
    enc = torch.cat([sin_enc, cos_enc], dim=1).reshape(coords.shape[0], -1)
    return torch.cat([coords, enc], dim=-1) if include_input else enc

class VoxelMLP(nn.Module):
    """
    Voxel MLP Model
    Maps 3D coordinates (XYZ) to 3D color values (RGB).
    """
    def __init__(self, config):
        super(VoxelMLP, self).__init__()
        self.config = config

        if config.activation == 'siren':
            LayerType = SirenLayer
        elif config.activation == 'relu':
            LayerType = ReLULayer
        else:
            raise ValueError(f"Unknown layer type: {config.layer_type}")
        
        # Build network layers
        layers = []
        
        # Input layer -> First hidden layer
        layers.append(LayerType(
            config.input_dim, 
            config.hidden_dims[0], 
            is_first=True,
        ))
        
        # Hidden layers
        for i in range(1, len(config.hidden_dims)):
            layers.append(LayerType(
                config.hidden_dims[i - 1], 
                config.hidden_dims[i], 
            ))
            
        # Last hidden layer -> Output layer
        layers.append(LayerType(
            config.hidden_dims[-1], 
            config.output_dim, 
            is_last=True
        ))
        
        # Combine into Sequential container
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class FeatureCubeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.activation == 'siren':
            LayerType = SirenLayer
        elif config.activation == 'relu':
            LayerType = ReLULayer
        else:
            raise ValueError(f"Unknown layer type: {getattr(config, 'activation', None)}")

        feature_dim = getattr(config, "feature_cube_channels", None)
        if feature_dim is None:
            feature_dim = getattr(config, "feature_dim", None)
        if feature_dim is None:
            feature_dim = config.hidden_dims[0]
        feature_dim = int(feature_dim)

        self.pe_frequencies = int(getattr(config, "feature_pe_frequencies", 0))
        if self.pe_frequencies < 0:
            raise ValueError("feature_pe_frequencies must be >= 0")
        pe_dim = 3 + 6 * self.pe_frequencies

        plane_dim = getattr(config, "feature_cube_dim", None)
        if plane_dim is None:
            plane_dim = getattr(config, "volume_dim", None)
        if plane_dim is None or len(plane_dim) != 3:
            raise ValueError("Config must define feature_cube_dim (nx, ny, nz) or volume_dim")
        nx, ny, nz = int(plane_dim[0]), int(plane_dim[1]), int(plane_dim[2])

        # Three feature planes: XY (ny,nx), YZ (ny,nz), XZ (nz,nx)
        self.plane_xy = nn.Parameter(torch.randn(1, feature_dim, ny, nx) * 0.01)
        self.plane_yz = nn.Parameter(torch.randn(1, feature_dim, ny, nz) * 0.01)
        self.plane_xz = nn.Parameter(torch.randn(1, feature_dim, nz, nx) * 0.01)

        layers = []
        layers.append(LayerType(3 * feature_dim + pe_dim, config.hidden_dims[0], is_first=True))
        for i in range(1, len(config.hidden_dims)):
            layers.append(LayerType(config.hidden_dims[i - 1], config.hidden_dims[i]))
        layers.append(LayerType(config.hidden_dims[-1], config.output_dim, is_last=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != 3:
            raise ValueError(f"FeatureCubeMLP expects input shape (N, 3); got {tuple(x.shape)}")

        pe = positional_encoding(x, self.pe_frequencies)
        n = x.size(0)

        # Build grids in [-1, 1] for each plane
        xy_grid = (x[:, [0, 1]] * 2.0 - 1.0).view(n, 1, 1, 2)
        yz_grid = torch.stack([x[:, 2], x[:, 1]], dim=-1).mul_(2.0).sub_(1.0).view(n, 1, 1, 2)
        xz_grid = torch.stack([x[:, 0], x[:, 2]], dim=-1).mul_(2.0).sub_(1.0).view(n, 1, 1, 2)

        feat_xy = F.grid_sample(
            self.plane_xy.expand(n, -1, -1, -1),
            xy_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(n, -1)

        feat_yz = F.grid_sample(
            self.plane_yz.expand(n, -1, -1, -1),
            yz_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(n, -1)

        feat_xz = F.grid_sample(
            self.plane_xz.expand(n, -1, -1, -1),
            xz_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(n, -1)

        features = torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
        if pe.numel() > 0:
            features = torch.cat([features, pe], dim=-1)
        return self.mlp(features)

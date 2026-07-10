import torch
import torch.nn as nn
import torch.nn.functional as F



# 43 features, 6 classes, 33 users
class HAR_Model0(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6):
        super(HAR_Model0, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class HAR_Model1(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6):
        super(HAR_Model1, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.25),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class HAR_ModelDeep(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6, depth = 4):
        super(HAR_ModelDeep, self).__init__()
        
        self.in_proj = nn.Linear(input_size, hidden_size)
        layers = []

        for _ in range(depth):
            layer = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            )
            layers.append(layer)

        self.model = nn.ModuleList(layers)

        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.model:
            x = layer(x)
        x = self.out_proj(x)
        return x
    
    
class HAR_ModelDeepAdjst(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6,
                 depth = 4,
                 activation = 'GELU',
                 dropout = 0.25,
                 norm = 'LayerNorm',
                 residual = False):
        super(HAR_ModelDeepAdjst, self).__init__()
        self.residual = residual
        self.in_proj = nn.Linear(input_size, hidden_size)
        layers = []

        activation_fn = None
        norm_fn = None

        match activation:
            case 'GELU':
                activation_fn = nn.GELU
            case 'ReLU':
                activation_fn = nn.ReLU
            case 'SiLU':
                activation_fn = nn.SiLU
            case 'Tanh':
                activation_fn = nn.Tanh
            case 'LeakyReLU':
                activation_fn = nn.LeakyReLU
            case 'PReLU':
                activation_fn = nn.PReLU
            case _:
                raise ValueError(f"Unsupported activation function: {activation}")

        match norm:
            case 'LayerNorm':
                norm_fn = nn.LayerNorm
            case 'BatchNorm1d':
                norm_fn = nn.BatchNorm1d
            case 'Identity':
                norm_fn = nn.Identity
            case _:
                raise ValueError(f"Unsupported normalization function: {norm}")
                    
        for _ in range(depth):
            layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            norm_fn(hidden_size),
            activation_fn () if activation_fn != nn.PReLU else activation_fn(hidden_size)
            )
            layers.append(layer)

        self.model = nn.ModuleList(layers)

        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.in_proj(x)
        residual = x
        for layer in self.model:
            x = layer(x) + residual if self.residual else layer(x)
        x = self.out_proj(x)
        return x
    
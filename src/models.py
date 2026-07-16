"""Model definitions for WISDM human-activity classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Keep the simplest baseline available for quick comparison against deeper variants.
class HAR_Model0(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6):
        """Build a minimal two-layer baseline classifier.

        Parameters
        ----------
        input_size : int, optional
            Number of input features, by default 43.
        hidden_size : int, optional
            Width of the hidden layer, by default 128.
        output_size : int, optional
            Number of activity classes, by default 6.
        """
        super(HAR_Model0, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """Run a batch of features through the baseline classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input feature batch.

        Returns
        -------
        torch.Tensor
            Logits for each activity class.
        """
        x = self.model(x)
        return x
    
# Add light regularization and normalization for a stronger shallow reference model.
class HAR_Model1(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6):
        """Build a shallow classifier with dropout and layer normalization.

        Parameters
        ----------
        input_size : int, optional
            Number of input features, by default 43.
        hidden_size : int, optional
            Width of the hidden layer, by default 128.
        output_size : int, optional
            Number of activity classes, by default 6.
        """
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
        """Run the regularized shallow network on a batch of features.

        Parameters
        ----------
        x : torch.Tensor
            Input feature batch.

        Returns
        -------
        torch.Tensor
            Logits for each activity class.
        """
        x = self.model(x)
        return x
    
# Use a deeper stack when capacity matters more than simplicity.
class HAR_ModelDeep(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6, depth = 4):
        """Build a deeper feed-forward classifier with repeated hidden blocks.

        Parameters
        ----------
        input_size : int, optional
            Number of input features, by default 43.
        hidden_size : int, optional
            Width of each hidden block, by default 128.
        output_size : int, optional
            Number of activity classes, by default 6.
        depth : int, optional
            Number of repeated hidden blocks, by default 4.
        """
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
        """Propagate features through the deeper classifier stack.

        Parameters
        ----------
        x : torch.Tensor
            Input feature batch.

        Returns
        -------
        torch.Tensor
            Logits for each activity class.
        """
        x = self.in_proj(x)
        for layer in self.model:
            x = layer(x)
        x = self.out_proj(x)
        return x
    
    
# Keep the adjustable variant flexible for search and training experiments.
class HAR_ModelDeepAdjst(nn.Module):
    def __init__(self, input_size: int = 43, hidden_size: int = 128, output_size: int = 6,
                 depth = 4,
                 activation = 'GELU',
                 dropout = 0.25,
                 norm = 'LayerNorm',
                 residual = False):
        """Build the configurable deep classifier used for training and tuning.

        Parameters
        ----------
        input_size : int, optional
            Number of input features, by default 43.
        hidden_size : int, optional
            Width of the hidden layers, by default 128.
        output_size : int, optional
            Number of activity classes, by default 6.
        depth : int, optional
            Number of repeated hidden blocks, by default 4.
        activation : str, optional
            Activation name to instantiate, by default 'GELU'.
        dropout : float, optional
            Dropout probability inside each block, by default 0.25.
        norm : str, optional
            Normalization layer name, by default 'LayerNorm'.
        residual : bool, optional
            Whether to add residual connections, by default False.
        """
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
                    
        # Assemble identical blocks so search and config changes stay localized.
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
        """Run the configurable classifier and optionally apply residual links.

        Parameters
        ----------
        x : torch.Tensor
            Input feature batch.

        Returns
        -------
        torch.Tensor
            Logits for each activity class.
        """
        x = self.in_proj(x)
        residual = x
        for layer in self.model:
            x = layer(x) + residual if self.residual else layer(x)
        x = self.out_proj(x)
        return x
    
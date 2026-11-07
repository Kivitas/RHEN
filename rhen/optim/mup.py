import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple

class RecurrentMuP:
    """
    Model Upstream Parallelization (mUP) scaling for recurrent models.
    
    Implements the mUP scaling factors that adjust initialization and learning rates
    based on model architecture to ensure stable training across different hidden
    dimensions and loop depths.
    
    Key features:
    - Width scaling: Init std adjusted by sqrt(base_hidden / hidden_size)
    - Depth scaling: Learning rate scaled by sqrt(fan_in_0 / fan_in)
    - Loop-aware scaling: Accounts for repeated application of shared core
    
    Reference: mUP paper on scaling initialization and LR for large models.
    """
    
    def __init__(
        self,
        hidden_size: int,
        base_hidden: int = 1024,
        base_loops: int = 16,
        num_recurrent_layers: int = 4,
        max_recurrent_loops: int = 12,
        lr_scale_factor: float = 1.0,
    ):
        self.hidden_size = hidden_size
        self.base_hidden = base_hidden
        self.base_loops = base_loops
        self.num_recurrent_layers = num_recurrent_layers
        self.max_recurrent_loops = max_recurrent_loops
        self.lr_scale_factor = lr_scale_factor
        
        # Compute scaling factors
        self.width_scale = math.sqrt(base_hidden / hidden_size)
        self.depth_scale = self._compute_depth_scale()
        self.loop_scale = self._compute_loop_scale()
    
    def _compute_depth_scale(self) -> float:
        """
        Compute depth scaling factor based on number of recurrent layers.
        
        For recurrent models, we scale initialization to account for the
        repeated application of shared layers through time.
        """
        base_depth = self.num_recurrent_layers * self.base_loops
        actual_depth = self.num_recurrent_layers * self.max_recurrent_loops
        
        # Scale factor for depth: sqrt(base_depth / actual_depth)
        return math.sqrt(base_depth / max(actual_depth, 1))
    
    def _compute_loop_scale(self) -> float:
        """
        Compute loop scaling factor for recurrent depth.
        
        This accounts for the fact that the same core is applied multiple times,
        so gradients accumulate differently.
        """
        # Scale based on loop ratio
        loop_ratio = self.max_recurrent_loops / max(self.base_loops, 1)
        return 1.0 / math.sqrt(loop_ratio)
    
    def get_init_scale(self, module: nn.Module, name: str) -> float:
        """
        Get the initialization scale for a specific module.
        
        Args:
            module: The module to get scale for
            name: Module name (used for special handling)
            
        Returns:
            Scaling factor for initialization std
        """
        base_scale = self.width_scale
        
        # Apply depth scaling for residual connections
        if getattr(module, "_residual_out", False):
            base_scale *= self.depth_scale
        
        # Apply loop scaling for recurrent core
        if "recurrent_core" in name or "core" in name:
            base_scale *= self.loop_scale
        
        return base_scale
    
    def get_lr_scale(
        self, 
        param: torch.Tensor, 
        param_name: str,
        base_lr: float,
    ) -> float:
        """
        Get the learning rate scale for a specific parameter.
        
        The mUP paper shows that LR should be scaled by sqrt(fan_in_0 / fan_in)
        where fan_in_0 is the fan_in of the reference architecture.
        
        Args:
            param: Parameter tensor
            param_name: Parameter name for special handling
            base_lr: Base learning rate
            
        Returns:
            Scaled learning rate
        """
        if param.ndim < 2:
            # Non-matrix parameters (biases, norms) get full LR
            return base_lr
        
        fan_in = param.shape[1] if param.ndim >= 2 else param.shape[0]
        fan_in_0 = self.base_hidden
        
        # LR scale: sqrt(fan_in_0 / fan_in)
        lr_scale = math.sqrt(fan_in_0 / max(fan_in, 1))
        
        # Additional scaling for recurrent parameters
        if "recurrent_core" in param_name or "core" in param_name:
            lr_scale *= self.loop_scale
        
        return base_lr * lr_scale * self.lr_scale_factor


class MuPParameterGroups:
    """
    Parameter groups with mUP scaling for optimizer configuration.
    
    Creates parameter groups with appropriate learning rate scaling
    based on mUP principles.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        mup_config: Optional[RecurrentMuP] = None,
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.base_lr = base_lr
        self.mup_config = mup_config
        self.weight_decay = weight_decay
        
        self.param_groups = self._create_groups()
    
    def _create_groups(self) -> Dict[str, list]:
        """Create parameter groups with mUP scaling."""
        groups = {
            "muon_2d": [],      # 2D matrices for Muon optimizer
            "adamw_1d": [],     # 1D tensors for AdamW (biases, norms)
            "no_decay": [],     # Parameters without weight decay
            "decay": [],        # Parameters with weight decay
        }
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            lr = self.base_lr
            wd = self.weight_decay
            
            if self.mup_config is not None:
                lr = self.mup_config.get_lr_scale(param, name, self.base_lr)
            
            group_entry = {
                "params": [param],
                "lr": lr,
                "weight_decay": wd if param.ndim >= 2 and not name.endswith(".bias") else 0.0,
                "name": name,
            }
            
            # Categorize for Muon vs AdamW
            if param.ndim >= 2 and not name.endswith(".bias"):
                groups["muon_2d"].append(group_entry)
            else:
                groups["adamw_1d"].append(group_entry)
            
            # Categorize for decay/no-decay
            if param.ndim >= 2 and not name.endswith(".bias"):
                groups["decay"].append(group_entry)
            else:
                groups["no_decay"].append(group_entry)
        
        return groups
    
    def get_muon_params(self) -> list:
        """Get parameters for Muon optimizer (2D matrices)."""
        return self.param_groups["muon_2d"]
    
    def get_adamw_params(self) -> list:
        """Get parameters for AdamW optimizer (1D tensors and biases)."""
        return self.param_groups["adamw_1d"]
    
    def get_no_decay_params(self) -> list:
        """Get parameters without weight decay."""
        return self.param_groups["no_decay"]


def apply_mup_init(
    model: nn.Module,
    base_std: float = 0.02,
    mup_config: Optional[RecurrentMuP] = None,
) -> nn.Module:
    """
    Apply mUP initialization to a model.
    
    Modifies the model's parameter initialization to use mUP scaling,
    which adjusts initialization std based on model width and depth.
    
    Args:
        model: The model to initialize
        base_std: Base standard deviation for initialization
        mup_config: Optional mUP configuration
        
    Returns:
        The model with mUP initialization applied
    """
    if mup_config is None:
        mup_config = RecurrentMuP(
            hidden_size=model.config.hidden_size if hasattr(model, 'config') else 560,
            base_hidden=1024,
            base_loops=16,
        )
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            scale = mup_config.get_init_scale(module, name)
            std = base_std * scale
            
            with torch.no_grad():
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                else:
                    module.weight.data.normal_(mean=0.0, std=std)
                
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    return model


def create_mup_config(
    config,
    base_hidden: int = 1024,
    base_loops: int = 16,
    lr_scale_factor: float = 1.0,
) -> RecurrentMuP:
    """
    Factory function to create mUP configuration from model config.
    
    Args:
        config: LoopedLLMConfig containing model configuration
        base_hidden: Reference hidden size for scaling
        base_loops: Reference loop count for scaling
        lr_scale_factor: Additional LR scaling factor
        
    Returns:
        RecurrentMuP configuration instance
    """
    return RecurrentMuP(
        hidden_size=config.hidden_size,
        base_hidden=base_hidden,
        base_loops=base_loops,
        num_recurrent_layers=config.num_recurrent_layers,
        max_recurrent_loops=config.max_recurrent_loops,
        lr_scale_factor=lr_scale_factor,
    )


def compute_mup_scaling(
    hidden_size: int,
    num_layers: int,
    base_hidden: int = 1024,
    base_layers: int = 16,
) -> Tuple[float, float]:
    """
    Compute mUP scaling factors for width and depth.
    
    Args:
        hidden_size: Model hidden dimension
        num_layers: Number of layers
        base_hidden: Reference hidden size
        base_layers: Reference layer count
        
    Returns:
        Tuple of (width_scale, depth_scale)
    """
    width_scale = math.sqrt(base_hidden / hidden_size)
    
    # Depth scale based on layer ratio
    depth_scale = math.sqrt(base_layers / max(num_layers, 1))
    
    return width_scale, depth_scale

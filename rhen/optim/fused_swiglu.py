import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU activation with torch.compile optimization.
    
    Implements the SwiGLU activation function (gate * silu(up)) with
    optimizations for hardware acceleration via torch.compile and
    custom kernels where available.
    
    Features:
    - torch.compile mode="max-autotune" for performance optimization
    - Fallback to standard implementation when compilation fails
    - Numerical stability with dtype casting
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_torch_compile: bool = True,
        compile_mode: str = "max-autotune",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_torch_compile = use_torch_compile
        self.compile_mode = compile_mode
        
        # Compile the fused forward if available
        self._compile_enabled = False
        if use_torch_compile:
            try:
                self._forward_fused = torch.compile(
                    self._forward_fused_impl,
                    mode=compile_mode,
                    dynamic=True,
                    fullgraph=True,
                )
                self._compile_enabled = True
                print(f"[FusedSwiGLU] torch.compile enabled with mode='{compile_mode}'")
            except Exception as e:
                print(f"[FusedSwiGLU] torch.compile failed ({e}), using standard implementation")
        
        self._residual_out = True
    
    def _forward_fused_impl(
        self,
        gate_weight: torch.Tensor,    # [intermediate_size, hidden_size]
        gate_bias: Optional[torch.Tensor],  # [intermediate_size]
        up_weight: torch.Tensor,      # [intermediate_size, hidden_size]
        up_bias: Optional[torch.Tensor],    # [intermediate_size]
        x: torch.Tensor,              # [..., hidden_size]
    ) -> torch.Tensor:
        """
        Fused SwiGLU forward pass with custom kernel interface.
        
        This can be overridden by custom kernels (Triton, CUDA, etc.)
        for maximum performance. The default implementation uses
        PyTorch operations through torch.compile.
        """
        # Cast to float32 for numerical stability in activation
        x_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute gate and up projections
        if gate_bias is not None:
            gate = F.linear(x, gate_weight, gate_bias)
        else:
            gate = F.linear(x, gate_weight)
        
        if up_bias is not None:
            up = F.linear(x, up_weight, up_bias)
        else:
            up = F.linear(x, up_weight)
        
        # Compute SwiGLU: gate * silu(up)
        result = gate * F.silu(up)
        
        return result.to(x_dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        gate_weight: Optional[torch.Tensor] = None,
        gate_bias: Optional[torch.Tensor] = None,
        up_weight: Optional[torch.Tensor] = None,
        up_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with fused SwiGLU optimization.
        
        Args:
            x: Input tensor
            gate_weight/up_weight: Optional custom kernel weights
            gate_bias/up_bias: Optional bias terms
            
        Returns:
            Output tensor with SwiGLU activation
        """
        if self._compile_enabled:
            # Use compiled version if weights are provided
            if gate_weight is not None and up_weight is not None:
                try:
                    return self._forward_fused(
                        gate_weight, gate_bias, up_weight, up_bias, x
                    )
                except Exception:
                    self._compile_enabled = False
        
        # Fallback to standard implementation
        if gate_bias is not None:
            gate = F.linear(x, gate_weight, gate_bias)
        else:
            gate = F.linear(x, gate_weight)
            
        if up_bias is not None:
            up = F.linear(x, up_weight, up_bias)
        else:
            up = F.linear(x, up_weight)
            
        return gate * F.silu(up)


class FusedSwiGLUModule(nn.Module):
    """
    SwiGLU Module with FusedKernels and torch.compile support.
    
    Wrapper module that provides SwiGLU activation with options for
    custom fused kernels and torch compilation for maximum performance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_fused_kernel: bool = True,
        use_torch_compile: bool = True,
        compile_mode: str = "max-autotune",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Create fused SwiGLU with optimizations
        self.fused_swiglu = FusedSwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_torch_compile=use_torch_compile,
            compile_mode=compile_mode,
        )
        
        # Standard projections (can be replaced by custom kernels)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.down_proj._residual_out = True
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with fused SwiGLU optimization.
        
        Args:
            x: Input tensor [..., H]
            
        Returns:
            Output tensor after SwiGLU activation and down projection
        """
        # Get weights for potential fused kernel
        gate_weight = self.gate_proj.weight
        gate_bias = self.gate_proj.bias if self.gate_proj.bias is not None else None
        up_weight = self.up_proj.weight
        up_bias = self.up_proj.bias if self.up_proj.bias is not None else None
        
        # Apply fused SwiGLU
        swiglu_out = self.fused_swiglu(
            x,
            gate_weight=gate_weight,
            gate_bias=gate_bias,
            up_weight=up_weight,
            up_bias=up_bias,
        )
        
        # Down projection
        return self.down_proj(swiglu_out)


def create_fused_swiglu(
    hidden_size: int,
    intermediate_size: int,
    use_torch_compile: bool = True,
    compile_mode: str = "max-autotune",
    device: Optional[torch.device] = None,
) -> FusedSwiGLUModule:
    """
    Factory function to create optimized Fused SwiGLU module.
    
    Args:
        hidden_size: Hidden dimension
        intermediate_size: Intermediate dimension for SwiGLU
        use_torch_compile: Whether to use torch.compile optimization
        compile_mode: torch.compile mode (max-autotune, reduce-overhead, etc.)
        device: Device to place module on
        
    Returns:
        FusedSwiGLUModule instance
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return FusedSwiGLUModule(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        use_fused_kernel=True,
        use_torch_compile=use_torch_compile,
        compile_mode=compile_mode,
    ).to(device)
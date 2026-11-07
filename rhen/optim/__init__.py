import logging

import torch
from typing import Any
from optim.grouped_gemm import GroupedGEMMWrapper, create_grouped_gemm_moe
from optim.quantized_kv import QuantizedMultiLatentAttention, create_quantized_mla
from optim.checkpoint import CheckpointedRecurrentCore
from optim.mup import RecurrentMuP, create_mup_config, apply_mup_init
from optim.fused_swiglu import FusedSwiGLUModule, create_fused_swiglu

logger = logging.getLogger(__name__)


class AdvancedOptimizations:
    """
    Container class for advanced optimizations to the EOS recurrent model.
    
    Manages all optimization modules including:
    - Grouped GEMM for MoE
    - Quantized KV cache
    - Recurrent-core gradient checkpointing
    - Model Upstream Parallelization (mUP)
    - Fused SwiGLU (eager; torch.compile disabled)
    
    Provides factory methods to create each optimization component
    and integrates them into the model architecture.
    """
    
    def __init__(self, config):
        self.config = config
        # Fused recurrent Triton kernels were removed with optim/kernels.py; always None.
        self.fused_kernels = None
        self.grouped_gemm_moe = None
        self.quantized_mla = None
        self.checkpointed_core = False  # True once the recurrent core is checkpoint-wrapped
        self.mup_config = None
        self.fused_swiglu = None
        
        # Initialize based on config flags
        self._init_optimizations()
    
    def _init_optimizations(self):
        """Initialize all optimization modules based on config."""
        dtype = torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Fused recurrent kernels: removed with optim/kernels.py; fused_kernels stays None.
        
        # 2. Grouped GEMM MoE
        if self.config.enable_grouped_gemm:
            self.grouped_gemm_moe = create_grouped_gemm_moe(
                config=self.config,
                dtype=dtype,
                device=device,
            )
            logger.info("Initialized GroupedGEMM MoE")
        
        # 3. Quantized KV Cache (for inference)
        if self.config.quantized_kv_cache:
            self.quantized_mla = create_quantized_mla(
                config=self.config,
                dtype=dtype,
                device=device,
            )
            logger.info("Initialized QuantizedMultiLatentAttention")
        
        # 4. Model Upstream Parallelization (mUP)
        if self.config.mup_enabled:
            self.mup_config = create_mup_config(
                config=self.config,
                base_hidden=self.config.mup_base_hidden,
                base_loops=self.config.mup_base_loops,
            )
            logger.info("Initialized mUP scaling")
        
        # 5. Fused SwiGLU (eager; torch.compile disabled). create_fused_swiglu
        #    requires intermediate_size.
        self.fused_swiglu = create_fused_swiglu(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            use_torch_compile=False,
            device=device,
        )
        logger.info("Initialized Fused SwiGLU")
    
    def apply_to_model(self, model):
        """
        Apply optimizations to a model instance.
        
        Args:
            model: EOSLoopedLLM model instance
            
        Returns:
            Model with optimizations applied
        """
        self.model = model
        
        # Apply mUP initialization if enabled
        if self.mup_config is not None:
            model = apply_mup_init(model, mup_config=self.mup_config)
        
        # Replace standard components with optimized versions
        self._replace_components(model)
        
        return model
    
    def _replace_components(self, model):
        """Replace standard components with optimized versions."""
        
        # 1. Recurrent-core gradient checkpointing. CheckpointedRecurrentCore reads
        #    config.gradient_checkpointing itself; skip if the core is already wrapped.
        if self.config.enable_selective_checkpoint and self.config.gradient_checkpointing and hasattr(model, "recurrent_core"):
            if not isinstance(model.recurrent_core, CheckpointedRecurrentCore):
                model.recurrent_core = CheckpointedRecurrentCore(model.recurrent_core, config=self.config)
            self.checkpointed_core = True


def apply_advanced_optimizations(
    model: Any,
    config=None,
) -> Any:
    """
    Apply advanced optimizations to EOS model.
    
    High-level function to apply all supported optimizations based on
    model configuration. Handles optimization initialization and
    component replacement.
    
    Args:
        model: EOS model instance to optimize
        config: Optional config override (uses model.config if not provided)
        
    Returns:
        Optimized model instance
    """
    if config is None:
        config = model.config
    
    # Initialize optimizations manager
    optim = AdvancedOptimizations(config)
    
    # Apply optimizations
    optimized_model = optim.apply_to_model(model)
    
    logger.info(
        "Applied optimizations: FusedKernels=%s, GroupedGEMM=%s, QuantizedKV=%s, "
        "Checkpoint=%s, mUP=%s",
        optim.fused_kernels is not None,
        optim.grouped_gemm_moe is not None,
        optim.quantized_mla is not None,
        optim.checkpointed_core,
        optim.mup_config is not None,
    )
    
    return optimized_model


def create_mup_parameter_groups(model, config, base_lr=1e-4):
    """
    Create parameter groups with mUP scaling for optimizer.
    
    Utility function to create optimizer parameter groups using mUP scaling.
    
    Args:
        model: Model to create groups for
        config: Model configuration
        base_lr: Base learning rate
        
    Returns:
        List of parameter groups for optimizer
    """
    if config.mup_enabled:
        from optimizers import build_muon_adamw
        # Per-group mUP LR scaling is not applied here; only base_lr is used.
        muon_opt, adamw_opt, _ = build_muon_adamw(
            model,
            adamw_lr=base_lr,
        )
        
        return [muon_opt, adamw_opt]
    
    # Fall back to standard parameter groups
    from optimizers import build_muon_adamw
    muon_opt, adamw_opt, _ = build_muon_adamw(model, adamw_lr=base_lr)
    return [muon_opt, adamw_opt]


# Export key classes and functions for use
__all__ = [
    "AdvancedOptimizations",
    "apply_advanced_optimizations",
    "create_mup_parameter_groups",
    "RecurrentMuP",
    "create_mup_config",
    "apply_mup_init",
    "GroupedGEMMWrapper",
    "create_grouped_gemm_moe",
    "QuantizedMultiLatentAttention",
    "create_quantized_mla",
    "CheckpointedRecurrentCore",
    "FusedSwiGLUModule",
    "create_fused_swiglu",
]

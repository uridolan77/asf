"""
Configuration classes for CL-PEFT adapters.

This module provides configuration classes for CL-PEFT adapters.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

from .enums import CLStrategy, QuantizationMode

class CLPEFTAdapterConfig(BaseModel):
    """Configuration for CL-PEFT adapter."""
    adapter_id: str = Field(..., description="Unique identifier for the adapter.")
    adapter_name: str = Field(..., description="Human-readable name for the adapter.")
    base_model_name: str = Field(..., description="Base model identifier.")
    description: Optional[str] = Field(None, description="Description of the adapter.")
    
    # CL configuration
    cl_strategy: CLStrategy = Field(CLStrategy.NAIVE, description="Continual Learning strategy.")
    replay_buffer_size: int = Field(1000, description="Size of the replay buffer for replay-based strategies.")
    ewc_lambda: float = Field(5000.0, description="EWC regularization strength.")
    
    # PEFT configuration
    peft_method: str = Field("lora", description="PEFT method (lora, prefix_tuning, etc.).")
    lora_r: int = Field(16, description="LoRA attention dimension.")
    lora_alpha: int = Field(32, description="Alpha parameter for LoRA scaling.")
    lora_dropout: float = Field(0.05, description="Dropout probability for LoRA layers.")
    
    # Target modules to apply LoRA to
    target_modules: List[str] = Field(
        ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="List of modules to apply LoRA to."
    )
    
    # Quantization configuration
    quantization_mode: QuantizationMode = Field(
        QuantizationMode.NONE, 
        description="Quantization mode for the base model."
    )
    
    # Task configuration
    task_type: str = Field("causal_lm", description="Task type (causal_lm, seq_cls, etc.).")
    
    # Additional configuration
    bias: str = Field("none", description="Bias type for LoRA.")
    fan_in_fan_out: bool = Field(False, description="Fan in/out for LoRA.")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for the adapter.")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True

# CL-PEFT: Continual Learning with Parameter-Efficient Fine-Tuning

This module implements comprehensive continual learning strategies for parameter-efficient fine-tuning (PEFT) of large language models (LLMs).

## Overview

CL-PEFT combines continual learning techniques with parameter-efficient fine-tuning methods like LoRA to enable sequential adaptation of LLMs to multiple tasks while mitigating catastrophic forgetting.

## Key Components

### Core Strategies

1. **EWC (Elastic Weight Consolidation)**
   - `ewc_base.py`: Base implementation of EWC
   - `ewc_online.py`: Memory-efficient Online EWC variant
   - `ewc_synaptic.py`: Synaptic Intelligence variant

2. **Experience Replay**
   - `replay_base.py`: Base replay strategy
   - `replay_experience.py`: Experience replay with memory buffer
   - `replay_prioritized.py`: Prioritized experience replay
   - `replay_utils.py`: Utilities for replay strategies
   - `replay_quality.py`: Quality control for replay examples
   - `replay_generative.py`: Generative replay using the model itself

3. **Orthogonal Strategies**
   - `orthogonal_base.py`: Orthogonal LoRA implementation
   - `adaptive_svd.py`: Adaptive SVD for identifying important parameter subspaces

4. **Mask-Based Strategies**
   - `mask_based.py`: Base mask-based continual learning
   - `mask_based_enhanced.py`: Enhanced mask-based CL with visualization

### Optimizations

- `optimizations.py`: Memory and computation optimizations including:
  - Gradient checkpointing
  - 8-bit optimizers
  - Distributed training support
  - Quantization methods
  - DeepSpeed and FSDP integration

## Strategy Details

### EWC Strategies

EWC mitigates catastrophic forgetting by adding a regularization term to the loss that penalizes changes to parameters that were important for previous tasks.

- **Base EWC**: Original implementation from Kirkpatrick et al.
- **Online EWC**: Memory-efficient version that maintains a single Fisher matrix
- **Synaptic Intelligence**: Similar to EWC but with path integral of gradients

### Replay Strategies

Replay strategies mitigate catastrophic forgetting by replaying examples from previous tasks during training on new tasks.

- **Experience Replay**: Stores and replays examples from previous tasks
- **Prioritized Experience Replay**: Prioritizes examples based on their importance
- **Generative Replay**: Uses the model to generate synthetic examples from previous tasks

### Orthogonal Strategies

Orthogonal strategies enforce orthogonality between parameter updates for different tasks to reduce interference.

- **Orthogonal LoRA**: Enforces orthogonality between LoRA matrices for different tasks
- **Adaptive SVD**: Projects gradient updates onto low-rank subspaces orthogonal to directions important for previous tasks

### Mask-Based Strategies

Mask-based strategies learn binary masks to activate specific parameters for each task, preventing interference between tasks.

- **MaskBasedCL**: Basic implementation of mask-based continual learning
- **EnhancedMaskBasedCL**: Extended implementation with improved initialization, visualization, and analysis

## Usage Examples

### Basic Usage with EWC

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from asf.medical.ml.cl_peft.strategies.ewc_base import BaseEWC

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# Create EWC strategy
ewc_strategy = BaseEWC(
    model=model,
    ewc_lambda=5000.0,
    fisher_sample_size=200
)

# Train on first task
ewc_strategy.before_training(task_id="task1")
# ... train model on task1 ...
ewc_strategy.after_training(task_id="task1", train_dataset=task1_dataset)

# Train on second task with EWC regularization
ewc_strategy.before_training(task_id="task2")
# ... train model on task2 with ewc_strategy.modify_loss() ...
ewc_strategy.after_training(task_id="task2", train_dataset=task2_dataset)
```

### Using Generative Replay

```python
from asf.medical.ml.cl_peft.strategies.replay_generative import GenerativeReplay

# Create Generative Replay strategy
gen_replay = GenerativeReplay(
    model=model,
    tokenizer=tokenizer,
    task_prompts={
        "task1": "Generate an example for task 1: ",
        "task2": "Generate an example for task 2: "
    },
    examples_per_task=100,
    replay_ratio=0.3
)

# Train on first task
gen_replay.before_training(task_id="task1")
# ... train model on task1 ...
gen_replay.after_training(task_id="task1")

# Train on second task with generated examples from first task
gen_replay.before_training(task_id="task2", train_dataset=task2_dataset)
# ... train model on task2 with mixed dataset ...
gen_replay.after_training(task_id="task2")
```

### Using Memory Optimizations

```python
from asf.medical.ml.cl_peft.optimizations import (
    enable_gradient_checkpointing,
    get_8bit_optimizer,
    get_memory_efficient_trainer
)

# Enable gradient checkpointing
model = enable_gradient_checkpointing(model)

# Get 8-bit optimizer
optimizer = get_8bit_optimizer(model, lr=5e-5)

# Create memory-efficient trainer
trainer = get_memory_efficient_trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    use_8bit_optimizer=True,
    use_gradient_checkpointing=True
)
```

## References

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.
2. Schwarz, J., et al. (2018). Progress & compress: A scalable framework for continual learning. ICML.
3. Zenke, F., et al. (2017). Continual learning through synaptic intelligence. ICML.
4. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
5. Serra, J., et al. (2018). Overcoming catastrophic forgetting with hard attention to the task. ICML.
6. Shin, H., et al. (2017). Continual learning with deep generative replay. NeurIPS.

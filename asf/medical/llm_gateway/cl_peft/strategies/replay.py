"""
Replay-based strategies for CL-PEFT.

This module implements replay-based strategies for mitigating catastrophic forgetting
in sequential fine-tuning of LLMs with PEFT, including:
- Experience Replay: Stores and replays examples from previous tasks
- Generative Replay: Uses the model to generate synthetic examples from previous tasks
"""

import torch
import random
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
import numpy as np

from peft import PeftModel
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, ConcatDataset, Subset

from asf.medical.core.logging_config import get_logger
from .base import CLStrategy

logger = get_logger(__name__)

class ExperienceReplay(CLStrategy):
    """
    Experience Replay strategy for CL-PEFT.

    This strategy maintains a buffer of examples from previous tasks and
    mixes them with current task examples during training.
    """

    def __init__(
        self,
        model: PeftModel,
        buffer_size: int = 1000,
        replay_ratio: float = 0.3,
        **kwargs
    ):
        """
        Initialize the Experience Replay strategy.

        Args:
            model: The PEFT model to apply the strategy to
            buffer_size: Maximum number of examples to store in the replay buffer
            replay_ratio: Ratio of replay examples to current task examples
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio

        # Replay buffer: task_id -> list of examples
        self.replay_buffer = {}

        # Current dataset being used for training
        self.current_dataset = None

    def before_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Prepare for training on a new task.

        This method creates a mixed dataset with examples from the current task
        and the replay buffer.

        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset for the current task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id

        if train_dataset is None:
            logger.warning("No training dataset provided for Experience Replay")
            return

        # Store the original dataset
        self.current_dataset = train_dataset

        # If there are no previous tasks, no replay needed
        if not self.replay_buffer:
            logger.info(f"No previous tasks in replay buffer for task {task_id}")
            return

        logger.info(f"Preparing mixed dataset with replay for task {task_id}")

        # Create a mixed dataset with examples from the replay buffer
        kwargs['train_dataset'] = self._create_mixed_dataset(train_dataset)

    def _create_mixed_dataset(self, current_dataset):
        """
        Create a mixed dataset with examples from the current task and replay buffer.

        Args:
            current_dataset: Dataset for the current task

        Returns:
            Mixed dataset
        """
        # Flatten the replay buffer into a single list of examples
        replay_examples = []
        for task_examples in self.replay_buffer.values():
            replay_examples.extend(task_examples)

        # If replay buffer is empty, return the current dataset
        if not replay_examples:
            return current_dataset

        # Determine how many replay examples to use
        num_replay = int(len(current_dataset) * self.replay_ratio)
        num_replay = min(num_replay, len(replay_examples))

        # Sample replay examples
        sampled_replay = random.sample(replay_examples, num_replay)

        # Create a custom mixed dataset
        return MixedDataset(current_dataset, sampled_replay)

    def modify_loss(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Modify the loss function.

        For Experience Replay, the loss is already modified by using a mixed dataset,
        so no additional modification is needed here.

        Args:
            loss: Original loss value
            **kwargs: Additional arguments

        Returns:
            Modified loss value (same as original for Experience Replay)
        """
        # No loss modification needed for Experience Replay
        return loss

    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.

        This method updates the replay buffer with examples from the current task.

        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        if self.current_dataset is None:
            logger.warning(f"No dataset available to update replay buffer for task {task_id}")
            return

        logger.info(f"Updating replay buffer with examples from task {task_id}")

        # Sample examples from the current dataset for the replay buffer
        examples_per_task = self.buffer_size // max(1, len(self.replay_buffer) + 1)

        # Sample examples from the current dataset
        indices = random.sample(
            range(len(self.current_dataset)),
            min(examples_per_task, len(self.current_dataset))
        )

        # Store examples in the replay buffer
        self.replay_buffer[task_id] = [self.current_dataset[i] for i in indices]

        # If the buffer is too large, reduce the number of examples per task
        if sum(len(examples) for examples in self.replay_buffer.values()) > self.buffer_size:
            self._resize_buffer()

        logger.info(f"Replay buffer updated: {sum(len(examples) for examples in self.replay_buffer.values())} examples total")

    def _resize_buffer(self):
        """
        Resize the replay buffer to stay within the buffer size limit.
        """
        # Calculate new examples per task
        examples_per_task = self.buffer_size // len(self.replay_buffer)

        # Resize each task's examples
        for task_id in self.replay_buffer:
            if len(self.replay_buffer[task_id]) > examples_per_task:
                self.replay_buffer[task_id] = random.sample(
                    self.replay_buffer[task_id],
                    examples_per_task
                )

    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.

        For Experience Replay, this is a no-op as the dataset mixing handles the replay.

        Args:
            **kwargs: Additional arguments
        """
        # No gradient modification needed for Experience Replay
        pass

class GenerativeReplay(CLStrategy):
    """
    Generative Replay strategy for CL-PEFT.

    This strategy uses the model itself to generate synthetic examples from previous tasks
    and mixes them with current task examples during training.
    """

    def __init__(
        self,
        model: PeftModel,
        task_prompts: Dict[str, str],
        examples_per_task: int = 100,
        replay_ratio: float = 0.3,
        replay_frequency: int = 10,
        **kwargs
    ):
        """
        Initialize the Generative Replay strategy.

        Args:
            model: The PEFT model to apply the strategy to
            task_prompts: Dictionary mapping task IDs to prompts for generating examples
            examples_per_task: Number of examples to generate per previous task
            replay_ratio: Ratio of replay examples to current task examples
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.task_prompts = task_prompts
        self.examples_per_task = examples_per_task
        self.replay_ratio = replay_ratio
        self.replay_frequency = replay_frequency

        # Generated examples: task_id -> list of examples
        self.generated_examples = {}

        # Current dataset being used for training
        self.current_dataset = None

        # Tokenizer for generating examples
        self.tokenizer = kwargs.get('tokenizer', None)

        # Batch counter for replay frequency
        self.batch_counter = 0

    def before_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Prepare for training on a new task.

        This method generates synthetic examples from previous tasks and
        creates a mixed dataset with examples from the current task.

        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset for the current task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id

        if train_dataset is None:
            logger.warning("No training dataset provided for Generative Replay")
            return

        # Store the original dataset
        self.current_dataset = train_dataset

        # If there are no previous tasks, no replay needed
        if not self.task_history:
            logger.info(f"No previous tasks for generative replay for task {task_id}")
            return

        # Check if tokenizer is available
        if self.tokenizer is None and 'tokenizer' in kwargs:
            self.tokenizer = kwargs['tokenizer']

        if self.tokenizer is None:
            logger.warning("No tokenizer provided for Generative Replay, cannot generate examples")
            return

        logger.info(f"Generating synthetic examples for previous tasks for task {task_id}")

        # Generate synthetic examples for previous tasks
        self._generate_examples()

        # Create a mixed dataset with synthetic examples
        kwargs['train_dataset'] = self._create_mixed_dataset(train_dataset)

    def _generate_examples(self):
        """
        Generate synthetic examples for previous tasks.
        """
        # Skip the current task
        previous_tasks = [task['task_id'] for task in self.task_history
                         if task['task_id'] != self.current_task_id]

        for task_id in previous_tasks:
            # Skip if no prompt is available for this task
            if task_id not in self.task_prompts:
                logger.warning(f"No prompt available for task {task_id}, skipping generation")
                continue

            logger.info(f"Generating {self.examples_per_task} examples for task {task_id}")

            # Get the prompt for this task
            prompt = self.task_prompts[task_id]

            # Generate examples
            examples = []
            for _ in range(self.examples_per_task):
                # Generate text
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Create an example (format depends on the dataset structure)
                # This is a simplified example, actual implementation would depend on the dataset format
                example = {
                    "text": generated_text,
                    "task_id": task_id,
                    "is_generated": True
                }

                examples.append(example)

            # Store the generated examples
            self.generated_examples[task_id] = examples

    def _create_mixed_dataset(self, current_dataset):
        """
        Create a mixed dataset with examples from the current task and generated examples.

        Args:
            current_dataset: Dataset for the current task

        Returns:
            Mixed dataset
        """
        # Flatten the generated examples into a single list
        generated_examples = []
        for task_examples in self.generated_examples.values():
            generated_examples.extend(task_examples)

        # If no generated examples, return the current dataset
        if not generated_examples:
            return current_dataset

        # Determine how many generated examples to use
        num_generated = int(len(current_dataset) * self.replay_ratio)
        num_generated = min(num_generated, len(generated_examples))

        # Sample generated examples
        sampled_generated = random.sample(generated_examples, num_generated)

        # Create a custom mixed dataset
        return MixedDataset(current_dataset, sampled_generated)

    def modify_loss(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Modify the loss function.

        For Generative Replay, the loss is already modified by using a mixed dataset,
        so no additional modification is needed here.

        Args:
            loss: Original loss value
            **kwargs: Additional arguments

        Returns:
            Modified loss value (same as original for Generative Replay)
        """
        # No loss modification needed for Generative Replay
        return loss

    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.

        For Generative Replay, this is mainly a bookkeeping operation.

        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        logger.info(f"Completed training on task {task_id} with Generative Replay")

    def on_batch_start(self, trainer, batch, **kwargs):
        """
        Called at the start of each batch during training.

        This method implements the replay mechanism by injecting generated
        examples into the current batch.

        Args:
            trainer: Trainer instance
            batch: Current batch
            **kwargs: Additional arguments

        Returns:
            Modified batch
        """
        # Increment batch counter
        self.batch_counter += 1

        # Check if replay should be performed
        if self.batch_counter % self.replay_frequency != 0 or not self.generated_examples:
            return batch

        # Get replay batch
        replay_batch = self.get_replay_batch(len(batch))

        # If no replay batch, return original batch
        if not replay_batch:
            return batch

        # Combine current batch with replay batch
        # Note: This requires careful handling based on the model's expectations
        if isinstance(batch, dict):
            # Dictionary batch (most common in HF)
            combined_batch = {}
            for key in batch.keys():
                if key in replay_batch[0]:
                    current_tensors = batch[key]

                    # For text inputs, we need to tokenize them first
                    if key == "text" or key == "input_ids":
                        # Tokenize text inputs if needed
                        if key == "text" and self.tokenizer:
                            replay_inputs = self.tokenizer(
                                [example[key] for example in replay_batch],
                                return_tensors="pt",
                                padding=True,
                                truncation=True
                            ).input_ids.to(current_tensors.device)
                        else:
                            # If already tokenized or no tokenizer
                            replay_tensors = [torch.tensor(example[key]) for example in replay_batch]
                            replay_inputs = torch.stack(replay_tensors).to(current_tensors.device)

                        # Combine with current batch
                        combined_batch[key] = torch.cat([current_tensors, replay_inputs], dim=0)
                    else:
                        # Handle other tensor types
                        if isinstance(current_tensors, torch.Tensor):
                            # Try to convert to tensors and stack
                            try:
                                replay_tensors = [torch.tensor(example[key]).to(current_tensors.device)
                                                 for example in replay_batch]
                                replay_tensors = torch.stack(replay_tensors)
                                combined_batch[key] = torch.cat([current_tensors, replay_tensors], dim=0)
                            except:
                                # Fallback if tensor conversion fails
                                logger.warning(f"Could not combine tensors for key {key}")
                                combined_batch[key] = current_tensors
                        else:
                            # For non-tensor data
                            combined_batch[key] = current_tensors + [example[key] for example in replay_batch]
                else:
                    # Key not in replay batch, keep original
                    combined_batch[key] = batch[key]

            return combined_batch
        else:
            # For non-dictionary batches, this needs to be customized based on the expected format
            logger.warning(f"Unsupported batch format for replay: {type(batch)}")
            return batch

    def get_replay_batch(self, batch_size):
        """
        Get a batch of examples for replay.

        Args:
            batch_size: Size of the current batch

        Returns:
            List of examples for replay
        """
        # Flatten the generated examples into a single list
        all_examples = []
        for task_examples in self.generated_examples.values():
            all_examples.extend(task_examples)

        if not all_examples:
            return []

        # Determine replay batch size
        replay_size = int(batch_size * self.replay_ratio)
        replay_size = min(replay_size, len(all_examples))

        if replay_size == 0:
            return []

        # Sample examples for replay
        return random.sample(all_examples, replay_size)

    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.

        For Generative Replay, this is a no-op as the replay is handled
        through batch modification.

        Args:
            **kwargs: Additional arguments
        """
        # No explicit gradient modification needed
        pass

class MixedDataset(Dataset):
    """
    Custom dataset that combines original examples with replay/generated examples.
    """

    def __init__(self, original_dataset, replay_examples):
        """
        Initialize the mixed dataset.

        Args:
            original_dataset: Dataset for the current task
            replay_examples: List of examples from replay buffer or generation
        """
        self.original_dataset = original_dataset
        self.replay_examples = replay_examples

        # Total length is the sum of original and replay examples
        self.length = len(original_dataset) + len(replay_examples)

    def __len__(self):
        """Get the total length of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Dataset item
        """
        # If index is within original dataset range, return from original
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]

        # Otherwise, return from replay examples
        replay_idx = idx - len(self.original_dataset)
        return self.replay_examples[replay_idx]

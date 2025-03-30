import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional
class ComplianceBalancedSampler:
    """Balanced sampling from multiple compliance datasets"""
    
    def __init__(self, datasets):
        self.datasets = datasets
        self.samplers = {name: iter(dataset) for name, dataset in datasets.items()}
        self.weights = self._calculate_sampling_weights()
        
    def _calculate_sampling_weights(self):
        """Calculate sampling weights for different datasets"""
        # Give higher weight to framework-specific and edge cases
        weights = {
            "general": 0.3,
            "framework_specific": 0.4,
            "edge_cases": 0.2,
            "adversarial": 0.1
        }
        
        # Normalize weights for actual datasets present
        total = sum(weights[k] for k in self.datasets.keys() if k in weights)
        return {k: weights.get(k, 0.1) / total for k in self.datasets.keys()}
    
    def get_batches(self, batch_size):
        """Get batches sampled from datasets according to weights"""
        # Determine number of samples from each dataset
        total_samples = 0
        dataset_samples = {}
        
        for name, weight in self.weights.items():
            dataset_samples[name] = max(1, int(batch_size * weight))
            total_samples += dataset_samples[name]
            
        # Adjust to match batch size exactly
        diff = batch_size - total_samples
        if diff != 0:
            # Distribute difference among datasets
            for name in sorted(self.weights, key=self.weights.get, reverse=True):
                dataset_samples[name] += 1
                diff -= 1
                if diff == 0:
                    break
        
        # Create infinite iterator
        while True:
            # Sample from each dataset
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            regulatory_constraints = {}
            
            for name, count in dataset_samples.items():
                for _ in range(count):
                    try:
                        sample = next(self.samplers[name])
                    except StopIteration:
                        # Reset sampler if exhausted
                        self.samplers[name] = iter(self.datasets[name])
                        sample = next(self.samplers[name])
                        
                    # Add to batch
                    for key in batch:
                        if key in sample:
                            batch[key].append(sample[key])
                            
                    # Collect regulatory constraints
                    if "regulatory_constraints" in sample:
                        for constraint, value in sample["regulatory_constraints"].items():
                            if constraint not in regulatory_constraints:
                                regulatory_constraints[constraint] = []
                            regulatory_constraints[constraint].append(value)
            
            # Combine batch elements
            for key in batch:
                batch[key] = torch.stack(batch[key]) if batch[key] else None
                
            # Add regulatory constraints
            batch["regulatory_constraints"] = regulatory_constraints
            
            yield batch


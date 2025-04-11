import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

class AdaptivePriorityManager:
    """
    Uses reinforcement learning to dynamically adjust task priorities.
    Optimizes semantic organization processing based on performance metrics and context.
    Implements Seth's principle of active inference for resource allocation.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # State-action value function
        self.experience_buffer = []  # For experience replay
        self.buffer_size = 1000
        self.batch_size = 32
        self.state_features = {}  # Cache of state features
        self.logger = logging.getLogger("ASF.Layer3.AdaptivePriorityManager")
        
        self.use_neural_model = False
        self.neural_model = None
        
        self.task_predictions = {}  # Task type -> predicted priorities
        self.priority_errors = defaultdict(list)  # Task type -> prediction errors
        self.priority_precision = {}  # Task type -> precision
        
    def initialize_neural_model(self, state_dim=8, action_dim=3):
        """Initialize neural network model for deep reinforcement learning."""
        self.use_neural_model = True
        
        # Simple feedforward network for Q-value prediction
        self.neural_model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5, verbose=True
        )
        
    async def get_priority(self, task_type, context):
        """
        Determine optimal priority for a task based on current state.
        
        Args:
            task_type: Type of task to prioritize
            context: Additional context about the task
            
        Returns:
            Priority value between 0 and 1
        state_tensor = torch.tensor(list(state.values()), 
                                   dtype=torch.float32).to(self.neural_model[0].weight.device)
        
        with torch.no_grad():
            q_values = self.neural_model(state_tensor)
            
        action_idx = torch.argmax(q_values).item()
        
        priority_values = [0.2, 0.5, 0.9]  # low, medium, high
        
        return priority_values[action_idx]
        
    async def predict_priority(self, task_type, future_context):
        future_key = self._extract_context_key(future_context)
        prediction_key = f"{task_type}_{future_key}"
        
        if prediction_key in self.task_predictions:
            return self.task_predictions[prediction_key]
            
        future_state = self._extract_state_features(task_type, future_context)
        
        precision = self.priority_precision.get(task_type, 1.0)
        
        if self.use_neural_model and self.neural_model is not None:
            state_tensor = torch.tensor(list(future_state.values()), 
                                      dtype=torch.float32).to(self.neural_model[0].weight.device)
            
            with torch.no_grad():
                q_values = self.neural_model(state_tensor)
                
                probs = F.softmax(q_values * precision, dim=0)
                
                priority_values = torch.tensor([0.2, 0.5, 0.9], 
                                             device=probs.device)
                predicted_priority = torch.sum(probs * priority_values)
                
                result = predicted_priority.item()
        else:
            state_key = self._discretize_state(future_state)
            
            if state_key in self.q_values:
                q_values = self.q_values[state_key]
                
                weighted_sum = 0.0
                total_weight = 0.0
                
                priority_map = {
                    'high': 0.9,
                    'medium': 0.5,
                    'low': 0.2
                }
                
                for action, q_value in q_values.items():
                    weight = np.exp(q_value * precision)
                    weighted_sum += priority_map[action] * weight
                    total_weight += weight
                    
                if total_weight > 0:
                    result = weighted_sum / total_weight
                else:
                    result = 0.5  # Default
            else:
                result = 0.5  # Default
                
        self.task_predictions[prediction_key] = result
        
        return result
        
    async def update_from_feedback(self, task_type, metrics):
        if task_type not in self.state_features:
            return
            
        reward = self._calculate_reward(metrics)
        
        state = self.state_features[task_type]
        
        if self.use_neural_model and self.neural_model is not None:
            await self._update_neural_model(state, 
                                          metrics.get('priority', 0.5), 
                                          reward)
            return
            
        state_key = self._discretize_state(state)
        
        priority = metrics.get('priority', 0.5)
        action = 'medium'
        
        if priority >= 0.7:
            action = 'high'
        elif priority <= 0.3:
            action = 'low'
            
        if state_key in self.q_values and action in self.q_values[state_key]:
            old_value = self.q_values[state_key][action]
            self.q_values[state_key][action] = old_value + self.learning_rate * reward
            
        experience = (state_key, action, reward)
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer = self.experience_buffer[-self.buffer_size:]
            
        if len(self.experience_buffer) >= self.batch_size:
            await self._replay_experiences()
            
        context_key = self._extract_context_key(metrics.get('context', {}))
        prediction_key = f"{task_type}_{context_key}"
        
        if prediction_key in self.task_predictions:
            predicted = self.task_predictions[prediction_key]
            actual = priority
            
            error = abs(predicted - actual)
            
            self.priority_errors[task_type].append(error)
            
            if len(self.priority_errors[task_type]) > 20:
                self.priority_errors[task_type] = self.priority_errors[task_type][-20:]
                
            if len(self.priority_errors[task_type]) > 1:
                variance = np.var(self.priority_errors[task_type])
                precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                self.priority_precision[task_type] = precision
            
    async def _update_neural_model(self, state, priority, reward):
        if len(self.experience_buffer) < self.batch_size:
            return
            
        batch_indices = np.random.choice(
            len(self.experience_buffer),
            self.batch_size,
            replace=False
        )
        
        for idx in batch_indices:
            state_key, action, reward = self.experience_buffer[idx]
            
            if state_key in self.q_values and action in self.q_values[state_key]:
                old_value = self.q_values[state_key][action]
                self.q_values[state_key][action] = old_value + self.learning_rate * (
                    reward - old_value
                )
                
    def _extract_state_features(self, task_type, context):
        """Extract relevant features for state representation."""
        features = {
            'task_type_id': hash(task_type) % 10,  # Hash task type to a small integer
            'queue_length': min(1.0, context.get('queue_length', 0) / 100),
            'system_load': context.get('system_load', 0.5),
            'time_of_day': datetime.datetime.now().hour / 24.0,
            'is_critical': 1.0 if context.get('is_critical', False) else 0.0,
            'expected_duration': min(1.0, context.get('expected_duration', 0.5) / 10.0),
            'resource_availability': context.get('resource_availability', 0.8),
            'priority_trend': context.get('priority_trend', 0.0)  # -1 to 1 range
        }
        
        return features
        
    def _extract_context_key(self, context):
        """Generate a string key for a context dict."""
        return "_".join(f"{k}:{v}" for k, v in sorted(context.items()))
        
    def _discretize_state(self, state):
        """Convert continuous state to discrete representation for lookup."""
        # Create a tuple of discretized features
        discrete_state = (
            state['task_type_id'],
            min(5, int(state['queue_length'] * 5)),  # Bucket queue length
            min(5, int(state['system_load'] * 5)),  # Bucket system load
            min(3, int(state['time_of_day'] * 4)),  # 4 time of day buckets
            1 if state['is_critical'] > 0.5 else 0,
            min(3, int(state['expected_duration'] * 3))  # 3 duration buckets
        )
        
        return str(discrete_state)  # Convert to string for dictionary key
        
    def _calculate_reward(self, metrics):
        """Calculate reward based on performance metrics."""
        processing_time = metrics.get('processing_time', 1.0)
        success = metrics.get('success', True)
        quality = metrics.get('quality', 0.5)
        resource_efficiency = metrics.get('resource_efficiency', 0.5)
        
        time_reward = max(0, 1.0 - min(1.0, processing_time / 10.0))
        success_reward = 1.0 if success else -0.5
        quality_reward = quality - 0.5  # -0.5 to 0.5
        efficiency_reward = resource_efficiency - 0.5  # -0.5 to 0.5
        
        reward = (
            time_reward * 0.3 +
            success_reward * 0.4 +
            quality_reward * 0.2 +
            efficiency_reward * 0.1
        )
        
        return reward

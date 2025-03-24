# Enhancement for async_queue.py - Add to existing class

import asyncio
import datetime
import time
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable

class AsyncProcessingQueue:
    # Existing initialization and methods...
    
    # Add these fields to __init__:
    # self.task_patterns = defaultdict(list)  # Task type -> past occurrences
    # self.task_transitions = defaultdict(lambda: defaultdict(int))  # Task A -> Task B -> count
    # self.context_task_frequencies = defaultdict(lambda: defaultdict(int))  # Context -> Task type -> count
    # self.prediction_errors = defaultdict(list)  # Task type -> prediction errors
    # self.precision_values = {}  # Task type -> precision
    
    async def predict_tasks(self, context):
        """
        Predict which tasks are likely to be needed in the near future.
        Implements Seth's predictive processing principle.
        
        Args:
            context: Current processing context
            
        Returns:
            Dict mapping task types to predicted priorities
        """
        # Initialize prediction fields if not present
        if not hasattr(self, 'task_patterns'):
            self.task_patterns = defaultdict(list)
        if not hasattr(self, 'task_transitions'):
            self.task_transitions = defaultdict(lambda: defaultdict(int))
        if not hasattr(self, 'context_task_frequencies'):
            self.context_task_frequencies = defaultdict(lambda: defaultdict(int))
            
        predicted_tasks = {}
        
        # Method 1: Use task transitions (which tasks typically follow others)
        recent_tasks = self._get_recent_task_types(5)  # Get 5 most recent task types
        
        for task_type in recent_tasks:
            if task_type in self.task_transitions:
                transitions = self.task_transitions[task_type]
                if transitions:
                    # Find most likely next tasks
                    total_count = sum(transitions.values())
                    for next_task, count in transitions.items():
                        probability = count / total_count
                        if probability > 0.3:  # Threshold for prediction
                            predicted_tasks[next_task] = probability
        
        # Method 2: Use context-based predictions
        context_key = self._get_context_key(context)
        if context_key in self.context_task_frequencies:
            frequencies = self.context_task_frequencies[context_key]
            total_count = sum(frequencies.values())
            if total_count > 0:
                for task_type, count in frequencies.items():
                    probability = count / total_count
                    if probability > 0.3:  # Threshold for prediction
                        predicted_tasks[task_type] = max(predicted_tasks.get(task_type, 0), probability)
        
        # Convert probabilities to priorities (0.5-0.9 range)
        predicted_priorities = {
            task_type: 0.5 + (0.4 * probability) 
            for task_type, probability in predicted_tasks.items()
        }
        
        return predicted_priorities
    
    async def preemptively_schedule(self, context):
        """
        Schedule predicted high-priority tasks before they're explicitly requested.
        Implements Seth's active inference principle.
        
        Args:
            context: Current processing context
            
        Returns:
            List of preemptively scheduled task IDs
        """
        # Initialize precision values if not present
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
            
        # Predict tasks
        predicted_tasks = await self.predict_tasks(context)
        
        # Filter to high-confidence predictions
        high_priority_predictions = {
            task_type: priority 
            for task_type, priority in predicted_tasks.items() 
            if priority > 0.7 and task_type in self.precision_values and self.precision_values[task_type] > 2.0
        }
        
        # No high-confidence predictions
        if not high_priority_predictions:
            return []
        
        # Schedule top predictions
        scheduled_tasks = []
        for task_type, priority in sorted(high_priority_predictions.items(), key=lambda x: -x[1]):
            # Create a preemptive task
            task_id = f"preemptive_{task_type}_{int(time.time()*1000)}"
            
            # In real implementation, we would need task factory functions to create actual tasks
            # This is a simplified placeholder
            scheduled_tasks.append(task_id)
            
            if hasattr(self, 'logger'):
                self.logger.info(f"Preemptively scheduled task {task_id} with priority {priority}")
        
        return scheduled_tasks
    
    async def submit_task(self, task_id, task_func, priority=0.5, dependencies=None, args=None, kwargs=None):
        """Submit a task with prediction evaluation."""
        # Initialize prediction fields if not present
        if not hasattr(self, 'prediction_errors'):
            self.prediction_errors = defaultdict(list)
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
        
        async with self.lock:
            # Existing task submission code...
            
            # Extract task type from kwargs
            args = args or []
            kwargs = kwargs or {}
            dependencies = dependencies or []
            task_type = kwargs.get('task_type', 'unknown')
            context = kwargs.get('context', {})
            
            # Check if this task was predicted
            predicted_priority = await self._check_predicted_task(task_type, context)
            if predicted_priority is not None:
                # Task was predicted, evaluate accuracy
                prediction_error = abs(predicted_priority - priority) / (1.0 + priority)
                self.prediction_errors[task_type].append(prediction_error)
                
                # Limit history size
                if len(self.prediction_errors[task_type]) > 20:
                    self.prediction_errors[task_type] = self.prediction_errors[task_type][-20:]
                
                # Update precision (inverse variance)
                if len(self.prediction_errors[task_type]) > 1:
                    variance = np.var(self.prediction_errors[task_type])
                    precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                    self.precision_values[task_type] = min(10.0, precision)  # Cap very high precision
                    
                # Use predicted priority if it's higher (preemptive processing)
                priority = max(priority, predicted_priority)
            
            # Record task pattern for future prediction
            await self._record_task_pattern(task_type, context)
            
            # Continue with existing task submission code...
            
            # IMPORTANT: Make sure to save this result and return it
            result = await super().submit_task(task_id, task_func, priority, dependencies, args, kwargs)
            return result
    
    async def _check_predicted_task(self, task_type, context):
        """Check if a task was predicted and return its predicted priority."""
        # Initialize if needed
        if not hasattr(self, 'task_patterns'):
            return None
            
        # Predict tasks for current context
        predicted_tasks = await self.predict_tasks(context)
        
        # Check if this task type was predicted
        if task_type in predicted_tasks:
            return predicted_tasks[task_type]
        
        return None
    
    async def _record_task_pattern(self, task_type, context):
        """Record task pattern for future prediction."""
        # Initialize if needed
        if not hasattr(self, 'task_patterns'):
            self.task_patterns = defaultdict(list)
        if not hasattr(self, 'task_transitions'):
            self.task_transitions = defaultdict(lambda: defaultdict(int))
        if not hasattr(self, 'context_task_frequencies'):
            self.context_task_frequencies = defaultdict(lambda: defaultdict(int))
            
        # Record task in history
        self.task_patterns[task_type].append({
            'timestamp': time.time(),
            'context': context
        })
        
        # Limit history size
        if len(self.task_patterns[task_type]) > 100:
            self.task_patterns[task_type] = self.task_patterns[task_type][-100:]
        
        # Record transition from previous task
        recent_tasks = self._get_recent_task_types(1)
        if recent_tasks:
            previous_task = recent_tasks[0]
            self.task_transitions[previous_task][task_type] += 1
        
        # Record context frequency
        context_key = self._get_context_key(context)
        self.context_task_frequencies[context_key][task_type] += 1
    
    def _get_recent_task_types(self, count=5):
        """Get most recent task types."""
        # Initialize if needed
        if not hasattr(self, 'task_patterns'):
            return []
            
        recent_tasks = []
        all_tasks = []
        
        # Collect all tasks with timestamps
        for task_type, occurrences in self.task_patterns.items():
            if occurrences:
                all_tasks.append((task_type, occurrences[-1]['timestamp']))
        
        # Sort by timestamp (descending)
        all_tasks.sort(key=lambda x: -x[1])
        
        # Take most recent
        return [task_type for task_type, _ in all_tasks[:count]]
    
    def _get_context_key(self, context):
        """Generate stable key for context."""
        # Simple implementation - in production would use better hashing
        if not context:
            return "default_context"
        
        # Sort context items for consistent key
        sorted_items = sorted((str(k), str(v)) for k, v in context.items())
        return "_".join(f"{k}:{v}" for k, v in sorted_items)
    
    async def get_prediction_stats(self):
        """Get statistics about task predictions."""
        # Initialize if needed
        if not hasattr(self, 'prediction_errors'):
            return {}
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
        if not hasattr(self, 'task_transitions'):
            self.task_transitions = defaultdict(lambda: defaultdict(int))
            
        stats = {}
        
        for task_type in self.prediction_errors:
            errors = self.prediction_errors[task_type]
            if errors:
                precision = self.precision_values.get(task_type, 1.0)
                stats[task_type] = {
                    'mean_error': sum(errors) / len(errors),
                    'error_count': len(errors),
                    'precision': precision,
                    'transitions': dict(self.task_transitions.get(task_type, {}))
                }
        
        return stats

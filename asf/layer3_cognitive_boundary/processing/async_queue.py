import asyncio
import datetime
import time
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable

class AsyncProcessingQueue:
    
    
    async def predict_tasks(self, context):
        if not hasattr(self, 'task_patterns'):
            self.task_patterns = defaultdict(list)
        if not hasattr(self, 'task_transitions'):
            self.task_transitions = defaultdict(lambda: defaultdict(int))
        if not hasattr(self, 'context_task_frequencies'):
            self.context_task_frequencies = defaultdict(lambda: defaultdict(int))
            
        predicted_tasks = {}
        
        recent_tasks = self._get_recent_task_types(5)  # Get 5 most recent task types
        
        for task_type in recent_tasks:
            if task_type in self.task_transitions:
                transitions = self.task_transitions[task_type]
                if transitions:
                    total_count = sum(transitions.values())
                    for next_task, count in transitions.items():
                        probability = count / total_count
                        if probability > 0.3:  # Threshold for prediction
                            predicted_tasks[next_task] = probability
        
        context_key = self._get_context_key(context)
        if context_key in self.context_task_frequencies:
            frequencies = self.context_task_frequencies[context_key]
            total_count = sum(frequencies.values())
            if total_count > 0:
                for task_type, count in frequencies.items():
                    probability = count / total_count
                    if probability > 0.3:  # Threshold for prediction
                        predicted_tasks[task_type] = max(predicted_tasks.get(task_type, 0), probability)
        
        predicted_priorities = {
            task_type: 0.5 + (0.4 * probability) 
            for task_type, probability in predicted_tasks.items()
        }
        
        return predicted_priorities
    
    async def preemptively_schedule(self, context):
        if not hasattr(self, 'precision_values'):
            self.precision_values = {}
            
        predicted_tasks = await self.predict_tasks(context)
        
        high_priority_predictions = {
            task_type: priority 
            for task_type, priority in predicted_tasks.items() 
            if priority > 0.7 and task_type in self.precision_values and self.precision_values[task_type] > 2.0
        }
        
        if not high_priority_predictions:
            return []
        
        scheduled_tasks = []
        for task_type, priority in sorted(high_priority_predictions.items(), key=lambda x: -x[1]):
            task_id = f"preemptive_{task_type}_{int(time.time()*1000)}"
            
            scheduled_tasks.append(task_id)
            
            if hasattr(self, 'logger'):
                self.logger.info(f"Preemptively scheduled task {task_id} with priority {priority}")
        
        return scheduled_tasks
    
    async def submit_task(self, task_id, task_func, priority=0.5, dependencies=None, args=None, kwargs=None):
        if not hasattr(self, 'task_patterns'):
            return None
            
        predicted_tasks = await self.predict_tasks(context)
        
        if task_type in predicted_tasks:
            return predicted_tasks[task_type]
        
        return None
    
    async def _record_task_pattern(self, task_type, context):
        if not hasattr(self, 'task_patterns'):
            return []
            
        recent_tasks = []
        all_tasks = []
        
        for task_type, occurrences in self.task_patterns.items():
            if occurrences:
                all_tasks.append((task_type, occurrences[-1]['timestamp']))
        
        all_tasks.sort(key=lambda x: -x[1])
        
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

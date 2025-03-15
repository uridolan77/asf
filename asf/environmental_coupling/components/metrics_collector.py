import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class PerformanceMetricsCollector:
    """
    Collects and aggregates performance metrics across Layer 4.
    Enhanced with predictive performance monitoring capabilities.
    """
    def __init__(self):
        self.operation_times = defaultdict(list)  # Maps operation_type to execution times
        self.operation_counts = defaultdict(int)  # Maps operation_type to count
        self.system_metrics = {
            'start_time': time.time(),
            'total_interactions': 0,
            'total_couplings': 0,
            'peak_interactions_per_second': 0
        }
        
        # Time series data
        self.time_series = {
            'interaction_counts': defaultdict(int),  # Maps timestamp (minute) to count
            'coupling_counts': defaultdict(int),  # Maps timestamp (minute) to count
            'avg_response_times': defaultdict(list)  # Maps timestamp (minute) to response times
        }
        
        # Seth's Data Paradox enhancements
        self.predicted_performance = {}  # Maps metric name to predicted values
        self.prediction_accuracy = defaultdict(list)  # Maps metric name to accuracy history
        
        self.logger = logging.getLogger("ASF.Layer4.PerformanceMetricsCollector")
        
    def record_operation(self, operation_type, execution_time):
        """Record an operation's execution time."""
        self.operation_times[operation_type].append(execution_time)
        self.operation_counts[operation_type] += 1
        
        # Limit history size
        if len(self.operation_times[operation_type]) > 1000:
            self.operation_times[operation_type] = self.operation_times[operation_type][-1000:]
            
        # Update time series data (rounded to the minute)
        timestamp_minute = int(time.time() / 60) * 60
        
        if operation_type == 'process_interaction':
            self.system_metrics['total_interactions'] += 1
            self.time_series['interaction_counts'][timestamp_minute] += 1
            self.time_series['avg_response_times'][timestamp_minute].append(execution_time)
            
            # Check for peak interactions per second
            current_minute_count = self.time_series['interaction_counts'][timestamp_minute]
            seconds_in_minute = min(60, int(time.time()) % 60 + 1)  # Add 1 to avoid division by zero
            current_rate = current_minute_count / seconds_in_minute
            
            if current_rate > self.system_metrics['peak_interactions_per_second']:
                self.system_metrics['peak_interactions_per_second'] = current_rate
                
        elif operation_type == 'establish_coupling':
            self.system_metrics['total_couplings'] += 1
            self.time_series['coupling_counts'][timestamp_minute] += 1
            
        # Predict future performance
        self._predict_future_performance()
        
    def get_metrics(self):
        """Get comprehensive performance metrics."""
        # Calculate average operation times
        avg_times = {}
        for op_type, times in self.operation_times.items():
            if times:
                avg_times[op_type] = np.mean(times)
                
        # Calculate current load
        uptime = time.time() - self.system_metrics['start_time']
        interactions_per_second = self.system_metrics['total_interactions'] / max(1, uptime)
        
        # Calculate recent load (last 5 minutes)
        current_minute = int(time.time() / 60) * 60
        recent_interactions = sum(
            count for ts, count in self.time_series['interaction_counts'].items()
            if current_minute - ts <= 300  # 5 minutes
        )
        recent_load = recent_interactions / 300.0
        
        # Calculate prediction accuracy
        prediction_accuracies = {}
        for metric, accuracies in self.prediction_accuracy.items():
            if accuracies:
                prediction_accuracies[metric] = np.mean(accuracies[-10:])  # Last 10 predictions
                
        return {
            'operation_counts': dict(self.operation_counts),
            'avg_operation_times': avg_times,
            'uptime': uptime,
            'total_interactions': self.system_metrics['total_interactions'],
            'total_couplings': self.system_metrics['total_couplings'],
            'interactions_per_second': interactions_per_second,
            'recent_load': recent_load,
            'peak_interactions_per_second': self.system_metrics['peak_interactions_per_second'],
            'predicted_performance': self.predicted_performance,
            'prediction_accuracy': prediction_accuracies
        }
        
    def _predict_future_performance(self):
        """
        Predict future performance metrics.
        Implements Seth's controlled hallucination principle for system metrics.
        """
        # Only predict occasionally to avoid overhead
        if np.random.random() > 0.1:  # 10% chance to predict
            return
            
        # Get time series data for prediction (last hour)
        current_minute = int(time.time() / 60) * 60
        recent_minutes = [ts for ts in self.time_series['interaction_counts'].keys()
                         if current_minute - ts <= 3600]  # 1 hour
                         
        if len(recent_minutes) < 10:  # Need enough data for prediction
            return
            
        # Sort timestamps
        recent_minutes.sort()
        
        # Extract interaction counts
        interaction_counts = [self.time_series['interaction_counts'][ts] for ts in recent_minutes]
        
        # Simple linear regression for prediction
        # X = minutes since start
        x = np.array([(ts - recent_minutes[0])/60 for ts in recent_minutes])
        y = np.array(interaction_counts)
        
        if len(x) < 2:
            return
            
        # Fit line: y = mx + b
        m, b = np.polyfit(x, y, 1)
        
        # Predict next hour
        predictions = {}
        for i in range(1, 13):  # Next 12 5-minute intervals
            future_x = len(x) + i*5/60  # 5 minutes per step, converted to hours
            predicted_y = m * future_x + b
            predictions[i*5] = max(0, predicted_y)  # Can't have negative interactions
            
        # Store prediction
        self.predicted_performance['interactions_per_5min'] = predictions
        
        # Evaluate previous predictions
        if 'interactions_per_5min' in self.predicted_performance:
            prev_predictions = self.predicted_performance['interactions_per_5min']
            
            for minutes, predicted in prev_predictions.items():
                predicted_time = current_minute - (3600 - minutes*60)  # When this was predicted for
                
                # Check if we have actual data for this time
                if predicted_time in self.time_series['interaction_counts']:
                    actual = self.time_series['interaction_counts'][predicted_time]
                    
                    # Calculate accuracy (1.0 = perfect, 0.0 = completely wrong)
                    max_value = max(actual, predicted, 1.0)  # Avoid division by zero
                    accuracy = 1.0 - min(1.0, abs(actual - predicted) / max_value)
                    
                    # Store accuracy
                    self.prediction_accuracy['interactions_per_5min'].append(accuracy)
                    
        # Limit history size
        for metric in self.prediction_accuracy:
            if len(self.prediction_accuracy[metric]) > 50:
                self.prediction_accuracy[metric] = self.prediction_accuracy[metric][-50:]

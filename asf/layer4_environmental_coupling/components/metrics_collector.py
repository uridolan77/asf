# === FILE: asf/environmental_coupling/components/metrics_collector.py ===
import asyncio
import time
import logging
import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

class PerformanceMetricsCollector:
    """
    Collects and analyzes performance metrics for the Environmental Coupling Layer.
    Enables optimization through data-driven insights.
    """
    
    def __init__(self):
        self.operation_timings = defaultdict(list)  # Maps operation type to timing list
        self.entity_metrics = defaultdict(dict)     # Maps entity_id to metric dict
        self.global_counters = defaultdict(int)     # Global counters for events
        self.recent_events = []                     # Queue of recent events
        self.start_time = time.time()
        self.last_reset = time.time()
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("ASF.Layer4.PerformanceMetricsCollector")
        
    async def record_operation(self, operation_type: str, duration: float, 
                             metadata: Optional[Dict] = None) -> None:
        """Record timing for an operation."""
        async with self.lock:
            # Record the timing
            self.operation_timings[operation_type].append({
                'timestamp': time.time(),
                'duration': duration,
                'metadata': metadata or {}
            })
            
            # Limit number of timings stored per operation
            if len(self.operation_timings[operation_type]) > 1000:
                self.operation_timings[operation_type] = self.operation_timings[operation_type][-1000:]
            
            # Update global counter
            self.global_counters[f"operation.{operation_type}"] += 1
            
            # Add to recent events
            self.recent_events.append({
                'type': 'operation',
                'operation_type': operation_type,
                'duration': duration,
                'timestamp': time.time()
            })
            
            # Limit recent events
            if len(self.recent_events) > 100:
                self.recent_events = self.recent_events[-100:]
    
    async def record_entity_event(self, entity_id: str, event_type: str,
                                data: Optional[Dict] = None) -> None:
        """Record an event for a specific entity."""
        async with self.lock:
            # Initialize entity metrics if needed
            if entity_id not in self.entity_metrics:
                self.entity_metrics[entity_id] = {
                    'first_seen': time.time(),
                    'event_count': 0,
                    'event_types': defaultdict(int),
                    'last_seen': time.time()
                }
            
            # Update entity metrics
            metrics = self.entity_metrics[entity_id]
            metrics['event_count'] += 1
            metrics['event_types'][event_type] += 1
            metrics['last_seen'] = time.time()
            
            # Store additional data if provided
            if data:
                if 'recent_data' not in metrics:
                    metrics['recent_data'] = []
                    
                metrics['recent_data'].append({
                    'timestamp': time.time(),
                    'event_type': event_type,
                    'data': data
                })
                
                # Limit recent data
                if len(metrics['recent_data']) > 10:
                    metrics['recent_data'] = metrics['recent_data'][-10:]
            
            # Update global counter
            self.global_counters[f"entity_event.{event_type}"] += 1
            
            # Add to recent events
            self.recent_events.append({
                'type': 'entity_event',
                'entity_id': entity_id,
                'event_type': event_type,
                'timestamp': time.time()
            })
            
            # Limit recent events
            if len(self.recent_events) > 100:
                self.recent_events = self.recent_events[-100:]
    
    async def increment_counter(self, counter_name: str, value: int = 1) -> int:
        """Increment a custom counter."""
        async with self.lock:
            self.global_counters[counter_name] += value
            return self.global_counters[counter_name]
    
    async def get_operation_stats(self, operation_type: str = None) -> Dict:
        """Get statistics for operations."""
        if operation_type:
            # Stats for specific operation type
            timings = self.operation_timings.get(operation_type, [])
            
            if not timings:
                return {
                    'operation_type': operation_type,
                    'count': 0,
                    'found': False
                }
                
            durations = [t['duration'] for t in timings]
            
            return {
                'operation_type': operation_type,
                'count': len(timings),
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'recent_duration': durations[-1] if durations else 0,
                'found': True
            }
        else:
            # Stats for all operation types
            stats = {}
            
            for op_type in self.operation_timings:
                stats[op_type] = await self.get_operation_stats(op_type)
                
            return stats
    
    async def get_entity_stats(self, entity_id: str = None) -> Dict:
        """Get statistics for entities."""
        if entity_id:
            # Stats for specific entity
            if entity_id not in self.entity_metrics:
                return {
                    'entity_id': entity_id,
                    'found': False
                }
                
            metrics = self.entity_metrics[entity_id]
            
            # Calculate active period
            active_period = metrics['last_seen'] - metrics['first_seen']
            
            # Calculate event frequency (events per hour)
            event_frequency = (metrics['event_count'] / active_period) * 3600 if active_period > 0 else 0
            
            return {
                'entity_id': entity_id,
                'event_count': metrics['event_count'],
                'first_seen': metrics['first_seen'],
                'last_seen': metrics['last_seen'],
                'active_period': active_period,
                'event_frequency': event_frequency,
                'event_types': dict(metrics['event_types']),
                'found': True
            }
        else:
            # Summary stats for all entities
            entity_count = len(self.entity_metrics)
            
            if entity_count == 0:
                return {
                    'entity_count': 0,
                    'total_events': 0
                }
                
            total_events = sum(m['event_count'] for m in self.entity_metrics.values())
            active_entities = sum(1 for m in self.entity_metrics.values() 
                                if time.time() - m['last_seen'] < 3600)  # Active in last hour
            
            return {
                'entity_count': entity_count,
                'total_events': total_events,
                'active_entities': active_entities,
                'avg_events_per_entity': total_events / entity_count if entity_count > 0 else 0
            }
    
    async def get_counters(self, pattern: str = None) -> Dict:
        """Get counter values, optionally filtered by pattern."""
        if pattern:
            # Filter counters by pattern
            return {
                name: value for name, value in self.global_counters.items()
                if pattern in name
            }
        else:
            # Return all counters
            return dict(self.global_counters)
    
    async def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get the most recent events."""
        return self.recent_events[-limit:]
    
    async def reset_metrics(self, keep_history: bool = False) -> Dict:
        """
        Reset metrics collectors.
        If keep_history is True, keeps historical data but resets counters.
        """
        async with self.lock:
            old_counters = dict(self.global_counters)
            
            if keep_history:
                # Just reset counters
                self.global_counters = defaultdict(int)
            else:
                # Reset everything
                self.operation_timings = defaultdict(list)
                self.entity_metrics = defaultdict(dict)
                self.global_counters = defaultdict(int)
                self.recent_events = []
            
            self.last_reset = time.time()
            
            return {
                'reset_time': self.last_reset,
                'previous_counters': old_counters,
                'kept_history': keep_history
            }
    
    async def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in the specified format."""
        # Collect all metrics
        metrics = {
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'since_reset': time.time() - self.last_reset,
            'counters': dict(self.global_counters),
            'entity_summary': await self.get_entity_stats(),
            'recent_events_count': len(self.recent_events)
        }
        
        # Format the metrics
        if format.lower() == 'json':
            return json.dumps(metrics, indent=2)
        elif format.lower() == 'text':
            # Simple text format
            lines = [
                f"Timestamp: {metrics['timestamp']}",
                f"Uptime: {metrics['uptime']:.2f} seconds",
                f"Since reset: {metrics['since_reset']:.2f} seconds",
                "Counters:",
            ]
            
            for name, value in sorted(metrics['counters'].items()):
                lines.append(f"  {name}: {value}")
                
            lines.append(f"Entities: {metrics['entity_summary']['entity_count']}")
            lines.append(f"Total events: {metrics['entity_summary']['total_events']}")
            
            return "\n".join(lines)
        else:
            return f"Unsupported format: {format}"
    
    async def analyze_performance(self) -> Dict:
        """
        Analyze performance metrics to identify optimization opportunities.
        Returns insights about system behavior.
        """
        insights = {
            'timestamp': time.time(),
            'bottlenecks': [],
            'anomalies': [],
            'trends': [],
            'recommendations': []
        }
        
        # Analyze operation timings for bottlenecks
        for op_type, timings in self.operation_timings.items():
            if not timings:
                continue
                
            durations = [t['duration'] for t in timings]
            avg_duration = sum(durations) / len(durations)
            
            # Check for operations that take longer than 1 second on average
            if avg_duration > 1.0:
                insights['bottlenecks'].append({
                    'operation_type': op_type,
                    'avg_duration': avg_duration,
                    'sample_count': len(timings)
                })
                
                insights['recommendations'].append({
                    'type': 'optimization',
                    'target': op_type,
                    'recommendation': f"Optimize {op_type} operations (avg: {avg_duration:.2f}s)"
                })
        
        # Analyze entity metrics for anomalies
        for entity_id, metrics in self.entity_metrics.items():
            # Check for very high event frequency
            if metrics['event_count'] > 1000:
                active_period = metrics['last_seen'] - metrics['first_seen']
                events_per_hour = (metrics['event_count'] / active_period) * 3600 if active_period > 0 else 0
                
                if events_per_hour > 100:  # More than 100 events per hour
                    insights['anomalies'].append({
                        'entity_id': entity_id,
                        'event_count': metrics['event_count'],
                        'events_per_hour': events_per_hour
                    })
                    
                    insights['recommendations'].append({
                        'type': 'investigation',
                        'target': entity_id,
                        'recommendation': f"Investigate high activity for entity {entity_id} ({events_per_hour:.1f} events/hour)"
                    })
        
        # Analyze global counters for trends
        operation_counts = {
            name.split('.')[-1]: value
            for name, value in self.global_counters.items()
            if name.startswith('operation.')
        }
        
        if operation_counts:
            # Identify imbalanced operation types
            total_ops = sum(operation_counts.values())
            for op_type, count in operation_counts.items():
                ratio = count / total_ops if total_ops > 0 else 0
                
                if ratio > 0.7:  # Operation type represents >70% of all operations
                    insights['trends'].append({
                        'type': 'dominant_operation',
                        'operation_type': op_type,
                        'ratio': ratio,
                        'count': count
                    })
        
        return insights

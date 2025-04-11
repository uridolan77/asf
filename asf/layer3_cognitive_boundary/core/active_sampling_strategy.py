import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import logging

class ActiveSamplingStrategy:
    """
    Implements active sampling strategies to optimize information seeking in ASF.
    
    This module proactively identifies knowledge areas with high uncertainty or 
    potential information gain, directing the system to seek information that 
    would most improve its knowledge. This embodies Seth's principle of active 
    inference - that an intelligent system doesn't just passively predict its 
    environment, but actively samples it to minimize prediction error.
    
    Key features:
    - Uncertainty-based exploration
    - Information gain estimation
    - Adaptive exploration/exploitation balance
    - Domain-specific sampling strategies
    - Curiosity-driven knowledge expansion
    """
    def __init__(self, semantic_network, predictive_processor, config=None):
        self.semantic_network = semantic_network
        self.predictive_processor = predictive_processor
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3.ActiveSampling")
        
        self.exploration_rate = self.config.get('exploration_rate', 0.3)  # Balance between exploration and exploitation
        self.novelty_weight = self.config.get('novelty_weight', 0.5)  # Weight for novelty in sampling decisions
        self.max_sampling_candidates = self.config.get('max_sampling_candidates', 50)  # Maximum candidates to evaluate
        self.min_expected_gain = self.config.get('min_expected_gain', 0.1)  # Minimum expected gain to consider sampling
        
        self.uncertainty_map = {}  # Maps knowledge regions to uncertainty levels
        self.sampling_history = []  # History of sampling decisions
        self.information_gain_history = {}  # Tracks actual information gain from previous sampling
        self.domain_importance = defaultdict(lambda: 1.0)  # Domain importance weights
        self.last_map_update = {}  # Last update time for each domain
        
        self.exploration_decay = self.config.get('exploration_decay', 0.999)  # Decay rate for exploration
        self.adaptive_exploration = self.config.get('adaptive_exploration', True)  # Enable adaptive exploration
        self.strategy_performance = defaultdict(list)  # Track performance of different strategies
        
    async def select_sampling_targets(self, domain: str, context: Dict[str, Any] = None, 
                                     max_targets: int = 3) -> List[Dict[str, Any]]:
        context = context or {}
        
        if domain not in self.last_map_update or \
           time.time() - self.last_map_update.get(domain, 0) > 300:  # 5 minutes
            await self._update_uncertainty_map(domain)
            
        candidates = await self._get_sampling_candidates(domain, context)
        
        targets = []
        
        strategies = [
            ('uncertainty', self._uncertainty_based_sampling),
            ('novelty', self._novelty_based_sampling),
            ('contradiction', self._contradiction_based_sampling),
            ('confidence', self._confidence_based_sampling)
        ]
        
        strategy_weights = self._calculate_strategy_weights()
        
        for candidate in candidates:
            candidate_scores = {}
            
            for strategy_name, strategy_func in strategies:
                score = await strategy_func(candidate, domain, context)
                candidate_scores[strategy_name] = score
                
                weight = strategy_weights.get(strategy_name, 1.0)
                candidate['expected_gain'] = candidate.get('expected_gain', 0) + (score * weight)
            
            candidate['strategy_scores'] = candidate_scores
            
            if candidate['expected_gain'] >= self.min_expected_gain:
                targets.append(candidate)
        
        targets.sort(key=lambda x: x['expected_gain'], reverse=True)
        
        for target in targets[:max_targets]:
            self.sampling_history.append({
                'domain': domain,
                'region': target['region'],
                'expected_gain': target['expected_gain'],
                'uncertainty': target.get('uncertainty', 0),
                'strategy_scores': target.get('strategy_scores', {}),
                'timestamp': time.time()
            })
        
        return targets[:max_targets]
    
    async def report_sampling_outcome(self, region: str, outcome: Dict[str, Any]) -> None:
        sampling_decision = None
        for decision in reversed(self.sampling_history):
            if decision['region'] == region:
                sampling_decision = decision
                break
                
        if not sampling_decision:
            return
            
        actual_gain = outcome.get('information_gain', 0)
        expected_gain = sampling_decision.get('expected_gain', 0)
        
        prediction_error = abs(actual_gain - expected_gain)
        
        self.information_gain_history[region] = {
            'expected': expected_gain,
            'actual': actual_gain,
            'prediction_error': prediction_error,
            'timestamp': time.time()
        }
        
        strategy_scores = sampling_decision.get('strategy_scores', {})
        for strategy, score in strategy_scores.items():
            strategy_error = abs(score - actual_gain)
            self.strategy_performance[strategy].append({
                'error': strategy_error,
                'score': score,
                'actual': actual_gain,
                'timestamp': time.time()
            })
            
            if len(self.strategy_performance[strategy]) > 100:
                self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
                
        if self.adaptive_exploration:
            if prediction_error > 0.3:  # Significant error
                self.exploration_rate = min(0.9, self.exploration_rate * 1.1)
            else:
                self.exploration_rate *= self.exploration_decay
                self.exploration_rate = max(0.05, self.exploration_rate)
    
    def set_domain_importance(self, domain: str, importance: float) -> None:
        """
        Set the importance weight for a knowledge domain.
        
        Args:
            domain: Domain identifier
            importance: Importance weight (0-1)
        """
        self.domain_importance[domain] = max(0, min(1, importance))
        
    async def get_domain_uncertainty(self, domain: str) -> float:
        if domain not in self.uncertainty_map:
            await self._update_uncertainty_map(domain)
            
        domain_regions = [r for r in self.uncertainty_map if r.startswith(f"{domain}:")]
        if not domain_regions:
            return 0.0
            
        uncertainties = [self.uncertainty_map[r] for r in domain_regions]
        return sum(uncertainties) / len(uncertainties)
    
    async def get_sampling_recommendations(self, context: Dict[str, Any] = None, 
                                         max_recommendations: int = 5) -> List[Dict[str, Any]]:
        context = context or {}
        all_recommendations = []
        
        domains = list(set([d.split(':')[0] for d in self.uncertainty_map.keys()]))
        
        for domain in domains:
            domain_targets = await self.select_sampling_targets(domain, context, max_targets=2)
            
            for target in domain_targets:
                importance = self.domain_importance.get(domain, 1.0)
                target['domain'] = domain
                target['expected_gain'] *= importance
                all_recommendations.append(target)
        
        all_recommendations.sort(key=lambda x: x['expected_gain'], reverse=True)
        
        return all_recommendations[:max_recommendations]
    
    async def _update_uncertainty_map(self, domain: str) -> None:
        domain_nodes = {}
        
        for node_id, node in self.semantic_network.nodes.items():
            if hasattr(node, 'metadata') and node.metadata.get('domain') == domain:
                domain_nodes[node_id] = node
                
        
        regions = await self._identify_knowledge_regions(domain, domain_nodes)
        
        for region, region_nodes in regions.items():
            region_id = f"{domain}:{region}"
            
            uncertainty = await self._calculate_region_uncertainty(region_id, region_nodes)
            
            self.uncertainty_map[region_id] = uncertainty
            
        self.last_map_update[domain] = time.time()
        
    async def _identify_knowledge_regions(self, domain: str, 
                                        domain_nodes: Dict[str, Any]) -> Dict[str, List[str]]:
        
        regions = defaultdict(list)
        
        for node_id, node in domain_nodes.items():
            if hasattr(node, 'properties') and 'category' in node.properties:
                region_name = node.properties['category']
            elif hasattr(node, 'label'):
                region_name = node.label.split()[0] if node.label else "general"
            else:
                region_name = "general"
                
            regions[region_name].append(node_id)
            
        return regions
    
    async def _calculate_region_uncertainty(self, region_id: str, 
                                          node_ids: List[str]) -> float:
        if not node_ids:
            return 0.0
            
        uncertainty_factors = []
        
        confidence_values = []
        for node_id in node_ids:
            node = self.semantic_network.nodes.get(node_id)
            if node and hasattr(node, 'confidence'):
                confidence_values.append(node.confidence)
                
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            confidence_uncertainty = 1.0 - avg_confidence
            uncertainty_factors.append(confidence_uncertainty)
        
        precision_values = []
        for node_id in node_ids:
            precision = self.predictive_processor.get_precision(node_id)
            if precision is not None:
                precision_values.append(precision)
                
        if precision_values:
            avg_precision = sum(precision_values) / len(precision_values)
            precision_uncertainty = 1.0 / (1.0 + avg_precision)
            uncertainty_factors.append(precision_uncertainty)
            
        contradiction_uncertainty = 0.3  # Placeholder
        uncertainty_factors.append(contradiction_uncertainty)
        
        temporal_uncertainty = 0.4  # Placeholder
        uncertainty_factors.append(temporal_uncertainty)
        
        if uncertainty_factors:
            return sum(uncertainty_factors) / len(uncertainty_factors)
        else:
            return 0.5  # Default medium uncertainty
    
    async def _get_sampling_candidates(self, domain: str, 
                                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = []
        
        domain_regions = [(r, u) for r, u in self.uncertainty_map.items() 
                         if r.startswith(f"{domain}:")]
        
        for region_id, uncertainty in domain_regions:
            region_parts = region_id.split(':')
            pure_region = region_parts[1] if len(region_parts) > 1 else region_id
            
            node_ids = []
            for node_id, node in self.semantic_network.nodes.items():
                if (hasattr(node, 'metadata') and node.metadata.get('domain') == domain and
                    hasattr(node, 'properties') and node.properties.get('category') == pure_region):
                    node_ids.append(node_id)
            
            expected_gain = uncertainty
            
            candidate = {
                'region': region_id,
                'pure_region': pure_region,
                'uncertainty': uncertainty,
                'node_ids': node_ids,
                'expected_gain': expected_gain
            }
            
            candidates.append(candidate)
            
        return candidates[:self.max_sampling_candidates]
    
    async def _uncertainty_based_sampling(self, candidate: Dict[str, Any], 
                                        domain: str, context: Dict[str, Any]) -> float:
        return candidate.get('uncertainty', 0.5)
    
    async def _novelty_based_sampling(self, candidate: Dict[str, Any], 
                                    domain: str, context: Dict[str, Any]) -> float:
        region_id = candidate.get('region')
        
        sampled_before = False
        for decision in self.sampling_history:
            if decision['region'] == region_id:
                sampled_before = True
                break
                
        if not sampled_before:
            return 0.8  # High score for novel regions
        else:
            for i, decision in enumerate(reversed(self.sampling_history)):
                if decision['region'] == region_id:
                    recency_factor = min(1.0, i / 10.0)  # Scale based on recency
                    return 0.3 + (0.5 * recency_factor)  # 0.3 to 0.8 based on recency
                    
        return 0.5
    
    async def _contradiction_based_sampling(self, candidate: Dict[str, Any], 
                                          domain: str, context: Dict[str, Any]) -> float:
        
        node_ids = candidate.get('node_ids', [])
        
        contradiction_potential = 0.5  # Default medium potential
        
        
        return contradiction_potential
    
    async def _confidence_based_sampling(self, candidate: Dict[str, Any], 
                                       domain: str, context: Dict[str, Any]) -> float:
        node_ids = candidate.get('node_ids', [])
        
        confidence_values = []
        for node_id in node_ids:
            node = self.semantic_network.nodes.get(node_id)
            if node and hasattr(node, 'confidence'):
                confidence_values.append(node.confidence)
                
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            return 1.0 - avg_confidence
        else:
            return 0.5  # Default medium score
    
    def _calculate_strategy_weights(self) -> Dict[str, float]:
        """
        Calculate weights for different sampling strategies based on performance.
        
        Returns:
            Dictionary mapping strategies to weights
        """
        weights = {}
        
        for strategy, performance in self.strategy_performance.items():
            if not performance:
                weights[strategy] = 1.0  # Default equal weight
                continue
                
            recent_performance = performance[-20:]  # Last 20 samples
            
            if not recent_performance:
                weights[strategy] = 1.0
                continue
                
            avg_error = sum(p['error'] for p in recent_performance) / len(recent_performance)
            
            weight = 2.0 - min(1.8, avg_error * 2.0)
            weights[strategy] = max(0.2, weight)
            
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {k: v / total_weight * len(weights) for k, v in weights.items()}
        else:
            return {k: 1.0 for k in weights}
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about sampling performance.
        
        Returns:
            Dictionary of sampling statistics
        """
        stats = {
            'parameters': {
                'exploration_rate': self.exploration_rate,
                'novelty_weight': self.novelty_weight,
                'adaptive_exploration': self.adaptive_exploration
            },
            'strategy_weights': self._calculate_strategy_weights(),
            'sampling_history': {
                'total_samples': len(self.sampling_history),
                'recent_samples': self.sampling_history[-5:] if self.sampling_history else []
            },
            'domains': {
                domain: {
                    'importance': importance,
                    'last_updated': self.last_map_update.get(domain, 0)
                }
                for domain, importance in self.domain_importance.items()
            }
        }
        
        if self.information_gain_history:
            expected_gains = [info['expected'] for info in self.information_gain_history.values()]
            actual_gains = [info['actual'] for info in self.information_gain_history.values()]
            prediction_errors = [info['prediction_error'] for info in self.information_gain_history.values()]
            
            stats['information_gain'] = {
                'average_expected': sum(expected_gains) / len(expected_gains) if expected_gains else 0,
                'average_actual': sum(actual_gains) / len(actual_gains) if actual_gains else 0,
                'average_error': sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0,
                'count': len(self.information_gain_history)
            }
            
        return stats
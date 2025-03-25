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
        
        # Configuration parameters
        self.exploration_rate = self.config.get('exploration_rate', 0.3)  # Balance between exploration and exploitation
        self.novelty_weight = self.config.get('novelty_weight', 0.5)  # Weight for novelty in sampling decisions
        self.max_sampling_candidates = self.config.get('max_sampling_candidates', 50)  # Maximum candidates to evaluate
        self.min_expected_gain = self.config.get('min_expected_gain', 0.1)  # Minimum expected gain to consider sampling
        
        # State tracking
        self.uncertainty_map = {}  # Maps knowledge regions to uncertainty levels
        self.sampling_history = []  # History of sampling decisions
        self.information_gain_history = {}  # Tracks actual information gain from previous sampling
        self.domain_importance = defaultdict(lambda: 1.0)  # Domain importance weights
        self.last_map_update = {}  # Last update time for each domain
        
        # Adaptive parameters
        self.exploration_decay = self.config.get('exploration_decay', 0.999)  # Decay rate for exploration
        self.adaptive_exploration = self.config.get('adaptive_exploration', True)  # Enable adaptive exploration
        self.strategy_performance = defaultdict(list)  # Track performance of different strategies
        
    async def select_sampling_targets(self, domain: str, context: Dict[str, Any] = None, 
                                     max_targets: int = 3) -> List[Dict[str, Any]]:
        """
        Select knowledge regions to actively sample based on expected information gain.
        Implements Seth's active inference principle.
        
        Args:
            domain: Knowledge domain to sample
            context: Optional context for sampling
            max_targets: Maximum number of sampling targets
            
        Returns:
            List of sampling targets with expected information gain
        """
        context = context or {}
        
        # Check if uncertainty map needs updating
        if domain not in self.last_map_update or \
           time.time() - self.last_map_update.get(domain, 0) > 300:  # 5 minutes
            await self._update_uncertainty_map(domain)
            
        # Get sampling candidates
        candidates = await self._get_sampling_candidates(domain, context)
        
        # Calculate information gain
        targets = []
        
        # Apply different sampling strategies
        strategies = [
            ('uncertainty', self._uncertainty_based_sampling),
            ('novelty', self._novelty_based_sampling),
            ('contradiction', self._contradiction_based_sampling),
            ('confidence', self._confidence_based_sampling)
        ]
        
        # Weight each strategy based on past performance
        strategy_weights = self._calculate_strategy_weights()
        
        # Blend target scores from different strategies
        for candidate in candidates:
            candidate_scores = {}
            
            for strategy_name, strategy_func in strategies:
                # Get score from this strategy
                score = await strategy_func(candidate, domain, context)
                candidate_scores[strategy_name] = score
                
                # Apply strategy weight
                weight = strategy_weights.get(strategy_name, 1.0)
                candidate['expected_gain'] = candidate.get('expected_gain', 0) + (score * weight)
            
            # Record the breakdown of scores
            candidate['strategy_scores'] = candidate_scores
            
            # Only include candidates with sufficient expected gain
            if candidate['expected_gain'] >= self.min_expected_gain:
                targets.append(candidate)
        
        # Sort by expected information gain
        targets.sort(key=lambda x: x['expected_gain'], reverse=True)
        
        # Record sampling targets for performance tracking
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
        """
        Report actual outcomes from sampling to refine future sampling strategies.
        
        Args:
            region: Knowledge region that was sampled
            outcome: Results of sampling including actual information gain
        """
        # Find relevant sampling decision
        sampling_decision = None
        for decision in reversed(self.sampling_history):
            if decision['region'] == region:
                sampling_decision = decision
                break
                
        if not sampling_decision:
            return
            
        # Calculate actual information gain
        actual_gain = outcome.get('information_gain', 0)
        expected_gain = sampling_decision.get('expected_gain', 0)
        
        # Calculate prediction error
        prediction_error = abs(actual_gain - expected_gain)
        
        # Record actual information gain
        self.information_gain_history[region] = {
            'expected': expected_gain,
            'actual': actual_gain,
            'prediction_error': prediction_error,
            'timestamp': time.time()
        }
        
        # Update strategy performance
        strategy_scores = sampling_decision.get('strategy_scores', {})
        for strategy, score in strategy_scores.items():
            # Calculate how well this strategy predicted the actual gain
            strategy_error = abs(score - actual_gain)
            self.strategy_performance[strategy].append({
                'error': strategy_error,
                'score': score,
                'actual': actual_gain,
                'timestamp': time.time()
            })
            
            # Limit history size
            if len(self.strategy_performance[strategy]) > 100:
                self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
                
        # Adjust exploration rate if adaptive exploration is enabled
        if self.adaptive_exploration:
            # If prediction error is high, increase exploration
            if prediction_error > 0.3:  # Significant error
                self.exploration_rate = min(0.9, self.exploration_rate * 1.1)
            else:
                # Gradually decay exploration rate
                self.exploration_rate *= self.exploration_decay
                # Ensure minimum exploration
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
        """
        Get overall uncertainty level for a domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Uncertainty score (0-1)
        """
        if domain not in self.uncertainty_map:
            await self._update_uncertainty_map(domain)
            
        domain_regions = [r for r in self.uncertainty_map if r.startswith(f"{domain}:")]
        if not domain_regions:
            return 0.0
            
        # Calculate average uncertainty across domain regions
        uncertainties = [self.uncertainty_map[r] for r in domain_regions]
        return sum(uncertainties) / len(uncertainties)
    
    async def get_sampling_recommendations(self, context: Dict[str, Any] = None, 
                                         max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get cross-domain sampling recommendations based on overall information needs.
        
        Args:
            context: Optional context
            max_recommendations: Maximum recommendations to return
            
        Returns:
            List of sampling recommendations
        """
        context = context or {}
        all_recommendations = []
        
        # Get domains for recommendation
        domains = list(set([d.split(':')[0] for d in self.uncertainty_map.keys()]))
        
        # For each domain, get top sampling target
        for domain in domains:
            domain_targets = await self.select_sampling_targets(domain, context, max_targets=2)
            
            for target in domain_targets:
                # Apply domain importance
                importance = self.domain_importance.get(domain, 1.0)
                target['domain'] = domain
                target['expected_gain'] *= importance
                all_recommendations.append(target)
        
        # Sort by expected gain
        all_recommendations.sort(key=lambda x: x['expected_gain'], reverse=True)
        
        return all_recommendations[:max_recommendations]
    
    async def _update_uncertainty_map(self, domain: str) -> None:
        """
        Update knowledge region uncertainty map for a domain.
        
        Args:
            domain: Domain to update
        """
        # Get related nodes in the domain
        domain_nodes = {}
        
        # Ideally, we would query the semantic network for domain nodes
        # For now, we'll simulate with sample code
        for node_id, node in self.semantic_network.nodes.items():
            if hasattr(node, 'metadata') and node.metadata.get('domain') == domain:
                domain_nodes[node_id] = node
                
        # Calculate uncertainty for regions
        # A region could be a concept, a cluster of concepts, or a specific area
        # of knowledge within the domain
        
        # First, identify regions based on node clustering or predefined categories
        regions = await self._identify_knowledge_regions(domain, domain_nodes)
        
        # Then calculate uncertainty for each region
        for region, region_nodes in regions.items():
            # Create region ID
            region_id = f"{domain}:{region}"
            
            # Calculate uncertainty based on multiple factors
            uncertainty = await self._calculate_region_uncertainty(region_id, region_nodes)
            
            # Update uncertainty map
            self.uncertainty_map[region_id] = uncertainty
            
        # Update last update time
        self.last_map_update[domain] = time.time()
        
    async def _identify_knowledge_regions(self, domain: str, 
                                        domain_nodes: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Identify knowledge regions within a domain.
        
        Args:
            domain: Domain identifier
            domain_nodes: Dictionary of domain nodes
            
        Returns:
            Dictionary mapping region names to lists of node IDs
        """
        # This would ideally use clustering or predefined categories
        # For now, we'll use a simple approach based on node properties
        
        regions = defaultdict(list)
        
        for node_id, node in domain_nodes.items():
            if hasattr(node, 'properties') and 'category' in node.properties:
                # Use category as region
                region_name = node.properties['category']
            elif hasattr(node, 'label'):
                # Use first word of label as region
                region_name = node.label.split()[0] if node.label else "general"
            else:
                region_name = "general"
                
            regions[region_name].append(node_id)
            
        return regions
    
    async def _calculate_region_uncertainty(self, region_id: str, 
                                          node_ids: List[str]) -> float:
        """
        Calculate uncertainty level for a knowledge region.
        
        Args:
            region_id: Region identifier
            node_ids: List of node IDs in the region
            
        Returns:
            Uncertainty score (0-1)
        """
        if not node_ids:
            return 0.0
            
        # Multiple factors contribute to uncertainty
        uncertainty_factors = []
        
        # 1. Confidence-based uncertainty
        confidence_values = []
        for node_id in node_ids:
            node = self.semantic_network.nodes.get(node_id)
            if node and hasattr(node, 'confidence'):
                confidence_values.append(node.confidence)
                
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            confidence_uncertainty = 1.0 - avg_confidence
            uncertainty_factors.append(confidence_uncertainty)
        
        # 2. Precision-based uncertainty
        precision_values = []
        for node_id in node_ids:
            precision = self.predictive_processor.get_precision(node_id)
            if precision is not None:
                precision_values.append(precision)
                
        if precision_values:
            # Normalize precision (higher precision = lower uncertainty)
            avg_precision = sum(precision_values) / len(precision_values)
            precision_uncertainty = 1.0 / (1.0 + avg_precision)
            uncertainty_factors.append(precision_uncertainty)
            
        # 3. Contradiction-based uncertainty
        # Ideally check for contradictions in the region
        # For now, use placeholder
        contradiction_uncertainty = 0.3  # Placeholder
        uncertainty_factors.append(contradiction_uncertainty)
        
        # 4. Temporal uncertainty - older knowledge has higher uncertainty
        temporal_uncertainty = 0.4  # Placeholder
        uncertainty_factors.append(temporal_uncertainty)
        
        # Calculate weighted uncertainty
        if uncertainty_factors:
            # Different weights could be applied here
            return sum(uncertainty_factors) / len(uncertainty_factors)
        else:
            return 0.5  # Default medium uncertainty
    
    async def _get_sampling_candidates(self, domain: str, 
                                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get candidate regions for sampling.
        
        Args:
            domain: Domain identifier
            context: Sampling context
            
        Returns:
            List of candidate regions with metadata
        """
        candidates = []
        
        # Filter to this domain
        domain_regions = [(r, u) for r, u in self.uncertainty_map.items() 
                         if r.startswith(f"{domain}:")]
        
        for region_id, uncertainty in domain_regions:
            # Get precision for this region
            region_parts = region_id.split(':')
            pure_region = region_parts[1] if len(region_parts) > 1 else region_id
            
            # Get associated node IDs (simplified)
            node_ids = []
            for node_id, node in self.semantic_network.nodes.items():
                if (hasattr(node, 'metadata') and node.metadata.get('domain') == domain and
                    hasattr(node, 'properties') and node.properties.get('category') == pure_region):
                    node_ids.append(node_id)
            
            # Calculate initial expected gain from uncertainty
            expected_gain = uncertainty
            
            # Create candidate
            candidate = {
                'region': region_id,
                'pure_region': pure_region,
                'uncertainty': uncertainty,
                'node_ids': node_ids,
                'expected_gain': expected_gain
            }
            
            candidates.append(candidate)
            
        # Limit number of candidates
        return candidates[:self.max_sampling_candidates]
    
    async def _uncertainty_based_sampling(self, candidate: Dict[str, Any], 
                                        domain: str, context: Dict[str, Any]) -> float:
        """
        Score candidate based on uncertainty.
        
        Args:
            candidate: Candidate information
            domain: Domain identifier
            context: Sampling context
            
        Returns:
            Sampling score
        """
        # Higher uncertainty = higher score
        return candidate.get('uncertainty', 0.5)
    
    async def _novelty_based_sampling(self, candidate: Dict[str, Any], 
                                    domain: str, context: Dict[str, Any]) -> float:
        """
        Score candidate based on novelty.
        
        Args:
            candidate: Candidate information
            domain: Domain identifier
            context: Sampling context
            
        Returns:
            Sampling score
        """
        region_id = candidate.get('region')
        
        # Check if we've sampled this region before
        sampled_before = False
        for decision in self.sampling_history:
            if decision['region'] == region_id:
                sampled_before = True
                break
                
        # Higher novelty = higher score
        if not sampled_before:
            return 0.8  # High score for novel regions
        else:
            # Check how recently it was sampled
            for i, decision in enumerate(reversed(self.sampling_history)):
                if decision['region'] == region_id:
                    # More recent = lower novelty score
                    recency_factor = min(1.0, i / 10.0)  # Scale based on recency
                    return 0.3 + (0.5 * recency_factor)  # 0.3 to 0.8 based on recency
                    
        # Default novelty score
        return 0.5
    
    async def _contradiction_based_sampling(self, candidate: Dict[str, Any], 
                                          domain: str, context: Dict[str, Any]) -> float:
        """
        Score candidate based on potential contradictions.
        
        Args:
            candidate: Candidate information
            domain: Domain identifier
            context: Sampling context
            
        Returns:
            Sampling score
        """
        # Higher contradiction potential = higher score
        
        # Placeholder implementation
        # In a real implementation, we would check for conflicting nodes or relations
        node_ids = candidate.get('node_ids', [])
        
        contradiction_potential = 0.5  # Default medium potential
        
        # In a real implementation, analyze nodes for contradictions
        
        return contradiction_potential
    
    async def _confidence_based_sampling(self, candidate: Dict[str, Any], 
                                       domain: str, context: Dict[str, Any]) -> float:
        """
        Score candidate based on confidence levels.
        
        Args:
            candidate: Candidate information
            domain: Domain identifier
            context: Sampling context
            
        Returns:
            Sampling score
        """
        # Lower confidence = higher score
        node_ids = candidate.get('node_ids', [])
        
        confidence_values = []
        for node_id in node_ids:
            node = self.semantic_network.nodes.get(node_id)
            if node and hasattr(node, 'confidence'):
                confidence_values.append(node.confidence)
                
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            # Lower confidence = higher sampling score
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
                
            # Calculate average error for this strategy
            recent_performance = performance[-20:]  # Last 20 samples
            
            if not recent_performance:
                weights[strategy] = 1.0
                continue
                
            avg_error = sum(p['error'] for p in recent_performance) / len(recent_performance)
            
            # Convert error to weight (lower error = higher weight)
            # Scale between 0.2 and 2.0
            weight = 2.0 - min(1.8, avg_error * 2.0)
            weights[strategy] = max(0.2, weight)
            
        # Normalize weights
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
        
        # Calculate average information gain
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
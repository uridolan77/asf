import asyncio
import logging
from typing import Dict, List, Any, Optional

# Import ASF components
from asf.layer3_cognitive_boundary.cognitive_boundary_layer import CognitiveBoundaryLayer
from asf.layer3_cognitive_boundary.core.semantic_tensor_network import SemanticTensorNetwork
from asf.layer3_cognitive_boundary.predictive_processor import PredictiveProcessor

# Import new components
from core.active_sampling_strategy import ActiveSamplingStrategy

class EnhancedCognitiveBoundaryLayer(CognitiveBoundaryLayer):
    """
    Enhanced Cognitive Boundary Layer with Active Sampling capabilities.
    """
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize active sampling after other components are initialized
        self.active_sampling = ActiveSamplingStrategy(
            self.semantic_network,
            self.predictive_processor,
            config=self.config.get('active_sampling', {})
        )
        
        self.logger.info("Active Sampling Strategy initialized")
        
    async def initialize(self):
        """Initialize the layer with enhanced components."""
        await super().initialize()
        
        # Register domains for sampling
        domains = self.config.get('knowledge_domains', ['general', 'medical', 'financial'])
        for domain in domains:
            self.active_sampling.set_domain_importance(domain, 1.0)
            
        self.logger.info(f"Registered {len(domains)} domains for active sampling")
        
        return True
        
    async def get_information_needs(self, context=None):
        """
        Get current information needs based on active sampling.
        
        Args:
            context: Optional context information
            
        Returns:
            List of information needs with expected gain
        """
        context = context or {}
        
        # Get sampling recommendations
        recommendations = await self.active_sampling.get_sampling_recommendations(
            context=context,
            max_recommendations=5
        )
        
        # Format as information needs
        info_needs = []
        for rec in recommendations:
            # Extract region information
            region = rec.get('pure_region', '')
            domain = rec.get('domain', 'general')
            
            # Create query components based on region and domain
            query_components = self._create_query_components(region, domain)
            
            info_need = {
                'domain': domain,
                'region': region,
                'expected_gain': rec.get('expected_gain', 0),
                'uncertainty': rec.get('uncertainty', 0),
                'query_components': query_components,
                'priority': min(0.9, 0.5 + rec.get('expected_gain', 0))
            }
            
            info_needs.append(info_need)
            
        return info_needs
    
    async def integrate_new_information(self, information, source_info, domain):
        """
        Integrate new information and report sampling outcome.
        
        Args:
            information: New information to integrate
            source_info: Information source metadata
            domain: Knowledge domain
            
        Returns:
            Integration result
        """
        # Extract region from information
        region = self._extract_region_from_information(information, domain)
        
        # Process the information (integrate into semantic network)
        # This would call existing integration methods
        integration_result = await self._process_information(information, domain)
        
        # Calculate actual information gain
        information_gain = self._calculate_information_gain(integration_result)
        
        # Report sampling outcome
        if region:
            await self.active_sampling.report_sampling_outcome(
                region=f"{domain}:{region}",
                outcome={
                    'information_gain': information_gain,
                    'integration_result': integration_result,
                    'source_info': source_info
                }
            )
        
        return integration_result
    
    def _create_query_components(self, region, domain):
        """Create query components from region and domain."""
        # In a real implementation, this would create specific query terms
        # based on the knowledge region and domain
        
        components = {
            'key_terms': [region],
            'domain_filters': [domain],
            'recency': 'recent' if domain in ['news', 'financial'] else 'any',
            'depth': 'comprehensive' if domain in ['medical', 'scientific'] else 'overview'
        }
        
        return components
    
    def _extract_region_from_information(self, information, domain):
        """Extract region from information."""
        # In a real implementation, this would analyze the information
        # to determine which knowledge region it belongs to
        
        # Simplified implementation
        if isinstance(information, dict) and 'category' in information:
            return information['category']
        elif isinstance(information, dict) and 'topic' in information:
            return information['topic']
        else:
            return "general"
    
    async def _process_information(self, information, domain):
        """Process and integrate new information."""
        # In a real implementation, this would call into the appropriate
        # integration methods to add the information to the semantic network
        
        # Simplified placeholder
        result = {
            'status': 'success',
            'nodes_created': 1,
            'relations_created': 2,
            'contradictions_detected': 0
        }
        
        return result
    
    def _calculate_information_gain(self, integration_result):
        """Calculate actual information gain from integration result."""
        # In a real implementation, this would analyze changes to the
        # semantic network to determine how much the information improved
        # the system's knowledge
        
        # Simplified placeholder calculation
        gain = 0.0
        
        if integration_result.get('status') == 'success':
            # Base gain on number of created elements
            nodes_created = integration_result.get('nodes_created', 0)
            relations_created = integration_result.get('relations_created', 0)
            contradictions = integration_result.get('contradictions_detected', 0)
            
            # More nodes/relations = more information
            element_gain = (nodes_created * 0.1) + (relations_created * 0.05)
            
            # Contradictions can be valuable for learning
            contradiction_gain = contradictions * 0.2
            
            gain = min(1.0, element_gain + contradiction_gain)
            
        return gain
    
    
async def example_usage():
    """Example usage of the enhanced layer."""
    # Initialize the enhanced layer
    config = {
        'knowledge_domains': ['medical', 'financial', 'news', 'scientific'],
        'active_sampling': {
            'exploration_rate': 0.4,
            'adaptive_exploration': True
        }
    }
    
    enhanced_layer = EnhancedCognitiveBoundaryLayer(config)
    await enhanced_layer.initialize()
    
    # Get information needs
    context = {
        'current_focus': 'medical research',
        'user_interests': ['cancer treatment', 'genetic analysis']
    }
    
    info_needs = await enhanced_layer.get_information_needs(context)
    
    print(f"Identified {len(info_needs)} information needs:")
    for need in info_needs:
        print(f"Domain: {need['domain']}, Region: {need['region']}")
        print(f"Expected gain: {need['expected_gain']:.2f}, Priority: {need['priority']:.2f}")
        print(f"Query components: {need['query_components']}")
        print("---")
    
    # Simulate receiving new information
    new_information = {
        'category': 'cancer_treatment',
        'content': "New research suggests combination therapy improves outcomes...",
        'source': "Journal of Medical Research",
        'timestamp': 1635781234
    }
    
    source_info = {
        'reliability': 0.9,
        'authority': 'scientific_journal',
        'recency': 'very_recent'
    }
    
    # Integrate the information
    result = await enhanced_layer.integrate_new_information(
        new_information, source_info, 'medical'
    )
    
    print(f"Integration result: {result}")
    
    # Check updated information needs
    updated_needs = await enhanced_layer.get_information_needs(context)
    print(f"Updated information needs: {len(updated_needs)}")
    
    # Get sampling statistics
    stats = enhanced_layer.active_sampling.get_sampling_statistics()
    print("Sampling statistics:")
    print(f"- Exploration rate: {stats['parameters']['exploration_rate']:.2f}")
    print(f"- Strategy weights: {stats['strategy_weights']}")
    if 'information_gain' in stats:
        print(f"- Average information gain: {stats['information_gain']['average_actual']:.2f}")
    
    # Demonstrate adaptive sampling across multiple iterations
    print("\nSimulating adaptive sampling across multiple iterations...")
    
    for i in range(5):
        print(f"\nIteration {i+1}:")
        
        # Get information needs
        info_needs = await enhanced_layer.get_information_needs(context)
        if not info_needs:
            print("No more information needs identified.")
            break
            
        top_need = info_needs[0]
        print(f"Top information need: {top_need['domain']}:{top_need['region']}")
        print(f"Expected gain: {top_need['expected_gain']:.2f}")
        
        # Simulate finding information with varying quality
        information_quality = 0.5 + (0.3 * (i % 3))  # Varies between 0.5 and 0.8
        
        # Generate simulated information
        simulated_info = {
            'category': top_need['region'],
            'content': f"Simulated content for {top_need['region']}...",
            'quality': information_quality
        }
        
        simulated_source = {
            'reliability': 0.7 + (0.1 * (i % 3)),
            'authority': 'simulation',
            'recency': 'recent'
        }
        
        # Integrate the information
        result = await enhanced_layer.integrate_new_information(
            simulated_info, simulated_source, top_need['domain']
        )
        
        print(f"Integration result: nodes={result['nodes_created']}, relations={result['relations_created']}")
        
        # Get updated statistics
        stats = enhanced_layer.active_sampling.get_sampling_statistics()
        print(f"Updated exploration rate: {stats['parameters']['exploration_rate']:.2f}")
        
        # Show how strategy weights evolve
        print("Strategy weights:")
        for strategy, weight in stats['strategy_weights'].items():
            print(f"  - {strategy}: {weight:.2f}")
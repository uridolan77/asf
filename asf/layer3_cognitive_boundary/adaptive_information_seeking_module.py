import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

class AdaptiveInformationSeekingModule:
    """
    Integrates Active Sampling Strategy with the Environmental Interface layer.
    
    This module bridges Layer 3 (Cognitive Boundary) and Layer 4 (Environmental Interface)
    by transforming abstract information needs identified through active sampling into
    concrete information seeking actions in the external environment.
    
    Key features:
    - Translation of information needs to environment-specific queries
    - Strategic scheduling of information seeking actions
    - Multi-domain information source management
    - Feedback loop for continuous improvement of information seeking
    """
    def __init__(self, environmental_interface, active_sampling_strategy, config=None):
        self.env_interface = environmental_interface
        self.active_sampling = active_sampling_strategy
        self.config = config or {}
        self.logger = logging.getLogger("ASF.InformationSeeking")
        
        # Configuration parameters
        self.max_concurrent_queries = self.config.get('max_concurrent_queries', 3)
        self.query_timeout = self.config.get('query_timeout', 60)  # seconds
        self.min_refresh_interval = self.config.get('min_refresh_interval', 300)  # seconds
        
        # State tracking
        self.active_queries = {}  # Query ID -> query info
        self.query_history = []  # History of executed queries
        self.source_performance = defaultdict(list)  # Source ID -> performance history
        self.domain_sources = defaultdict(set)  # Domain -> set of source IDs
        self.last_needs_refresh = 0  # Last time information needs were refreshed
        self.information_cache = {}  # Cache of recently retrieved information
        
    async def initialize(self):
        """Initialize the module."""
        self.logger.info("Initializing Adaptive Information Seeking Module")
        
        # Register initial information sources
        await self._register_initial_sources()
        
        return True
        
    async def seek_information(self, context=None):
        """
        Proactively seek information based on identified information needs.
        
        Args:
            context: Optional context information
            
        Returns:
            Results of information seeking
        """
        context = context or {}
        start_time = time.time()
        
        # Refresh information needs if needed
        await self._refresh_information_needs(context)
        
        # Get current information needs
        info_needs = await self.active_sampling.get_sampling_recommendations(
            context=context,
            max_recommendations=self.max_concurrent_queries * 2  # Get extra for prioritization
        )
        
        # Filter out needs that are already being queried
        filtered_needs = [need for need in info_needs 
                         if need['region'] not in [q['region'] for q in self.active_queries.values()]]
        
        # Determine how many new queries we can start
        available_slots = self.max_concurrent_queries - len(self.active_queries)
        new_queries_count = min(available_slots, len(filtered_needs))
        
        if new_queries_count <= 0:
            return {
                'status': 'no_slots_available',
                'active_queries': len(self.active_queries),
                'pending_needs': len(filtered_needs)
            }
            
        # Select top needs to query
        selected_needs = filtered_needs[:new_queries_count]
        query_results = []
        
        # Execute queries for selected needs
        for need in selected_needs:
            query_result = await self._execute_query_for_need(need, context)
            if query_result:
                query_results.append(query_result)
                
        # Check status of ongoing queries
        completed_queries = await self._check_active_queries()
        
        # Prepare result summary
        result = {
            'status': 'success',
            'new_queries': len(query_results),
            'completed_queries': len(completed_queries),
            'active_queries': len(self.active_queries),
            'pending_needs': len(filtered_needs) - new_queries_count,
            'execution_time': time.time() - start_time,
            'results': query_results,
            'completed': completed_queries
        }
        
        return result
    
    async def register_information_source(self, source_info):
        """
        Register a new information source for specific domains.
        
        Args:
            source_info: Source information including ID, domains, and access details
            
        Returns:
            Registration result
        """
        source_id = source_info.get('id')
        if not source_id:
            return {'status': 'error', 'message': 'Source ID is required'}
            
        # Register with environmental interface
        result = await self.env_interface.register_information_source(
            source_info['id'],
            source_info.get('domains', ['general']),
            source_info.get('credibility', 0.5)
        )
        
        # Register with our local tracking
        domains = source_info.get('domains', ['general'])
        for domain in domains:
            self.domain_sources[domain].add(source_id)
            
        return {
            'status': 'success',
            'source_id': source_id,
            'domains': domains
        }
    
    async def report_query_outcome(self, query_id, outcome):
        """
        Report outcome of an information query.
        
        Args:
            query_id: Query identifier
            outcome: Query outcome information
            
        Returns:
            Processing result
        """
        if query_id not in self.active_queries:
            return {'status': 'unknown_query', 'query_id': query_id}
            
        query_info = self.active_queries.pop(query_id)
        
        # Record in history
        query_info['outcome'] = outcome
        query_info['completion_time'] = time.time()
        self.query_history.append(query_info)
        
        # Limit history size
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
            
        # Update source performance
        source_id = query_info.get('source_id')
        if source_id:
            self.source_performance[source_id].append({
                'query_id': query_id,
                'success': outcome.get('success', False),
                'relevance': outcome.get('relevance', 0),
                'timestamp': time.time()
            })
            
            # Limit performance history
            if len(self.source_performance[source_id]) > 50:
                self.source_performance[source_id] = self.source_performance[source_id][-50:]
                
        # Report sampling outcome if information was found
        if outcome.get('success', False) and outcome.get('information'):
            region = query_info.get('region')
            domain = query_info.get('domain')
            
            if region and domain:
                region_id = f"{domain}:{region}"
                
                # Calculate information gain
                info_gain = outcome.get('relevance', 0) * outcome.get('quality', 0.5)
                
                # Report to active sampling
                await self.active_sampling.report_sampling_outcome(
                    region=region_id,
                    outcome={
                        'information_gain': info_gain,
                        'information': outcome.get('information')
                    }
                )
                
                # Process the information through the environmental interface
                if outcome.get('information'):
                    await self.env_interface.process_information(
                        outcome.get('information'),
                        source=outcome.get('source', {}),
                        domain=domain
                    )
                    
        return {
            'status': 'success',
            'query_id': query_id,
            'recorded': True
        }
    
    async def _refresh_information_needs(self, context):
        """Refresh information needs if needed."""
        current_time = time.time()
        
        # Check if we need to refresh
        if current_time - self.last_needs_refresh < self.min_refresh_interval:
            return
            
        # Trigger active sampling to update its uncertainty map
        for domain in self.domain_sources.keys():
            await self.active_sampling._update_uncertainty_map(domain)
            
        self.last_needs_refresh = current_time
    
    async def _register_initial_sources(self):
        """Register initial information sources."""
        # In a real implementation, this would load source configurations
        # from configuration or a database
        
        initial_sources = [
            {
                'id': 'scientific_journals',
                'domains': ['medical', 'scientific'],
                'credibility': 0.9,
                'access_details': {'type': 'api', 'rate_limit': 10}
            },
            {
                'id': 'financial_data',
                'domains': ['financial', 'economic'],
                'credibility': 0.85,
                'access_details': {'type': 'database', 'rate_limit': 20}
            },
            {
                'id': 'news_aggregator',
                'domains': ['news', 'current_events'],
                'credibility': 0.7,
                'access_details': {'type': 'api', 'rate_limit': 30}
            }
        ]
        
        for source in initial_sources:
            await self.register_information_source(source)
    
    async def _execute_query_for_need(self, need, context):
        """Execute a query for an information need."""
        domain = need.get('domain', 'general')
        region = need.get('pure_region', 'general')
        
        # Select appropriate source for this domain
        source_id = await self._select_best_source(domain, need)
        if not source_id:
            return None
            
        # Create query
        query_components = need.get('query_components', {})
        
        # Generate query ID
        query_id = f"query_{int(time.time())}_{domain}_{region}_{source_id}"
        
        # Add to active queries
        self.active_queries[query_id] = {
            'query_id': query_id,
            'domain': domain,
            'region': region,
            'source_id': source_id,
            'expected_gain': need.get('expected_gain', 0),
            'query_components': query_components,
            'start_time': time.time(),
            'timeout': time.time() + self.query_timeout
        }
        
        # Execute query through environmental interface
        try:
            query_result = await self.env_interface.execute_query(
                source_id=source_id,
                query_components=query_components,
                domain=domain,
                context=context
            )
            
            # Update active query with initial result
            self.active_queries[query_id]['initial_result'] = query_result
            
            return {
                'query_id': query_id,
                'domain': domain,
                'region': region,
                'source_id': source_id,
                'status': 'initiated',
                'expected_completion': time.time() + query_result.get('estimated_time', self.query_timeout)
            }
            
        except Exception as e:
            # Remove from active queries on error
            if query_id in self.active_queries:
                del self.active_queries[query_id]
                
            self.logger.error(f"Error executing query: {str(e)}")
            
            return {
                'query_id': query_id,
                'domain': domain,
                'region': region,
                'source_id': source_id,
                'status': 'error',
                'error': str(e)
            }
    
    async def _select_best_source(self, domain, need):
        """Select the best information source for a need."""
        # Get sources for this domain
        domain_source_ids = self.domain_sources.get(domain, set())
        
        # If no specific sources, try general
        if not domain_source_ids:
            domain_source_ids = self.domain_sources.get('general', set())
            
        if not domain_source_ids:
            return None
            
        # Calculate source scores
        source_scores = []
        
        for source_id in domain_source_ids:
            # Calculate performance score
            performance = self.source_performance.get(source_id, [])
            if performance:
                recent_performance = performance[-10:]
                success_rate = sum(1 for p in recent_performance if p.get('success', False)) / len(recent_performance)
                avg_relevance = sum(p.get('relevance', 0) for p in recent_performance) / len(recent_performance)
                
                performance_score = (success_rate * 0.6) + (avg_relevance * 0.4)
            else:
                performance_score = 0.5  # Default
                
            # Check current load (avoid overloading a source)
            current_load = sum(1 for q in self.active_queries.values() if q.get('source_id') == source_id)
            load_factor = max(0.1, 1.0 - (current_load * 0.2))  # Reduce score based on load
            
            # Calculate combined score
            score = performance_score * load_factor
            
            source_scores.append((source_id, score))
            
        # Select best scoring source
        if source_scores:
            return max(source_scores, key=lambda x: x[1])[0]
        else:
            return next(iter(domain_source_ids)) if domain_source_ids else None
    
    async def _check_active_queries(self):
        """Check status of active queries and handle timeouts."""
        current_time = time.time()
        completed = []
        
        # Check each active query
        for query_id, query_info in list(self.active_queries.items()):
            # Check if timed out
            if current_time > query_info.get('timeout', 0):
                # Handle timeout
                query_info['outcome'] = {'status': 'timeout', 'success': False}
                query_info['completion_time'] = current_time
                
                # Move to history
                self.query_history.append(query_info)
                del self.active_queries[query_id]
                
                completed.append({
                    'query_id': query_id,
                    'status': 'timeout',
                    'domain': query_info.get('domain'),
                    'region': query_info.get('region')
                })
                
                continue
                
            # Check with environmental interface for completion
            try:
                status = await self.env_interface.check_query_status(query_id)
                
                if status.get('status') == 'completed':
                    # Query completed
                    query_info['outcome'] = status
                    query_info['completion_time'] = current_time
                    
                    # Extract information
                    information = status.get('information')
                    if information:
                        # Cache the information
                        cache_key = f"{query_info.get('domain')}:{query_info.get('region')}"
                        self.information_cache[cache_key] = {
                            'information': information,
                            'timestamp': current_time,
                            'source': status.get('source')
                        }
                    
                    # Move to history
                    self.query_history.append(query_info)
                    del self.active_queries[query_id]
                    
                    completed.append({
                        'query_id': query_id,
                        'status': 'completed',
                        'success': status.get('success', False),
                        'domain': query_info.get('domain'),
                        'region': query_info.get('region'),
                        'has_information': information is not None
                    })
            except Exception as e:
                self.logger.error(f"Error checking query status: {str(e)}")
                
        return completed
    
    def get_source_statistics(self):
        """Get statistics about information sources."""
        stats = {}
        
        for source_id, performance in self.source_performance.items():
            if not performance:
                continue
                
            recent = performance[-10:]
            
            source_stats = {
                'query_count': len(performance),
                'success_rate': sum(1 for p in recent if p.get('success', False)) / max(1, len(recent)),
                'avg_relevance': sum(p.get('relevance', 0) for p in recent) / max(1, len(recent)),
                'last_query': performance[-1]['timestamp'] if performance else None
            }
            
            stats[source_id] = source_stats
            
        return stats
    
    def get_seeking_statistics(self):
        """Get statistics about information seeking performance."""
        stats = {
            'active_queries': len(self.active_queries),
            'query_history': len(self.query_history),
            'cache_size': len(self.information_cache),
            'sources': {domain: list(sources) for domain, sources in self.domain_sources.items()}
        }
        
        # Calculate success metrics
        if self.query_history:
            recent_history = self.query_history[-50:]
            
            stats['metrics'] = {
                'success_rate': sum(1 for q in recent_history 
                                  if q.get('outcome', {}).get('success', False)) / len(recent_history),
                'avg_completion_time': sum((q.get('completion_time', 0) - q.get('start_time', 0)) 
                                         for q in recent_history) / len(recent_history),
                'timeout_rate': sum(1 for q in recent_history 
                                  if q.get('outcome', {}).get('status') == 'timeout') / len(recent_history)
            }
            
        return stats


# Example implementation of simplified Environmental Interface for demonstration
class SimplifiedEnvironmentalInterface:
    """Simplified implementation of Environmental Interface for demonstration."""
    def __init__(self):
        self.sources = {}
        self.queries = {}
        self.logger = logging.getLogger("ASF.EnvironmentalInterface")
        
    async def register_information_source(self, source_id, domains, credibility=0.5):
        """Register an information source."""
        self.sources[source_id] = {
            'id': source_id,
            'domains': domains,
            'credibility': credibility,
            'query_count': 0
        }
        
        return {'status': 'success', 'source_id': source_id}
        
    async def execute_query(self, source_id, query_components, domain, context=None):
        """Execute a query against an information source."""
        if source_id not in self.sources:
            raise ValueError(f"Unknown source: {source_id}")
            
        # Generate query ID
        query_id = f"q_{int(time.time())}_{source_id}_{domain}"
        
        # Create simulated query
        self.queries[query_id] = {
            'id': query_id,
            'source_id': source_id,
            'components': query_components,
            'domain': domain,
            'start_time': time.time(),
            'estimated_completion': time.time() + 5,  # 5 seconds simulated time
            'status': 'in_progress'
        }
        
        # Increment source query count
        self.sources[source_id]['query_count'] += 1
        
        return {
            'query_id': query_id,
            'estimated_time': 5,
            'status': 'initiated'
        }
        
    async def check_query_status(self, query_id):
        """Check status of a query."""
        if query_id not in self.queries:
            return {'status': 'unknown', 'query_id': query_id}
            
        query = self.queries[query_id]
        current_time = time.time()
        
        # Check if query should be complete
        if current_time >= query['estimated_completion']:
            # Simulate query completion
            source_id = query['source_id']
            source = self.sources.get(source_id, {})
            
            # Simulate 80% success rate
            success = np.random.random() < 0.8
            
            # Generate simulated information
            information = None
            if success:
                information = {
                    'content': f"Simulated information for {query['domain']}",
                    'metadata': {
                        'source': source_id,
                        'timestamp': current_time,
                        'query_components': query['components']
                    }
                }
                
            result = {
                'status': 'completed',
                'query_id': query_id,
                'success': success,
                'information': information,
                'source': {
                    'id': source_id,
                    'credibility': source.get('credibility', 0.5)
                },
                'completion_time': current_time
            }
            
            # Update query status
            query['status'] = 'completed'
            query['result'] = result
            
            return result
        else:
            # Still in progress
            completion_percentage = (current_time - query['start_time']) / (query['estimated_completion'] - query['start_time'])
            
            return {
                'status': 'in_progress',
                'query_id': query_id,
                'progress': min(0.99, completion_percentage),
                'estimated_completion': query['estimated_completion']
            }
            
    async def process_information(self, information, source, domain):
        """Process received information."""
        # In a real implementation, this would integrate the information
        # into the cognitive boundary layer
        
        self.logger.info(f"Processing information from {source.get('id')} in domain {domain}")
        
        # Simplified processing result
        return {
            'status': 'success',
            'processed': True,
            'domain': domain
        }


async def example_usage():
    """Example usage of the Adaptive Information Seeking Module."""
    # Create simplified components
    from core.active_sampling_strategy import ActiveSamplingStrategy
    
    # Create a simplified semantic network
    class SimplifiedSemanticNetwork:
        def __init__(self):
            self.nodes = {}
            
    # Create a simplified predictive processor
    class SimplifiedPredictiveProcessor:
        def get_precision(self, entity_id):
            return 1.0
            
    # Initialize components
    semantic_network = SimplifiedSemanticNetwork()
    predictive_processor = SimplifiedPredictiveProcessor()
    
    # Create active sampling strategy
    active_sampling = ActiveSamplingStrategy(
        semantic_network,
        predictive_processor,
        config={'exploration_rate': 0.4}
    )
    
    # Create environmental interface
    env_interface = SimplifiedEnvironmentalInterface()
    
    # Create adaptive information seeking module
    info_seeking = AdaptiveInformationSeekingModule(
        env_interface,
        active_sampling,
        config={'max_concurrent_queries': 3}
    )
    
    # Initialize the module
    await info_seeking.initialize()
    
    # Simulate searching for information
    print("Starting information seeking simulation...")
    
    # Run through multiple cycles of information seeking
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        
        # Context for seeking
        context = {
            'current_focus': 'medical advances',
            'priority_domains': ['medical', 'scientific']
        }
        
        # Seek information
        result = await info_seeking.seek_information(context)
        
        print(f"Seeking result: {result['status']}")
        print(f"New queries: {result['new_queries']}")
        print(f"Active queries: {result['active_queries']}")
        
        # Wait for queries to complete
        print("Waiting for queries to complete...")
        await asyncio.sleep(6)  # Slightly longer than the simulated query time
        
        # Check for completed queries
        check_result = await info_seeking._check_active_queries()
        print(f"Completed queries: {len(check_result)}")
        
        if check_result:
            # Get the first completed query
            completed = check_result[0]
            print(f"Sample completed query: {completed['query_id']}")
            print(f"Status: {completed['status']}")
            print(f"Domain: {completed['domain']}, Region: {completed.get('region')}")
            
        # Get statistics
        seeking_stats = info_seeking.get_seeking_statistics()
        source_stats = info_seeking.get_source_statistics()
        
        print("\nInformation Seeking Statistics:")
        if 'metrics' in seeking_stats:
            print(f"Success rate: {seeking_stats['metrics']['success_rate']:.2f}")
            print(f"Avg completion time: {seeking_stats['metrics']['avg_completion_time']:.2f}s")
        print(f"Active queries: {seeking_stats['active_queries']}")
        print(f"Query history: {seeking_stats['query_history']}")
        
        print("\nSource Statistics:")
        for source_id, stats in source_stats.items():
            print(f"Source {source_id}:")
            print(f"  Success rate: {stats['success_rate']:.2f}")
            print(f"  Avg relevance: {stats['avg_relevance']:.2f}")
            print(f"  Query count: {stats['query_count']}")
        
        # Show how active sampling is adapting
        sampling_stats = active_sampling.get_sampling_statistics()
        print("\nActive Sampling Statistics:")
        print(f"Exploration rate: {sampling_stats['parameters']['exploration_rate']:.2f}")
        if 'information_gain' in sampling_stats:
            print(f"Avg information gain: {sampling_stats['information_gain']['average_actual']:.2f}")
        
    # Demonstrate the full feedback loop
    print("\n--- Demonstrating Complete Feedback Loop ---")
    
    # New context
    context = {
        'current_focus': 'emerging financial technologies',
        'priority_domains': ['financial', 'technology']
    }
    
    # Seek information
    result = await info_seeking.seek_information(context)
    print(f"Initiated {result['new_queries']} new queries")
    
    # Wait for completion
    await asyncio.sleep(6)
    completed = await info_seeking._check_active_queries()
    
    if completed:
        # Process a completed query with full feedback loop
        query = completed[0]
        query_id = query['query_id']
        
        print(f"\nProcessing completed query: {query_id}")
        
        # Simulate finding valuable information
        outcome = {
            'success': True,
            'relevance': 0.85,
            'quality': 0.9,
            'information': {
                'content': "Significant findings about blockchain application in banking",
                'confidence': 0.85,
                'category': 'blockchain_banking',
                'timestamp': time.time()
            },
            'source': {
                'id': query.get('source_id', 'unknown'),
                'authority': 'financial_research',
                'reliability': 0.9
            }
        }
        
        # Report outcome to close the feedback loop
        report_result = await info_seeking.report_query_outcome(query_id, outcome)
        print(f"Outcome reported: {report_result['status']}")
        
        # Check if active sampling has updated
        updated_stats = active_sampling.get_sampling_statistics()
        print("\nUpdated Active Sampling Statistics:")
        print(f"Exploration rate: {updated_stats['parameters']['exploration_rate']:.2f}")
        
        # Demonstrate how this affects future information seeking
        print("\nNext cycle of information seeking after feedback:")
        next_result = await info_seeking.seek_information(context)
        print(f"New queries: {next_result['new_queries']}")
        print(f"Note any changes in queried regions based on previous findings")
        
    print("\nSimulation complete. The feedback loop has demonstrated how:")
    print("1. Information needs are identified through active sampling")
    print("2. Information is sought from appropriate sources")
    print("3. Results are evaluated and integrated")
    print("4. The system adapts its sampling strategy based on outcomes")
    print("5. Future information seeking is influenced by past results")
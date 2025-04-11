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
        
        self.max_concurrent_queries = self.config.get('max_concurrent_queries', 3)
        self.query_timeout = self.config.get('query_timeout', 60)  # seconds
        self.min_refresh_interval = self.config.get('min_refresh_interval', 300)  # seconds
        
        self.active_queries = {}  # Query ID -> query info
        self.query_history = []  # History of executed queries
        self.source_performance = defaultdict(list)  # Source ID -> performance history
        self.domain_sources = defaultdict(set)  # Domain -> set of source IDs
        self.last_needs_refresh = 0  # Last time information needs were refreshed
        self.information_cache = {}  # Cache of recently retrieved information
        
    async def initialize(self):
        Proactively seek information based on identified information needs.
        
        Args:
            context: Optional context information
            
        Returns:
            Results of information seeking
        Register a new information source for specific domains.
        
        Args:
            source_info: Source information including ID, domains, and access details
            
        Returns:
            Registration result
        Report outcome of an information query.
        
        Args:
            query_id: Query identifier
            outcome: Query outcome information
            
        Returns:
            Processing result
        current_time = time.time()
        
        if current_time - self.last_needs_refresh < self.min_refresh_interval:
            return
            
        for domain in self.domain_sources.keys():
            await self.active_sampling._update_uncertainty_map(domain)
            
        self.last_needs_refresh = current_time
    
    async def _register_initial_sources(self):
        domain = need.get('domain', 'general')
        region = need.get('pure_region', 'general')
        
        source_id = await self._select_best_source(domain, need)
        if not source_id:
            return None
            
        query_components = need.get('query_components', {})
        
        query_id = f"query_{int(time.time())}_{domain}_{region}_{source_id}"
        
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
        
        try:
            query_result = await self.env_interface.execute_query(
                source_id=source_id,
                query_components=query_components,
                domain=domain,
                context=context
            )
            
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
        current_time = time.time()
        completed = []
        
        for query_id, query_info in list(self.active_queries.items()):
            if current_time > query_info.get('timeout', 0):
                query_info['outcome'] = {'status': 'timeout', 'success': False}
                query_info['completion_time'] = current_time
                
                self.query_history.append(query_info)
                del self.active_queries[query_id]
                
                completed.append({
                    'query_id': query_id,
                    'status': 'timeout',
                    'domain': query_info.get('domain'),
                    'region': query_info.get('region')
                })
                
                continue
                
            try:
                status = await self.env_interface.check_query_status(query_id)
                
                if status.get('status') == 'completed':
                    query_info['outcome'] = status
                    query_info['completion_time'] = current_time
                    
                    information = status.get('information')
                    if information:
                        cache_key = f"{query_info.get('domain')}:{query_info.get('region')}"
                        self.information_cache[cache_key] = {
                            'information': information,
                            'timestamp': current_time,
                            'source': status.get('source')
                        }
                    
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
        if query_id not in self.queries:
            return {'status': 'unknown', 'query_id': query_id}
            
        query = self.queries[query_id]
        current_time = time.time()
        
        if current_time >= query['estimated_completion']:
            source_id = query['source_id']
            source = self.sources.get(source_id, {})
            
            success = np.random.random() < 0.8
            
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
            
            query['status'] = 'completed'
            query['result'] = result
            
            return result
        else:
            completion_percentage = (current_time - query['start_time']) / (query['estimated_completion'] - query['start_time'])
            
            return {
                'status': 'in_progress',
                'query_id': query_id,
                'progress': min(0.99, completion_percentage),
                'estimated_completion': query['estimated_completion']
            }
            
    async def process_information(self, information, source, domain):
    from core.active_sampling_strategy import ActiveSamplingStrategy
    
    class SimplifiedSemanticNetwork:
        def __init__(self):
            self.nodes = {}
            
    class SimplifiedPredictiveProcessor:
        def get_precision(self, entity_id):
            return 1.0
            
    semantic_network = SimplifiedSemanticNetwork()
    predictive_processor = SimplifiedPredictiveProcessor()
    
    active_sampling = ActiveSamplingStrategy(
        semantic_network,
        predictive_processor,
        config={'exploration_rate': 0.4}
    )
    
    env_interface = SimplifiedEnvironmentalInterface()
    
    info_seeking = AdaptiveInformationSeekingModule(
        env_interface,
        active_sampling,
        config={'max_concurrent_queries': 3}
    )
    
    await info_seeking.initialize()
    
    print("Starting information seeking simulation...")
    
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        
        context = {
            'current_focus': 'medical advances',
            'priority_domains': ['medical', 'scientific']
        }
        
        result = await info_seeking.seek_information(context)
        
        print(f"Seeking result: {result['status']}")
        print(f"New queries: {result['new_queries']}")
        print(f"Active queries: {result['active_queries']}")
        
        print("Waiting for queries to complete...")
        await asyncio.sleep(6)  # Slightly longer than the simulated query time
        
        check_result = await info_seeking._check_active_queries()
        print(f"Completed queries: {len(check_result)}")
        
        if check_result:
            completed = check_result[0]
            print(f"Sample completed query: {completed['query_id']}")
            print(f"Status: {completed['status']}")
            print(f"Domain: {completed['domain']}, Region: {completed.get('region')}")
            
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
        
        sampling_stats = active_sampling.get_sampling_statistics()
        print("\nActive Sampling Statistics:")
        print(f"Exploration rate: {sampling_stats['parameters']['exploration_rate']:.2f}")
        if 'information_gain' in sampling_stats:
            print(f"Avg information gain: {sampling_stats['information_gain']['average_actual']:.2f}")
        
    print("\n--- Demonstrating Complete Feedback Loop ---")
    
    context = {
        'current_focus': 'emerging financial technologies',
        'priority_domains': ['financial', 'technology']
    }
    
    result = await info_seeking.seek_information(context)
    print(f"Initiated {result['new_queries']} new queries")
    
    await asyncio.sleep(6)
    completed = await info_seeking._check_active_queries()
    
    if completed:
        query = completed[0]
        query_id = query['query_id']
        
        print(f"\nProcessing completed query: {query_id}")
        
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
        
        report_result = await info_seeking.report_query_outcome(query_id, outcome)
        print(f"Outcome reported: {report_result['status']}")
        
        updated_stats = active_sampling.get_sampling_statistics()
        print("\nUpdated Active Sampling Statistics:")
        print(f"Exploration rate: {updated_stats['parameters']['exploration_rate']:.2f}")
        
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
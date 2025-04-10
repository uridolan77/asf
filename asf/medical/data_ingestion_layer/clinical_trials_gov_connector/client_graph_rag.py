import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

# Import the GraphRAG implementation
from medical_graphrag import ClinicalTrialsGraphRAG

# Import the clinical trials connector
sys.path.append('./asf/medical/data_ingestion_layer/clinical_trials_gov_connector')
from client import ClinicalTrialsClient
from client2 import ClinicalTrialsConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('clinical_trials_graphrag_client')

class ClinicalTrialsGraphRAGClient:
    """
    Client for interacting with the ClinicalTrials.gov API and GraphRAG system.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, auto_index: bool = True):
        """
        Initialize the client.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            auto_index: Whether to enable automatic indexing
        """
        # Set environment variables
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            # Try to get from environment
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if not openai_api_key:
            logger.warning("No OpenAI API key provided. Embeddings will not work.")
        
        # Initialize clients
        self.ts_client = ClinicalTrialsClient()  # TypeScript-style client
        self.py_client = ClinicalTrialsConnector()  # Python-style client
        self.graphrag = ClinicalTrialsGraphRAG(api_key=openai_api_key, auto_index=auto_index)
        
        logger.info("Initialized ClinicalTrials GraphRAG Client")
    
    def fetch_and_ingest_trials(self, 
                              query: str, 
                              max_results: int = 50,
                              register_for_updates: bool = True,
                              update_interval: int = 86400) -> List[int]:
        """
        Fetch clinical trials matching the query and ingest them into the GraphRAG system.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            register_for_updates: Whether to register for automatic updates
            update_interval: How often to check for updates (in seconds)
            
        Returns:
            List of document IDs
        """
        logger.info(f"Fetching and ingesting trials for query: '{query}'")
        
        source_id = f"api_query:{query.replace(' ', '_')[:50]}"
        
        # Use the API client to fetch and ingest trials
        doc_ids = self.graphrag.ingest_from_api(
            api_client=self.ts_client,
            query=query,
            max_results=max_results,
            source_id=source_id,
            register_for_updates=register_for_updates,
            update_interval=update_interval
        )
        
        logger.info(f"Ingested {len(doc_ids)} trials for query: '{query}'")
        
        return doc_ids
    
    def search_trials(self, 
                     query: str, 
                     use_graph: bool = True, 
                     k: int = 5,
                     rewrite_query: bool = True) -> List[Dict[str, Any]]:
        """
        Search for clinical trials.
        
        Args:
            query: Search query
            use_graph: Whether to use graph-based retrieval
            k: Number of initial results to retrieve
            rewrite_query: Whether to rewrite the query for better results
            
        Returns:
            List of matching documents
        """
        logger.info(f"Searching for trials with query: '{query}', use_graph={use_graph}")
        
        results = self.graphrag.search(
            query=query,
            use_graph=use_graph,
            k=k,
            rewrite_query=rewrite_query
        )
        
        logger.info(f"Found {len(results)} matching trials")
        
        return results
    
    def get_answer(self, 
                  query: str, 
                  use_graph: bool = True, 
                  k: int = 5) -> Dict[str, Any]:
        """
        Generate an answer to a query about clinical trials.
        
        Args:
            query: User query
            use_graph: Whether to use graph-based retrieval
            k: Number of initial results to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Generating answer for query: '{query}'")
        
        response = self.graphrag.generate_response(
            query=query,
            use_graph=use_graph,
            k=k
        )
        
        logger.info(f"Generated answer with {len(response['sources'])} sources")
        
        return response
    
    def visualize_graph(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Generate data for visualizing the graph of documents and entities.
        
        Args:
            document_ids: List of document IDs to include
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        logger.info(f"Generating graph visualization for {len(document_ids)} documents")
        
        visualization_data = self.graphrag.visualize_graph(document_ids)
        
        logger.info(f"Generated visualization with {len(visualization_data['nodes'])} nodes and {len(visualization_data['edges'])} edges")
        
        return visualization_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the GraphRAG system.
        
        Returns:
            Dictionary of statistics
        """
        logger.info("Fetching system statistics")
        
        statistics = self.graphrag.get_statistics()
        
        logger.info(f"System has {statistics['total_documents']} documents and {statistics['total_entities']} entities")
        
        return statistics
    
    def force_reindex(self, source_id: str) -> bool:
        """
        Force reindexing of a data source.
        
        Args:
            source_id: ID of the data source to reindex
            
        Returns:
            Success flag
        """
        if source_id not in self.graphrag.auto_indexers:
            logger.error(f"Data source '{source_id}' not found or not registered for auto-indexing")
            return False
            
        logger.info(f"Forcing reindex of data source '{source_id}'")
        
        success = self.graphrag.auto_indexers[source_id].force_run()
        
        if success:
            logger.info(f"Successfully reindexed data source '{source_id}'")
        else:
            logger.error(f"Failed to reindex data source '{source_id}'")
            
        return success

def main():
    """
    Main function for CLI operation
    """
    parser = argparse.ArgumentParser(description='Clinical Trials GraphRAG Client')
    
    # Setup subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest clinical trials')
    ingest_parser.add_argument('--query', type=str, required=True, help='Search query')
    ingest_parser.add_argument('--max', type=int, default=50, help='Maximum results to ingest')
    ingest_parser.add_argument('--no-auto-update', action='store_true', help='Disable automatic updates')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for clinical trials')
    search_parser.add_argument('--query', type=str, required=True, help='Search query')
    search_parser.add_argument('--no-graph', action='store_true', help='Disable graph-based retrieval')
    search_parser.add_argument('--k', type=int, default=5, help='Number of initial results')
    search_parser.add_argument('--no-rewrite', action='store_true', help='Disable query rewriting')
    
    # Answer command
    answer_parser = subparsers.add_parser('answer', help='Generate an answer to a query')
    answer_parser.add_argument('--query', type=str, required=True, help='User query')
    answer_parser.add_argument('--no-graph', action='store_true', help='Disable graph-based retrieval')
    answer_parser.add_argument('--k', type=int, default=5, help='Number of initial results')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Get system statistics')
    
    # Reindex command
    reindex_parser = subparsers.add_parser('reindex', help='Force reindexing of a data source')
    reindex_parser.add_argument('--source-id', type=str, required=True, help='Data source ID')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = ClinicalTrialsGraphRAGClient(openai_api_key=openai_api_key)
    
    # Execute command
    if args.command == 'ingest':
        doc_ids = client.fetch_and_ingest_trials(
            query=args.query,
            max_results=args.max,
            register_for_updates=not args.no_auto_update
        )
        print(f"Ingested {len(doc_ids)} trials")
        
    elif args.command == 'search':
        results = client.search_trials(
            query=args.query,
            use_graph=not args.no_graph,
            k=args.k,
            rewrite_query=not args.no_rewrite
        )
        
        print(f"\nSearch results for: '{args.query}'")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('title')} (NCT ID: {result.get('metadata', {}).get('nct_id', '')})")
            if 'connection' in result:
                connection = result['connection']
                if isinstance(connection, dict):
                    print(f"   Connected via: {connection.get('entity_type')}:{connection.get('entity_value')}")
                else:
                    print(f"   Connection: {connection}")
                    
    elif args.command == 'answer':
        response = client.get_answer(
            query=args.query,
            use_graph=not args.no_graph,
            k=args.k
        )
        
        print(f"\nAnswer: {response['response']}")
        print("\nSources:")
        for i, source in enumerate(response['sources'], 1):
            print(f"{i}. {source.get('title')} (NCT ID: {source.get('nct_id')})")
            print(f"   Relevance score: {source.get('relevance_score', 0):.2f}")
            
    elif args.command == 'stats':
        statistics = client.get_statistics()
        
        print("\nSystem Statistics:")
        print(f"Total documents: {statistics['total_documents']}")
        print(f"Total entities: {statistics['total_entities']}")
        print("\nEntity counts:")
        for entity_type, count in statistics['entity_counts'].items():
            print(f"  {entity_type}: {count}")
        print("\nDocument types:")
        for doc_type, count in statistics['document_types'].items():
            print(f"  {doc_type}: {count}")
        print("\nData sources:")
        for source_id, info in statistics['data_sources'].items():
            print(f"  {source_id}: {info['type']} (Last indexed: {info['last_indexed']})")
            
    elif args.command == 'reindex':
        success = client.force_reindex(args.source_id)
        if success:
            print(f"Successfully reindexed data source '{args.source_id}'")
        else:
            print(f"Failed to reindex data source '{args.source_id}'")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional

# Import the GraphRAG implementation
from medical_graphrag import ClinicalTrialsGraphRAG

# Import the clinical trials connector
sys.path.append('./asf/medical/data_ingestion_layer/clinical_trials_gov_connector')
from client import ClinicalTrialsClient
from search_query_builder import SearchQueryBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('clinical_trials_graphrag_client')

class ClinicalTrialsGraphRAGClient:
    """
    Client for interacting with the ClinicalTrials.gov API and GraphRAG system.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
        """
        # Set environment variables
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            # Try to get from environment
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if not openai_api_key:
            logger.warning("No OpenAI API key provided. Embeddings will not work.")
        
        # Initialize clients
        self.api_client = ClinicalTrialsClient()
        self.graphrag = ClinicalTrialsGraphRAG(api_key=openai_api_key)
        
        logger.info("Initialized ClinicalTrials GraphRAG Client")
    
    def fetch_and_ingest_trials(self, 
                              query: str, 
                              max_results: int = 50) -> List[int]:
        """
        Fetch clinical trials matching the query and ingest them into the GraphRAG system.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            
        Returns:
            List of document IDs
        """
        logger.info(f"Fetching and ingesting trials for query: '{query}'")
        
        # Use the query builder to construct the search query
        builder = SearchQueryBuilder()
        builder.condition(query)
        
        # Search for studies using the query builder
        search_results = self.api_client.searchStudies(builder, {
            'pageSize': max_results
        })
        
        # Check if search was successful
        if not search_results or 'studies' not in search_results:
            logger.error("Search failed or returned no results")
            return []
        
        studies = search_results.get('studies', [])
        logger.info(f"Found {len(studies)} studies")
        
        # Extract study IDs
        study_ids = []
        for study in studies:
            # Extract NCT ID from the study
            nct_id = None
            if 'protocolSection' in study and 'identificationModule' in study['protocolSection']:
                nct_id = study['protocolSection']['identificationModule'].get('nctId')
            
            if nct_id:
                study_ids.append(nct_id)
        
        # Fetch details and ingest each study
        doc_ids = []
        for nct_id in study_ids:
            try:
                # Get study details
                study = self.api_client.getStudy(nct_id)
                
                # Extract relevant information
                protocol = study.get('protocolSection', {})
                
                trial_data = {
                    'nct_id': nct_id,
                    'brief_title': protocol.get('identificationModule', {}).get('briefTitle', ''),
                    'official_title': protocol.get('identificationModule', {}).get('officialTitle', ''),
                    'status': protocol.get('statusModule', {}).get('overallStatus', ''),
                    'phase': protocol.get('designModule', {}).get('phases', []),
                    'study_type': protocol.get('designModule', {}).get('studyType', ''),
                    'brief_summary': protocol.get('descriptionModule', {}).get('briefSummary', ''),
                    'detailed_description': protocol.get('descriptionModule', {}).get('detailedDescription', ''),
                    'condition': protocol.get('conditionsModule', {}).get('conditions', []),
                    'intervention': [
                        i.get('name', '') 
                        for i in protocol.get('armsInterventionsModule', {}).get('interventions', [])
                    ],
                    'eligibility': protocol.get('eligibilityModule', {}).get('eligibilityCriteria', ''),
                    'location': [
                        {
                            'facility': loc.get('facility', ''),
                            'city': loc.get('city', ''),
                            'state': loc.get('state', ''),
                            'country': loc.get('country', '')
                        }
                        for loc in
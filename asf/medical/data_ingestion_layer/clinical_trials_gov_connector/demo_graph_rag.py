"""
Module description.

This module provides functionality for...
"""
import os
import sys
import logging

from clinical_trials_client import ClinicalTrialsGraphRAGClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("graphrag_demo")

def display_header(title: str):
    """Display a formatted header.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def run_demo():
    """Run the GraphRAG demo.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    display_header("Initializing GraphRAG Client")
    client = ClinicalTrialsGraphRAGClient(openai_api_key=api_key)
    print("Client initialized successfully.")
    
    display_header("Ingesting COVID-19 Clinical Trials")
    covid_doc_ids = client.fetch_and_ingest_trials(
        query="COVID-19 treatment",
        max_results=20,
        register_for_updates=True,
        update_interval=86400  # Daily updates
    )
    print(f"Ingested {len(covid_doc_ids)} COVID-19 clinical trials.")
    
    display_header("Ingesting Diabetes Clinical Trials")
    diabetes_doc_ids = client.fetch_and_ingest_trials(
        query="Type 2 Diabetes treatment",
        max_results=20,
        register_for_updates=True,
        update_interval=86400  # Daily updates
    )
    print(f"Ingested {len(diabetes_doc_ids)} diabetes clinical trials.")
    
    display_header("System Statistics")
    stats = client.get_statistics()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Entity types: {', '.join(stats['entity_counts'].keys())}")
    print(f"Data sources: {len(stats['data_sources'])}")
    
    display_header("Searching: COVID-19 Treatment Efficacy")
    
    vector_results = client.search_trials(
        query="What is the efficacy of remdesivir for COVID-19?",
        use_graph=False,
        k=3,
        rewrite_query=True
    )
    
    print("\nVECTOR SEARCH RESULTS:")
    for i, result in enumerate(vector_results, 1):
        print(f"{i}. {result.get('title')} (NCT ID: {result.get('metadata', {}).get('nct_id', '')})")
        print(f"   Relevance: {1.0 - result.get('distance', 0):.2f}")
    
    graph_results = client.search_trials(
        query="What is the efficacy of remdesivir for COVID-19?",
        use_graph=True,
        k=3,
        rewrite_query=True
    )
    
    print("\nGRAPH SEARCH RESULTS:")
    for i, result in enumerate(graph_results, 1):
        print(f"{i}. {result.get('title')} (NCT ID: {result.get('metadata', {}).get('nct_id', '')})")
        if 'connection' in result and isinstance(result['connection'], dict):
            print(f"   Connected via: {result['connection'].get('entity_type')}:{result['connection'].get('entity_value')}")
    
    display_header("Answering Questions with GraphRAG")
    
    questions = [
        "What are the common side effects of remdesivir in COVID-19 patients?",
        "How does remdesivir compare to other treatments for COVID-19?",
        "What inclusion criteria are typically used in COVID-19 clinical trials?",
        "What are the main treatment approaches for Type 2 Diabetes being studied?",
        "Are there any clinical trials studying both COVID-19 and diabetes?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        
        response = client.get_answer(
            query=question,
            use_graph=True,
            k=5
        )
        
        print(f"Answer: {response['response']}")
        print("Sources:")
        for j, source in enumerate(response['sources'][:3], 1):  # Show only top 3 sources
            print(f"  {j}. {source.get('title')} (NCT ID: {source.get('nct_id')})")
    
    display_header("Demonstrating GraphRAG Advantage")
    
    print("This example shows how GraphRAG can find related information through entity connections")
    print("that wouldn't be found with traditional vector search.")
    
    query = "What treatments are being studied for COVID-19 patients with diabetes?"
    
    print("\nTraditional Vector Search:")
    vector_answer = client.get_answer(
        query=query,
        use_graph=False,
        k=5
    )
    print(f"Answer: {vector_answer['response']}")
    
    print("\nGraph-Based RAG:")
    graph_answer = client.get_answer(
        query=query,
        use_graph=True,
        k=5
    )
    print(f"Answer: {graph_answer['response']}")
    
    display_header("Demo Complete")
    print("The GraphRAG demonstration is complete.")
    print("The system will continue to automatically index and update the clinical trials data.")

if __name__ == "__main__":
    run_demo()
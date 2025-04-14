#!/usr/bin/env python3
"""
Enhanced Medical Research Synthesizer Demo

This script demonstrates the enhanced capabilities of the Medical Research Synthesizer:

1. NLP Claim Extraction: Automatic extraction of claims from medical text
2. Explainable AI: Visual explanations for contradiction detection
3. Knowledge Graph with Ontology Integration: Enhanced retrieval using SNOMED CT and MeSH

Usage:
    python enhanced_research_demo.py --feature [claim|xai|kg|all]
"""

import os
import sys
import argparse
import asyncio
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asf.medical.ml.models.claim_extraction_model import ClaimExtractionModel
from asf.medical.ml.models.contradiction_explainer import ContradictionExplainer, ContradictionExplanation
from asf.medical.graph.graph_rag import GraphRAG
from asf.medical.graph.ontology_integration import OntologyIntegrationService, GraphRAGOntologyEnhancer


async def demo_claim_extraction():
    """Demonstrate automatic claim extraction from medical abstracts."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Automatic Claim Extraction from Medical Text")
    print("="*80)

    # Sample medical abstracts
    abstracts = [
        """
        BACKGROUND: Remdesivir is an RNA polymerase inhibitor with potent antiviral activity against a range of RNA viruses.
        METHODS: We conducted a randomized, controlled trial of intravenous remdesivir in adults hospitalized with COVID-19 with evidence of lower respiratory tract involvement. Patients were randomly assigned to receive either remdesivir (200 mg loading dose on day 1, followed by 100 mg daily for up to 9 additional days) or placebo for up to 10 days. The primary outcome was time to recovery.
        RESULTS: A total of 1062 patients underwent randomization. The data and safety monitoring board recommended early unblinding of the results on the basis of findings from an analysis that showed shortened time to recovery in the remdesivir group. Preliminary results from the 1059 patients (538 assigned to remdesivir and 521 to placebo) with data available after randomization indicated that those who received remdesivir had a median recovery time of 11 days (95% confidence interval [CI], 9 to 12), as compared with 15 days (95% CI, 13 to 19) in those who received placebo (rate ratio for recovery, 1.32; 95% CI, 1.12 to 1.55; P<0.001).
        CONCLUSIONS: Remdesivir was superior to placebo in shortening the time to recovery in adults hospitalized with COVID-19 and evidence of lower respiratory tract infection.
        """,
        """
        BACKGROUND: Molnupiravir is an oral, small-molecule antiviral prodrug that is active against SARS-CoV-2. The efficacy and safety of early treatment with molnupiravir in high-risk, unvaccinated adult outpatients with mild-to-moderate Covid-19 are unclear.
        METHODS: In this randomized, double-blind trial, we evaluated molnupiravir in unvaccinated adults with mild-to-moderate Covid-19 who were at risk for severe disease. Patients were randomly assigned to receive 800 mg of molnupiravir or placebo twice daily for 5 days. The primary efficacy end point was the incidence of hospitalization or death at day 29.
        RESULTS: A total of 1433 patients underwent randomization; 716 were assigned to receive molnupiravir and 717 to receive placebo. The risk of hospitalization or death through day 29 was lower with molnupiravir (6.8%, 48 of 709 patients) than with placebo (9.7%, 68 of 699 patients), for a difference of −3.0 percentage points (95% confidence interval [CI], −5.9 to −0.1); relative risk reduction, 30% (95% CI, 1 to 51).
        CONCLUSIONS: Early treatment with molnupiravir reduced the risk of hospitalization or death in at-risk, unvaccinated adults with Covid-19.
        """
    ]

    try:
        # Initialize the claim extraction model
        print("\nInitializing claim extraction model...")
        model = ClaimExtractionModel()

        # Process each abstract
        for i, abstract in enumerate(abstracts):
            print(f"\n\nAbstract {i+1}:")
            print("-" * 40)
            print(abstract.strip())
            print("-" * 40)
            
            # Extract claims
            print("\nExtracting claims...")
            claims = await model.extract_claims(abstract)
            
            # Display claims
            for j, claim in enumerate(claims):
                print(f"\nClaim {j+1}: {claim.text}")
                print(f"  - Type: {claim.claim_type}")
                print(f"  - Confidence: {claim.confidence:.2%}")
                print(f"  - Evidence Section: {claim.source_section}")
                
                if hasattr(claim, 'entities') and claim.entities:
                    print("  - Entities:")
                    for entity in claim.entities:
                        print(f"    * {entity['text']} ({entity['type']})")

        print("\nClaim extraction demonstration complete!")

    except Exception as e:
        print(f"\nError in claim extraction demo: {str(e)}")
        import traceback
        traceback.print_exc()


async def demo_explainable_ai():
    """Demonstrate explainable AI for contradiction detection."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Explainable AI for Contradiction Detection")
    print("="*80)

    # Sample contradictory claims
    contradiction_pairs = [
        {
            "claim1": "Remdesivir significantly reduces mortality in patients with severe COVID-19.",
            "claim2": "Remdesivir showed no significant effect on mortality in patients with severe COVID-19.",
            "contradiction_type": "direct",
            "confidence": 0.92
        },
        {
            "claim1": "Hydroxychloroquine is effective for treating mild COVID-19 in outpatients.",
            "claim2": "A randomized controlled trial found that hydroxychloroquine did not improve outcomes in mild COVID-19 cases.",
            "contradiction_type": "methodological",
            "confidence": 0.85
        },
        {
            "claim1": "Children under 12 with COVID-19 rarely develop severe symptoms.",
            "claim2": "Approximately 30% of children hospitalized with COVID-19 required ICU admission.",
            "contradiction_type": "population",
            "confidence": 0.78
        }
    ]

    try:
        # Initialize the explainer
        print("\nInitializing contradiction explainer...")
        explainer = ContradictionExplainer()

        # Process each contradiction pair
        for i, pair in enumerate(contradiction_pairs):
            print(f"\n\nContradiction Example {i+1}:")
            print("-" * 40)
            print(f"Claim 1: {pair['claim1']}")
            print(f"Claim 2: {pair['claim2']}")
            print(f"Contradiction Type: {pair['contradiction_type']}")
            print(f"Initial Confidence: {pair['confidence']:.2%}")
            print("-" * 40)
            
            # Generate explanation
            print("\nGenerating explanation...")
            explanation = explainer.explain(
                pair["claim1"],
                pair["claim2"],
                pair["contradiction_type"],
                pair["confidence"]
            )
            
            # Display explanation
            print(f"\nExplanation: {explanation.explanation_text}")
            print(f"Confidence: {explanation.confidence:.2%}")
            print(f"Uncertainty: {explanation.uncertainty:.2%}" if explanation.uncertainty is not None else "Uncertainty: N/A")
            
            # Generate visual explanation
            print("\nGenerating visual explanation...")
            vis_path = f"contradiction_explanation_{i+1}.png"
            explanation.generate_visual(vis_path)
            print(f"Visual explanation saved to {vis_path}")
            
            # Generate HTML explanation
            html_path = f"contradiction_explanation_{i+1}.html"
            with open(html_path, 'w') as f:
                f.write(explanation.generate_html_explanation())
            print(f"HTML explanation saved to {html_path}")
            
            # Show alternative outcomes if available
            if explanation.alternative_outcomes:
                print("\nAlternative classifications:")
                for outcome, prob in sorted(
                    explanation.alternative_outcomes.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    print(f"  - {outcome}: {prob:.2%}")

        print("\nExplainable AI demonstration complete!")

    except Exception as e:
        print(f"\nError in explainable AI demo: {str(e)}")
        import traceback
        traceback.print_exc()


async def demo_knowledge_graph_ontology():
    """Demonstrate knowledge graph with ontology integration."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Knowledge Graph with Ontology Integration")
    print("="*80)

    # Sample queries
    queries = [
        "COVID-19 treatment efficacy",
        "myocardial infarction risk factors in diabetic patients",
        "beta-lactam antibiotics for respiratory infections"
    ]

    try:
        # Initialize the GraphRAG service
        print("\nInitializing GraphRAG service with ontology integration...")
        graph_rag = GraphRAG()
        
        # Wait for ontology services to initialize
        print("Waiting for ontology services to initialize...")
        await asyncio.sleep(2)  # Give some time for async initialization
        
        # Search for medical concepts in ontologies
        print("\nSearching ontologies for medical concepts:")
        for query in queries:
            print(f"\nQuery: {query}")
            
            # Search in MeSH
            mesh_concepts = await graph_rag.search_ontology(
                query, 
                source_ontologies=["MESH"]
            )
            if mesh_concepts:
                print(f"  MeSH concepts found ({len(mesh_concepts)}):")
                for concept in mesh_concepts[:3]:  # Show top 3
                    print(f"    - {concept['text']} (ID: {concept['concept_id']}, Confidence: {concept['confidence']:.2%})")
            else:
                print("  No MeSH concepts found")
            
            # Search with cross-ontology mapping
            mapped_concepts = await graph_rag.search_ontology(
                query, 
                source_ontologies=["MESH"], 
                target_ontologies=["SNOMED"]
            )
            mesh_to_snomed = [c for c in mapped_concepts if c.get("mapped_from") is not None]
            if mesh_to_snomed:
                print(f"  MESH→SNOMED mappings found ({len(mesh_to_snomed)}):")
                for concept in mesh_to_snomed[:3]:  # Show top 3
                    print(f"    - {concept['text']} (SNOMED ID: {concept['concept_id']}, " +
                          f"mapped from MESH ID: {concept['mapped_from']['concept_id']}, " +
                          f"Confidence: {concept['confidence']:.2%})")
        
        # Demonstrate ontology-enhanced search
        print("\nDemonstrating ontology-enhanced search:")
        for query in queries:
            print(f"\nQuery: {query}")
            
            # Search with standard approach
            print("  Standard search:")
            standard_results = await graph_rag.search_articles(
                query, 
                max_results=5,
                use_ontology_enhancement=False
            )
            print(f"    Found {len(standard_results)} articles")
            
            # Search with ontology enhancement
            print("  Ontology-enhanced search:")
            enhanced_results = await graph_rag.search_articles(
                query, 
                max_results=5,
                use_ontology_enhancement=True
            )
            print(f"    Found {len(enhanced_results)} articles")
            
            # Compare results
            standard_ids = set(r.get('id') for r in standard_results)
            enhanced_ids = set(r.get('id') for r in enhanced_results)
            new_in_enhanced = [r for r in enhanced_results if r.get('id') not in standard_ids]
            
            if new_in_enhanced:
                print("  New articles found through ontology enhancement:")
                for article in new_in_enhanced[:3]:  # Show top 3
                    print(f"    - {article.get('title', 'Untitled')} (ID: {article.get('id')})")
                    if article.get('ontology_enhanced'):
                        print(f"      (Found via ontology enhancement)")
        
        # Generate summary with ontology enhancement
        print("\nGenerating summary with ontology enhancement:")
        sample_query = "efficacy of antivirals in treating COVID-19"
        print(f"\nQuery: {sample_query}")
        
        summary = await graph_rag.generate_summary(
            sample_query,
            max_articles=5,
            use_ontology_enhancement=True
        )
        
        print("Summary:")
        print(f"  {summary['summary']}")
        print(f"\nBased on {len(summary['articles'])} articles")
        if summary.get('ontology_enhanced'):
            print("  (Enhanced with ontology integration)")

        print("\nKnowledge graph ontology integration demonstration complete!")

    except Exception as e:
        print(f"\nError in knowledge graph ontology demo: {str(e)}")
        import traceback
        traceback.print_exc()


async def run_demo(feature: str):
    """Run the specified demo feature."""
    if feature == "claim" or feature == "all":
        await demo_claim_extraction()
    
    if feature == "xai" or feature == "all":
        await demo_explainable_ai()
    
    if feature == "kg" or feature == "all":
        await demo_knowledge_graph_ontology()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate enhanced medical research capabilities")
    parser.add_argument(
        "--feature", 
        type=str, 
        choices=["claim", "xai", "kg", "all"], 
        default="all",
        help="Feature to demonstrate: claim extraction, explainable AI, knowledge graph ontology, or all"
    )
    
    args = parser.parse_args()
    asyncio.run(run_demo(args.feature))
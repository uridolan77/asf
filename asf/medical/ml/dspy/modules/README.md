# DSPy Modules

This package provides domain-agnostic DSPy modules for various NLP tasks.

## Available Modules

### RAG Modules

- **AdvancedRAGModule**: Standard RAG pipeline with specialized handling for context and citations.
- **MultiStageRAGModule**: Advanced RAG pipeline with multi-stage processing including initial retrieval, query expansion, secondary retrieval, answer generation with citations, and fact checking.

### Contradiction Detection Modules

- **ContradictionDetectionModule**: Detects contradictions between statements and provides explanations.
- **TemporalContradictionModule**: Detects contradictions between statements with temporal context, considering the timestamps of statements.

### Evidence Extraction Modules

- **EvidenceExtractionModule**: Extracts evidence from text to support or refute claims.

### Content Summarization Modules

- **ContentSummarizationModule**: Summarizes texts with key findings and implications.
- **StructuredContentSummarizationModule**: Generates structured summaries with specific sections like background, methods, results, and conclusions.
- **MultiDocumentSummarizationModule**: Summarizes multiple documents, identifying common themes, agreements, and disagreements.

### Advanced QA Modules

- **AdvancedQAModule**: Answers complex questions with evidence assessment and confidence scoring.
- **GuidelineQAModule**: Answers questions based on guidelines with recommendation strength and evidence levels.

### Reasoning Modules

- **ReasoningModule**: Performs structured reasoning on complex cases, generating analyses, recommendations, and explanations.
- **ExpertConsultModule**: Extends reasoning with domain-specific knowledge for complex or specialized cases.

## Usage Examples

### Structured Reasoning Example

```python
import asyncio
from asf.medical.ml.dspy.client import get_enhanced_client
from asf.medical.ml.dspy.modules.reasoning import ReasoningModule

async def main():
    # Initialize DSPy client
    client = await get_enhanced_client()

    # Create a reasoning module
    reasoning_module = ReasoningModule(
        max_analyses=5,
        include_edge_cases=True
    )

    # Register the module
    await client.register_module(
        name="structured_reasoning",
        module=reasoning_module,
        description="Structured reasoning module"
    )

    # Example case description
    case_description = """
    A company is experiencing a significant drop in website conversion rates over the past month.
    The website traffic has remained stable, but the percentage of visitors completing purchases
    has decreased by 25%. The company recently updated their checkout process and implemented
    a new design for product pages. Analytics show that more users are abandoning their carts
    at the payment information step than before. Customer support has received several complaints
    about confusion with the new layout. Mobile users seem to be affected more than desktop users.
    """

    # Call the module
    result = await client.call_module(
        module_name="structured_reasoning",
        case_description=case_description
    )

    # Print the results
    print(f"Analysis: {result['analysis']}")
    print(f"Recommendations: {result['recommendations']}")
    print(f"Reasoning: {result['reasoning'][:200]}...")
    print(f"Confidence: {result['confidence']}")

    # Shut down the client
    await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Expert Consultation Example

```python
import asyncio
from asf.medical.ml.dspy.client import get_enhanced_client
from asf.medical.ml.dspy.modules.reasoning import ExpertConsultModule

async def main():
    # Initialize DSPy client
    client = await get_enhanced_client()

    # Create an expert consultation module
    finance_module = ExpertConsultModule(
        domain="finance"
    )

    # Register the module
    await client.register_module(
        name="finance_consult",
        module=finance_module,
        description="Finance expert consultation module"
    )

    # Example case description
    case_description = """
    A mid-sized company with 150 employees is considering changing their retirement benefits plan.
    Currently, they offer a 401(k) with a 3% match. They're debating between increasing the match
    to 5% or implementing a profit-sharing component that would vary year to year based on company
    performance. The company has been growing steadily at 12% annually for the past 3 years.
    Employee retention has been a challenge in their industry, with competitors offering various
    incentive packages. The CFO is concerned about long-term financial commitments, while the HR
    director emphasizes the need for competitive benefits to attract talent.
    """

    # Call the module
    result = await client.call_module(
        module_name="finance_consult",
        case_description=case_description
    )

    # Print the results
    print(f"Expert Analysis: {result['expert_analysis']}")
    print(f"Expert Recommendations: {result['expert_recommendations']}")
    print(f"Expert Assessment: {result['expert_assessment'][:200]}...")
    print(f"Confidence: {result['confidence']}")

    # Shut down the client
    await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

# DSPy Modules for Medical Research

This package contains specialized DSPy modules for medical research tasks.

## Available Modules

### Medical RAG Modules

- **MedicalRAGModule**: Standard RAG pipeline for medical questions with specialized handling for medical context and citations.
- **EnhancedMedicalRAGModule**: Advanced RAG pipeline with multi-stage processing including initial retrieval, query expansion, secondary retrieval, answer generation with citations, and fact checking.

### Contradiction Detection Modules

- **ContradictionDetectionModule**: Detects contradictions between medical statements and provides explanations.
- **TemporalContradictionModule**: Detects contradictions between medical statements with temporal context, considering the timestamps of statements.

### Evidence Extraction Modules

- **EvidenceExtractionModule**: Extracts evidence from medical text to support or refute claims.

### Medical Summarization Modules

- **MedicalSummarizationModule**: Summarizes medical texts with key findings and implications.

### Clinical QA Modules

- **ClinicalQAModule**: Answers clinical questions with evidence grading and clinical implications.

### Diagnostic Reasoning Modules

- **DiagnosticReasoningModule**: Performs structured diagnostic reasoning on medical cases, generating differential diagnoses, recommended tests, and explanations.
- **SpecialistConsultModule**: Extends diagnostic reasoning with specialist-specific knowledge for complex or specialized medical cases.

## Usage Examples

### Diagnostic Reasoning Example

```python
import asyncio
from asf.medical.ml.dspy.client import get_enhanced_client
from asf.medical.ml.dspy.modules.diagnostic_reasoning import DiagnosticReasoningModule

async def main():
    # Initialize DSPy client
    client = await get_enhanced_client()
    
    # Create a diagnostic reasoning module
    diagnostic_module = DiagnosticReasoningModule(
        max_diagnoses=5,
        include_rare_conditions=True
    )
    
    # Register the module
    await client.register_module(
        name="diagnostic_reasoning",
        module=diagnostic_module,
        description="Medical diagnostic reasoning module"
    )
    
    # Example case description
    case_description = """
    A 45-year-old male presents with sudden onset of severe chest pain radiating to the left arm and jaw. 
    The pain started about 2 hours ago while he was resting. He describes it as a heavy pressure sensation. 
    He has a history of hypertension and hyperlipidemia, and his father had a myocardial infarction at age 50. 
    He is a current smoker with a 20 pack-year history. On examination, he appears diaphoretic and anxious. 
    Vital signs show BP 160/95, HR 110, RR 22, and oxygen saturation 96% on room air. 
    His ECG shows ST-segment elevation in leads II, III, and aVF.
    """
    
    # Call the module
    result = await client.call_module(
        module_name="diagnostic_reasoning",
        case_description=case_description
    )
    
    # Print the results
    print(f"Differential Diagnosis: {result['differential_diagnosis']}")
    print(f"Recommended Tests: {result['recommended_tests']}")
    print(f"Reasoning: {result['reasoning'][:200]}...")
    print(f"Confidence: {result['confidence']}")
    
    # Shut down the client
    await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Specialist Consultation Example

```python
import asyncio
from asf.medical.ml.dspy.client import get_enhanced_client
from asf.medical.ml.dspy.modules.diagnostic_reasoning import SpecialistConsultModule

async def main():
    # Initialize DSPy client
    client = await get_enhanced_client()
    
    # Create a specialist consultation module
    cardiology_module = SpecialistConsultModule(
        specialty="cardiology"
    )
    
    # Register the module
    await client.register_module(
        name="cardiology_consult",
        module=cardiology_module,
        description="Cardiology specialist consultation module"
    )
    
    # Example case description
    case_description = """
    A 45-year-old male presents with sudden onset of severe chest pain radiating to the left arm and jaw. 
    The pain started about 2 hours ago while he was resting. He describes it as a heavy pressure sensation. 
    He has a history of hypertension and hyperlipidemia, and his father had a myocardial infarction at age 50. 
    He is a current smoker with a 20 pack-year history. On examination, he appears diaphoretic and anxious. 
    Vital signs show BP 160/95, HR 110, RR 22, and oxygen saturation 96% on room air. 
    His ECG shows ST-segment elevation in leads II, III, and aVF.
    """
    
    # Call the module
    result = await client.call_module(
        module_name="cardiology_consult",
        case_description=case_description
    )
    
    # Print the results
    print(f"Specialist Diagnosis: {result['specialist_diagnosis']}")
    print(f"Specialist Recommendations: {result['specialist_recommendations']}")
    print(f"Specialist Assessment: {result['specialist_assessment'][:200]}...")
    print(f"Confidence: {result['confidence']}")
    
    # Shut down the client
    await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

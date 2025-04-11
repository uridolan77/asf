"""
DSPy Signatures for Medical Research

This module provides predefined DSPy signatures for common medical research tasks.
These signatures define the input and output fields for DSPy modules.
"""

import dspy


class MedicalQA(dspy.Signature):
    """Answer a medical question based on the provided context."""
    context = dspy.InputField(desc="Medical literature, research papers, or clinical notes")
    question = dspy.InputField(desc="A specific medical question")
    answer = dspy.OutputField(desc="A comprehensive, accurate answer to the medical question")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")


class MedicalRAG(dspy.Signature):
    """Retrieve relevant medical information and answer a question."""
    question = dspy.InputField(desc="A specific medical question")
    passages = dspy.OutputField(desc="Relevant passages from medical literature")
    answer = dspy.OutputField(desc="A comprehensive, accurate answer based on the retrieved passages")
    citations = dspy.OutputField(desc="Citations for the information in the answer")


class EvidenceExtraction(dspy.Signature):
    """Extract evidence from medical text to support a claim."""
    text = dspy.InputField(desc="Medical text to analyze")
    claim = dspy.InputField(desc="The claim to find evidence for")
    evidence = dspy.OutputField(desc="Evidence from the text that supports or refutes the claim")
    relation = dspy.OutputField(desc="Whether the evidence supports, refutes, or is neutral to the claim")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")


class MedicalSummarization(dspy.Signature):
    """Summarize medical content with key findings and implications."""
    text = dspy.InputField(desc="Medical text to summarize")
    audience = dspy.InputField(desc="Target audience (e.g., 'clinician', 'researcher', 'patient')")
    summary = dspy.OutputField(desc="Concise summary of the medical content")
    key_findings = dspy.OutputField(desc="List of key findings")
    implications = dspy.OutputField(desc="Clinical or research implications")


class ContradictionDetection(dspy.Signature):
    """Detect contradictions between medical statements."""
    statement1 = dspy.InputField(desc="First medical statement")
    statement2 = dspy.InputField(desc="Second medical statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict each other (true/false)")
    explanation = dspy.OutputField(desc="Explanation of the contradiction or lack thereof")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")


class TemporalContradictionDetection(dspy.Signature):
    """Detect contradictions between medical statements with temporal context."""
    statement1 = dspy.InputField(desc="First medical statement")
    timestamp1 = dspy.InputField(desc="Timestamp of the first statement")
    statement2 = dspy.InputField(desc="Second medical statement")
    timestamp2 = dspy.InputField(desc="Timestamp of the second statement")
    contradiction = dspy.OutputField(desc="Whether the statements contradict each other (true/false)")
    temporal_relation = dspy.OutputField(desc="How time affects the contradiction (e.g., 'evolution', 'update', 'conflict')")
    explanation = dspy.OutputField(desc="Explanation of the contradiction or lack thereof")


class ClinicalTrialAnalysis(dspy.Signature):
    """Analyze a clinical trial description."""
    trial_description = dspy.InputField(desc="Description of the clinical trial")
    analysis = dspy.OutputField(desc="Analysis of the trial design, methods, and potential biases")
    strengths = dspy.OutputField(desc="Strengths of the trial design")
    limitations = dspy.OutputField(desc="Limitations or potential biases in the trial")
    implications = dspy.OutputField(desc="Implications for clinical practice or future research")


class StudyDesignClassification(dspy.Signature):
    """Classify the design of a medical study."""
    study_description = dspy.InputField(desc="Description of the medical study")
    design_type = dspy.OutputField(desc="Type of study design (e.g., RCT, cohort, case-control)")
    evidence_level = dspy.OutputField(desc="Level of evidence (1-5, with 1 being highest)")
    justification = dspy.OutputField(desc="Justification for the classification")


class BiasAssessment(dspy.Signature):
    """Assess potential biases in a medical study."""
    study_description = dspy.InputField(desc="Description of the medical study")
    biases = dspy.OutputField(desc="List of potential biases identified")
    impact = dspy.OutputField(desc="Assessment of how biases might impact the results")
    mitigation = dspy.OutputField(desc="Suggestions for how the biases could be mitigated")


class PRISMAScreening(dspy.Signature):
    """Screen a study for inclusion in a systematic review using PRISMA criteria."""
    title = dspy.InputField(desc="Title of the study")
    abstract = dspy.InputField(desc="Abstract of the study")
    inclusion_criteria = dspy.InputField(desc="PRISMA inclusion criteria")
    eligible = dspy.OutputField(desc="Whether the study meets inclusion criteria (true/false)")
    justification = dspy.OutputField(desc="Justification for the eligibility decision")
    relevant_sections = dspy.OutputField(desc="Sections of the abstract relevant to the decision")


class MedicalEntityExtraction(dspy.Signature):
    """Extract medical entities from text."""
    text = dspy.InputField(desc="Medical text to analyze")
    entity_types = dspy.InputField(desc="Types of entities to extract (e.g., 'condition', 'medication', 'procedure')")
    entities = dspy.OutputField(desc="List of extracted entities with their types and positions")
    relations = dspy.OutputField(desc="Relations between extracted entities")


class MedicalFactChecking(dspy.Signature):
    """Verify medical facts against reliable sources."""
    claim = dspy.InputField(desc="Medical claim to verify")
    sources = dspy.InputField(desc="Reliable sources to check against")
    verdict = dspy.OutputField(desc="Whether the claim is supported by the sources (true/false/partially)")
    evidence = dspy.OutputField(desc="Evidence from the sources supporting the verdict")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")


class TreatmentComparisonAnalysis(dspy.Signature):
    """Compare different treatments for a medical condition."""
    condition = dspy.InputField(desc="Medical condition being treated")
    treatment1 = dspy.InputField(desc="First treatment option")
    treatment2 = dspy.InputField(desc="Second treatment option")
    comparison = dspy.OutputField(desc="Comparison of efficacy, risks, and benefits")
    recommendation = dspy.OutputField(desc="Recommendation based on the comparison")
    evidence_quality = dspy.OutputField(desc="Assessment of the quality of evidence for the comparison")


class MedicalLiteratureReview(dspy.Signature):
    """Review medical literature on a specific topic."""
    topic = dspy.InputField(desc="Medical topic to review")
    papers = dspy.InputField(desc="List of relevant papers (titles and abstracts)")
    summary = dspy.OutputField(desc="Summary of the current state of knowledge")
    consensus = dspy.OutputField(desc="Areas of consensus in the literature")
    controversies = dspy.OutputField(desc="Areas of controversy or uncertainty")
    future_directions = dspy.OutputField(desc="Suggested directions for future research")


class DiagnosticReasoning(dspy.Signature):
    """Reason through a diagnostic process for a clinical case."""
    case_description = dspy.InputField(desc="Description of the clinical case")
    differential_diagnosis = dspy.OutputField(desc="List of potential diagnoses with reasoning")
    recommended_tests = dspy.OutputField(desc="Tests recommended to narrow the diagnosis")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process")


class MedicalImageDescription(dspy.Signature):
    """Describe and analyze a medical image."""
    image_description = dspy.InputField(desc="Description of the medical image")
    modality = dspy.InputField(desc="Imaging modality (e.g., X-ray, MRI, CT)")
    findings = dspy.OutputField(desc="Description of visible findings")
    interpretation = dspy.OutputField(desc="Clinical interpretation of the findings")
    recommendations = dspy.OutputField(desc="Recommendations for further imaging or clinical correlation")


class PatientEducationMaterial(dspy.Signature):
    """Generate patient education material for a medical condition."""
    condition = dspy.InputField(desc="Medical condition to explain")
    education_level = dspy.InputField(desc="Target education level (e.g., 'elementary', 'high school', 'college')")
    content = dspy.OutputField(desc="Educational content about the condition")
    self_care = dspy.OutputField(desc="Self-care recommendations")
    when_to_seek_help = dspy.OutputField(desc="Signs or symptoms that should prompt seeking medical attention")


# Export all signatures
__all__ = [
    'MedicalQA',
    'MedicalRAG',
    'EvidenceExtraction',
    'MedicalSummarization',
    'ContradictionDetection',
    'TemporalContradictionDetection',
    'ClinicalTrialAnalysis',
    'StudyDesignClassification',
    'BiasAssessment',
    'PRISMAScreening',
    'MedicalEntityExtraction',
    'MedicalFactChecking',
    'TreatmentComparisonAnalysis',
    'MedicalLiteratureReview',
    'DiagnosticReasoning',
    'MedicalImageDescription',
    'PatientEducationMaterial'
]

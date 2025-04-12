DSPy Metrics for Medical Research

This module provides evaluation metrics for DSPy modules in medical research.
These metrics are used to evaluate the quality of module outputs and optimize modules.

import logging
from typing import Dict, Any, List, Optional, Union, Callable

import dspy

# Set up logging
logger = logging.getLogger(__name__)


def medical_qa_accuracy(prediction: Any, example: Dict[str, Any]) -> float:
    """
    Evaluate the accuracy of a medical QA prediction.
    
    Args:
        prediction: The model's prediction
        example: The example with ground truth
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Extract prediction
    if hasattr(prediction, 'answer'):
        pred_answer = prediction.answer
    elif isinstance(prediction, dict) and 'answer' in prediction:
        pred_answer = prediction['answer']
    else:
        pred_answer = str(prediction)
    
    # Extract ground truth
    if '_output' in example and hasattr(example['_output'], 'answer'):
        true_answer = example['_output'].answer
    elif '_output' in example and isinstance(example['_output'], dict) and 'answer' in example['_output']:
        true_answer = example['_output']['answer']
    elif 'answer' in example:
        true_answer = example['answer']
    else:
        logger.warning("No ground truth answer found in example")
        return 0.0
    
    # Use LLM-as-judge to evaluate accuracy
    judge_signature = dspy.Signature(
        prediction=dspy.InputField(desc="The model's predicted answer"),
        reference=dspy.InputField(desc="The reference answer"),
        score=dspy.OutputField(desc="Accuracy score between 0 and 1"),
        explanation=dspy.OutputField(desc="Explanation of the score")
    )
    
    judge = dspy.Predict(judge_signature)
    
    try:
        result = judge(prediction=pred_answer, reference=true_answer)
        
        if hasattr(result, 'score'):
            score = result.score
        elif isinstance(result, dict) and 'score' in result:
            score = result['score']
        else:
            score = 0.0
            
        # Convert string score to float if needed
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.0
                
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score
    except Exception as e:
        logger.error(f"Error in medical_qa_accuracy: {str(e)}")
        return 0.0


def medical_rag_relevance(prediction: Any, example: Dict[str, Any]) -> float:
    """
    Evaluate the relevance of retrieved passages in a medical RAG prediction.
    
    Args:
        prediction: The model's prediction
        example: The example with ground truth
        
    Returns:
        float: Relevance score between 0 and 1
    """
    # Extract prediction
    if hasattr(prediction, 'passages'):
        pred_passages = prediction.passages
    elif isinstance(prediction, dict) and 'passages' in prediction:
        pred_passages = prediction['passages']
    else:
        logger.warning("No passages found in prediction")
        return 0.0
    
    # Extract question
    if hasattr(prediction, 'question'):
        question = prediction.question
    elif isinstance(prediction, dict) and 'question' in prediction:
        question = prediction['question']
    elif 'question' in example:
        question = example['question']
    else:
        logger.warning("No question found in prediction or example")
        return 0.0
    
    # Format passages for evaluation
    if isinstance(pred_passages, list):
        formatted_passages = "\n\n".join([
            f"Passage {i+1}: {p}" if isinstance(p, str) else 
            f"Passage {i+1}: {p.get('content', str(p)) if isinstance(p, dict) else str(p)}"
            for i, p in enumerate(pred_passages[:5])  # Limit to first 5 passages
        ])
    else:
        formatted_passages = str(pred_passages)
    
    # Use LLM to evaluate relevance
    judge_signature = dspy.Signature(
        question=dspy.InputField(desc="The medical question"),
        passages=dspy.InputField(desc="The retrieved passages"),
        relevance_score=dspy.OutputField(desc="Relevance score between 0 and 1"),
        explanation=dspy.OutputField(desc="Explanation of the relevance score")
    )
    
    judge = dspy.Predict(judge_signature)
    
    try:
        result = judge(question=question, passages=formatted_passages)
        
        if hasattr(result, 'relevance_score'):
            score = result.relevance_score
        elif isinstance(result, dict) and 'relevance_score' in result:
            score = result['relevance_score']
        else:
            score = 0.0
            
        # Convert string score to float if needed
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.0
                
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score
    except Exception as e:
        logger.error(f"Error in medical_rag_relevance: {str(e)}")
        return 0.0


def contradiction_detection_accuracy(prediction: Any, example: Dict[str, Any]) -> float:
    """
    Evaluate the accuracy of a contradiction detection prediction.
    
    Args:
        prediction: The model's prediction
        example: The example with ground truth
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Extract prediction
    if hasattr(prediction, 'contradiction'):
        pred_contradiction = prediction.contradiction
    elif isinstance(prediction, dict) and 'contradiction' in prediction:
        pred_contradiction = prediction['contradiction']
    else:
        logger.warning("No contradiction field found in prediction")
        return 0.0
    
    # Convert to boolean if needed
    if isinstance(pred_contradiction, str):
        pred_contradiction = pred_contradiction.lower() in ['true', 'yes', '1']
    
    # Extract ground truth
    if '_output' in example and hasattr(example['_output'], 'contradiction'):
        true_contradiction = example['_output'].contradiction
    elif '_output' in example and isinstance(example['_output'], dict) and 'contradiction' in example['_output']:
        true_contradiction = example['_output']['contradiction']
    elif 'contradiction' in example:
        true_contradiction = example['contradiction']
    else:
        logger.warning("No ground truth contradiction found in example")
        return 0.0
    
    # Convert to boolean if needed
    if isinstance(true_contradiction, str):
        true_contradiction = true_contradiction.lower() in ['true', 'yes', '1']
    
    # Calculate accuracy (1.0 if match, 0.0 if not)
    return 1.0 if pred_contradiction == true_contradiction else 0.0


def medical_summarization_quality(prediction: Any, example: Dict[str, Any]) -> float:
    """
    Evaluate the quality of a medical summarization prediction.
    
    Args:
        prediction: The model's prediction
        example: The example with ground truth
        
    Returns:
        float: Quality score between 0 and 1
    """
    # Extract prediction
    if hasattr(prediction, 'summary'):
        pred_summary = prediction.summary
    elif isinstance(prediction, dict) and 'summary' in prediction:
        pred_summary = prediction['summary']
    else:
        pred_summary = str(prediction)
    
    # Extract text
    if hasattr(prediction, 'text'):
        text = prediction.text
    elif isinstance(prediction, dict) and 'text' in prediction:
        text = prediction['text']
    elif 'text' in example:
        text = example['text']
    else:
        logger.warning("No text found in prediction or example")
        return 0.0
    
    # Use LLM to evaluate summary quality
    judge_signature = dspy.Signature(
        text=dspy.InputField(desc="The original medical text"),
        summary=dspy.InputField(desc="The generated summary"),
        completeness_score=dspy.OutputField(desc="Completeness score between 0 and 1"),
        accuracy_score=dspy.OutputField(desc="Accuracy score between 0 and 1"),
        conciseness_score=dspy.OutputField(desc="Conciseness score between 0 and 1"),
        overall_score=dspy.OutputField(desc="Overall quality score between 0 and 1"),
        explanation=dspy.OutputField(desc="Explanation of the scores")
    )
    
    judge = dspy.Predict(judge_signature)
    
    try:
        result = judge(text=text[:5000], summary=pred_summary)  # Limit text length
        
        if hasattr(result, 'overall_score'):
            score = result.overall_score
        elif isinstance(result, dict) and 'overall_score' in result:
            score = result['overall_score']
        else:
            # Calculate average of individual scores if available
            scores = []
            
            if hasattr(result, 'completeness_score'):
                scores.append(float(result.completeness_score) if isinstance(result.completeness_score, str) else result.completeness_score)
            elif isinstance(result, dict) and 'completeness_score' in result:
                scores.append(float(result['completeness_score']) if isinstance(result['completeness_score'], str) else result['completeness_score'])
                
            if hasattr(result, 'accuracy_score'):
                scores.append(float(result.accuracy_score) if isinstance(result.accuracy_score, str) else result.accuracy_score)
            elif isinstance(result, dict) and 'accuracy_score' in result:
                scores.append(float(result['accuracy_score']) if isinstance(result['accuracy_score'], str) else result['accuracy_score'])
                
            if hasattr(result, 'conciseness_score'):
                scores.append(float(result.conciseness_score) if isinstance(result.conciseness_score, str) else result.conciseness_score)
            elif isinstance(result, dict) and 'conciseness_score' in result:
                scores.append(float(result['conciseness_score']) if isinstance(result['conciseness_score'], str) else result['conciseness_score'])
            
            score = sum(scores) / len(scores) if scores else 0.0
            
        # Convert string score to float if needed
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.0
                
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score
    except Exception as e:
        logger.error(f"Error in medical_summarization_quality: {str(e)}")
        return 0.0


def evidence_extraction_accuracy(prediction: Any, example: Dict[str, Any]) -> float:
    """
    Evaluate the accuracy of an evidence extraction prediction.
    
    Args:
        prediction: The model's prediction
        example: The example with ground truth
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Extract prediction
    if hasattr(prediction, 'relation'):
        pred_relation = prediction.relation
    elif isinstance(prediction, dict) and 'relation' in prediction:
        pred_relation = prediction['relation']
    else:
        logger.warning("No relation field found in prediction")
        return 0.0
    
    # Normalize relation
    pred_relation = pred_relation.lower()
    if pred_relation in ['supports', 'support', 'supporting']:
        pred_relation = 'supports'
    elif pred_relation in ['refutes', 'refute', 'refuting', 'contradicts', 'contradict']:
        pred_relation = 'refutes'
    else:
        pred_relation = 'neutral'
    
    # Extract ground truth
    if '_output' in example and hasattr(example['_output'], 'relation'):
        true_relation = example['_output'].relation
    elif '_output' in example and isinstance(example['_output'], dict) and 'relation' in example['_output']:
        true_relation = example['_output']['relation']
    elif 'relation' in example:
        true_relation = example['relation']
    else:
        logger.warning("No ground truth relation found in example")
        return 0.0
    
    # Normalize ground truth
    true_relation = true_relation.lower()
    if true_relation in ['supports', 'support', 'supporting']:
        true_relation = 'supports'
    elif true_relation in ['refutes', 'refute', 'refuting', 'contradicts', 'contradict']:
        true_relation = 'refutes'
    else:
        true_relation = 'neutral'
    
    # Calculate accuracy (1.0 if match, 0.0 if not)
    return 1.0 if pred_relation == true_relation else 0.0


def clinical_qa_correctness(prediction: Any, example: Dict[str, Any]) -> float:
    """
    Evaluate the correctness of a clinical QA prediction.
    
    Args:
        prediction: The model's prediction
        example: The example with ground truth
        
    Returns:
        float: Correctness score between 0 and 1
    """
    # Extract prediction
    if hasattr(prediction, 'answer'):
        pred_answer = prediction.answer
    elif isinstance(prediction, dict) and 'answer' in prediction:
        pred_answer = prediction['answer']
    else:
        pred_answer = str(prediction)
    
    # Extract ground truth
    if '_output' in example and hasattr(example['_output'], 'answer'):
        true_answer = example['_output'].answer
    elif '_output' in example and isinstance(example['_output'], dict) and 'answer' in example['_output']:
        true_answer = example['_output']['answer']
    elif 'answer' in example:
        true_answer = example['answer']
    else:
        logger.warning("No ground truth answer found in example")
        return 0.0
    
    # Extract question
    if 'question' in example:
        question = example['question']
    else:
        question = "Unknown question"
    
    # Use LLM to evaluate clinical correctness
    judge_signature = dspy.Signature(
        question=dspy.InputField(desc="The clinical question"),
        prediction=dspy.InputField(desc="The model's predicted answer"),
        reference=dspy.InputField(desc="The reference answer"),
        correctness_score=dspy.OutputField(desc="Clinical correctness score between 0 and 1"),
        explanation=dspy.OutputField(desc="Explanation of the clinical correctness score")
    )
    
    judge = dspy.Predict(judge_signature)
    
    try:
        result = judge(question=question, prediction=pred_answer, reference=true_answer)
        
        if hasattr(result, 'correctness_score'):
            score = result.correctness_score
        elif isinstance(result, dict) and 'correctness_score' in result:
            score = result['correctness_score']
        else:
            score = 0.0
            
        # Convert string score to float if needed
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.0
                
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score
    except Exception as e:
        logger.error(f"Error in clinical_qa_correctness: {str(e)}")
        return 0.0


def combined_medical_metric(prediction: Any, example: Dict[str, Any]) -> float:
    """
    Combined metric for medical tasks that selects the appropriate metric based on the task.
    
    Args:
        prediction: The model's prediction
        example: The example with ground truth
        
    Returns:
        float: Score between 0 and 1
    """
    # Determine the task type based on prediction and example fields
    if (hasattr(prediction, 'passages') or 
        (isinstance(prediction, dict) and 'passages' in prediction)):
        # RAG task
        return medical_rag_relevance(prediction, example)
    
    elif (hasattr(prediction, 'contradiction') or 
          (isinstance(prediction, dict) and 'contradiction' in prediction)):
        # Contradiction detection task
        return contradiction_detection_accuracy(prediction, example)
    
    elif (hasattr(prediction, 'summary') or 
          (isinstance(prediction, dict) and 'summary' in prediction)):
        # Summarization task
        return medical_summarization_quality(prediction, example)
    
    elif (hasattr(prediction, 'relation') or 
          (isinstance(prediction, dict) and 'relation' in prediction)):
        # Evidence extraction task
        return evidence_extraction_accuracy(prediction, example)
    
    elif (hasattr(prediction, 'differential_diagnosis') or 
          (isinstance(prediction, dict) and 'differential_diagnosis' in prediction)):
        # Diagnostic reasoning task
        return clinical_qa_correctness(prediction, example)
    
    else:
        # Default to medical QA accuracy
        return medical_qa_accuracy(prediction, example)


def create_llm_judge_metric(
    task_description: str,
    score_criteria: List[str],
    max_score: int = 10
) -> Callable[[Any, Dict[str, Any]], float]:
    """
    Create a custom LLM-as-judge metric for medical tasks.
    
    Args:
        task_description: Description of the task being evaluated
        score_criteria: List of criteria for scoring
        max_score: Maximum score value
        
    Returns:
        Callable: A metric function that returns a score between 0 and 1
    """
    # Create the judge signature
    judge_signature = dspy.Signature(
        task=dspy.InputField(desc="Description of the task"),
        criteria=dspy.InputField(desc="Criteria for evaluation"),
        prediction=dspy.InputField(desc="The model's prediction"),
        reference=dspy.InputField(desc="The reference or context"),
        score=dspy.OutputField(desc=f"Score between 0 and {max_score}"),
        explanation=dspy.OutputField(desc="Detailed explanation of the score")
    )
    
    # Create the judge
    judge = dspy.ChainOfThought(judge_signature)
    
    # Format criteria
    formatted_criteria = "\n".join([f"{i+1}. {criterion}" for i, criterion in enumerate(score_criteria)])
    
    def metric_fn(prediction: Any, example: Dict[str, Any]) -> float:
        """
        Custom LLM-as-judge metric function.
        
        Args:
            prediction: The model's prediction
            example: The example with ground truth or context
            
        Returns:
            float: Score between 0 and 1
        """
        # Extract prediction
        if hasattr(prediction, '__dict__'):
            pred_text = str(prediction.__dict__)
        elif isinstance(prediction, dict):
            pred_text = str(prediction)
        else:
            pred_text = str(prediction)
        
        # Extract reference or context
        if '_output' in example:
            if hasattr(example['_output'], '__dict__'):
                ref_text = str(example['_output'].__dict__)
            elif isinstance(example['_output'], dict):
                ref_text = str(example['_output'])
            else:
                ref_text = str(example['_output'])
        elif 'text' in example:
            ref_text = example['text']
        else:
            ref_text = str(example)
        
        try:
            # Call the judge
            result = judge(
                task=task_description,
                criteria=formatted_criteria,
                prediction=pred_text,
                reference=ref_text
            )
            
            # Extract score
            if hasattr(result, 'score'):
                score = result.score
            elif isinstance(result, dict) and 'score' in result:
                score = result['score']
            else:
                score = 0
                
            # Convert string score to float if needed
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = 0
                    
            # Normalize to 0-1 range
            normalized_score = score / max_score
            
            # Ensure score is between 0 and 1
            normalized_score = max(0.0, min(1.0, normalized_score))
            
            return normalized_score
        except Exception as e:
            logger.error(f"Error in custom LLM judge metric: {str(e)}")
            return 0.0
    
    return metric_fn


# Export all metrics
__all__ = [
    'medical_qa_accuracy',
    'medical_rag_relevance',
    'contradiction_detection_accuracy',
    'medical_summarization_quality',
    'evidence_extraction_accuracy',
    'clinical_qa_correctness',
    'combined_medical_metric',
    'create_llm_judge_metric'
]

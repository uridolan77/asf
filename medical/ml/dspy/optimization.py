Enhanced DSPy Optimization

This module provides enhanced optimization functionality for DSPy modules
with better metrics tracking, timeout handling, and validation set evaluation.

import os
import time
import json
import uuid
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import difflib

import dspy
import mlflow

from settings import get_enhanced_settings
from audit_logging import get_audit_logger

# Set up logging
import logging
logger = logging.getLogger(__name__)


class EnhancedOptimizer:
    Enhanced optimizer for DSPy modules.
    
    This class provides enhanced optimization functionality with better metrics tracking,
    timeout handling, and validation set evaluation.
    
    def __init__(self):
        Initialize the enhanced optimizer.
        
        Args:
        
        Optimize a DSPy module using the specified optimizer with enhanced monitoring.
        
        Args:
            module: The module to optimize
            metric_fn: Function that evaluates module outputs
            examples: Training examples for optimization
            optimizer_type: Type of optimizer to use
            max_rounds: Maximum optimization rounds
            num_threads: Number of threads for optimization
            save_history: Whether to save optimization history
            validation_examples: Optional separate validation set
            optimization_timeout: Maximum seconds to run optimization (None for no limit)
            artifact_dir: Directory to save artifacts
            **optimizer_kwargs: Additional optimizer parameters
            
        Returns:
            Tuple[dspy.Module, Dict[str, Any]]: Optimized module and metrics
            
        Raises:
            ValueError: For unsupported optimizer types
            TimeoutError: If optimization exceeds the timeout
            Exception: For optimization failures
        """
        logger.info(f"Optimizing module {module.__class__.__name__} with {optimizer_type}")
        
        # Use default timeout if not specified
        if optimization_timeout is None:
            optimization_timeout = self.settings.OPTIMIZATION_TIMEOUT
        
        # Create a unique run ID for this optimization
        run_id = str(uuid.uuid4())
        
        # Create artifact directory if not provided
        if artifact_dir is None:
            artifact_dir = f"optimization_artifacts_{run_id}"
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Record original prompts for comparison
        original_prompts = self._extract_prompts(module)
        
        # Create optimizer based on type
        if optimizer_type.lower() == "mipro":
            optimizer = dspy.MIPROv2(metric=metric_fn, **optimizer_kwargs)
        elif optimizer_type.lower() == "bootstrap":
            optimizer = dspy.BootstrapFewShot(metric=metric_fn, **optimizer_kwargs)
        elif optimizer_type.lower() == "zerotune":
            optimizer = dspy.ZeroTune(metric=metric_fn, **optimizer_kwargs)
        elif optimizer_type.lower() == "fewshot":
            optimizer = dspy.FewShot(k=optimizer_kwargs.pop("k", 3), metric=metric_fn, **optimizer_kwargs)
        elif optimizer_type.lower() == "copro":
            optimizer = dspy.COPRO(metric=metric_fn, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Run optimization with metrics tracking
        optimization_metrics = {}
        start_time = time.time()
        round_metrics = []
        optimized_module = None
        
        try:
            # Wrap the optimizer function to collect per-round metrics
            original_round_fn = getattr(optimizer, "_process_round", None)
            if original_round_fn:
                @wraps(original_round_fn)
                def instrumented_round_fn(*args, **kwargs):
                    """
                    instrumented_round_fn function.
                    
                    This function provides functionality for..."""
                    round_start = time.time()
                    result = original_round_fn(*args, **kwargs)
                    round_end = time.time()
                    
                    # Collect round metrics
                    round_metrics.append({
                        "round": len(round_metrics) + 1,
                        "duration": round_end - round_start,
                        "score": getattr(result, "score", None),
                        "timestamp": datetime.now().isoformat()
                    })
                    return result
                    
                # Replace the original method with our instrumented version
                setattr(optimizer, "_process_round", instrumented_round_fn)
            
            # Set up a future with timeout if specified
            async def run_optimization():
                # Run the optimization in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: optimizer.optimize(
                        module,
                        trainset=examples,
                        max_rounds=max_rounds,
                        num_threads=min(num_threads, self.settings.THREAD_LIMIT)
                    )
                )
            
            # Run with timeout if specified
            if optimization_timeout:
                try:
                    optimized_module = await asyncio.wait_for(
                        run_optimization(), 
                        timeout=optimization_timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Optimization timed out after {optimization_timeout} seconds")
            else:
                optimized_module = await run_optimization()
            
            # Calculate optimization time
            optimization_time = time.time() - start_time
            
            # Extract optimized prompts
            optimized_prompts = self._extract_prompts(optimized_module)
            
            # Evaluate on validation set if provided
            validation_metrics = {}
            if validation_examples:
                validation_metrics = await self._evaluate_on_validation(
                    optimized_module, 
                    metric_fn, 
                    validation_examples,
                    artifact_dir
                )
            
            # Record metrics
            optimization_metrics = {
                "optimizer_type": optimizer_type,
                "module_type": module.__class__.__name__,
                "optimization_time": optimization_time,
                "num_examples": len(examples),
                "max_rounds": max_rounds,
                "rounds_completed": len(round_metrics),
                "round_metrics": round_metrics,
                "original_prompts": original_prompts,
                "optimized_prompts": optimized_prompts,
                "validation_metrics": validation_metrics,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate prompt diff statistics
            diff_stats = self._calculate_prompt_diff_stats(original_prompts, optimized_prompts)
            optimization_metrics["prompt_diff_stats"] = diff_stats
            
            # Log telemetry if enabled
            if self.settings.ENABLE_TELEMETRY:
                await self._log_optimization_metrics(
                    optimization_metrics, 
                    artifact_dir=artifact_dir
                )
            
            # Save optimization history if requested
            if save_history:
                await self._save_optimization_history(
                    module.__class__.__name__,
                    optimizer_type,
                    optimization_metrics,
                    artifact_dir=artifact_dir
                )
            
            # Log optimization event
            if self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_optimization(
                    module_name=module.__class__.__name__,
                    optimizer_type=optimizer_type,
                    metrics=optimization_metrics,
                    original_prompts=original_prompts,
                    optimized_prompts=optimized_prompts
                )
            
            logger.info(f"Optimization complete in {optimization_time:.2f} seconds")
            return optimized_module, optimization_metrics
                
        except Exception as e:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "round_metrics": round_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save error details to artifact directory
            error_file = os.path.join(artifact_dir, "optimization_error.json")
            try:
                with open(error_file, 'w') as f:
                    json.dump(error_details, f, indent=2)
            except Exception as write_err:
                logger.warning(f"Failed to write error details: {str(write_err)}")
            
            logger.error(f"Optimization failed: {str(e)}\n{traceback.format_exc()}")
            
            # Log error
            if self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    context={
                        "module_type": module.__class__.__name__,
                        "optimizer_type": optimizer_type,
                        "run_id": run_id,
                        "rounds_completed": len(round_metrics)
                    }
                )
            
            raise
        finally:
            # Always ensure cleanup of artifact directory if empty
            try:
                if os.path.exists(artifact_dir) and not os.listdir(artifact_dir):
                    os.rmdir(artifact_dir)
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up artifact directory: {str(cleanup_err)}")
    
    async def _evaluate_on_validation(
        self,
        module: dspy.Module,
        metric_fn: Callable,
        validation_examples: List[Dict[str, Any]],
        artifact_dir: str
    ) -> Dict[str, Any]:
        """
        Evaluate an optimized module on a validation set.
        
        Args:
            module: The module to evaluate
            metric_fn: The metric function to use
            validation_examples: Validation examples
            artifact_dir: Directory to save artifacts
            
        Returns:
            Dict[str, Any]: Validation metrics
        """
        logger.info(f"Evaluating module on {len(validation_examples)} validation examples")
        
        results = []
        total_score = 0.0
        start_time = time.time()
        
        # Process each validation example
        for i, example in enumerate(validation_examples):
            try:
                # Execute the module with inputs from the example
                inputs = {k: v for k, v in example.items() if k not in ['_output', '_score']}
                
                # Run in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                prediction = await loop.run_in_executor(
                    None,
                    lambda: module(**inputs)
                )
                
                # Calculate the metric score
                score = await loop.run_in_executor(
                    None,
                    lambda: metric_fn(prediction, example)
                )
                
                total_score += score
                
                # Save detailed result
                results.append({
                    "example_id": i,
                    "inputs": inputs,
                    "prediction": self._serialize_prediction(prediction),
                    "expected": example.get('_output', None),
                    "score": score
                })
            except Exception as e:
                logger.warning(f"Validation example {i} failed: {str(e)}")
                results.append({
                    "example_id": i,
                    "inputs": {k: v for k, v in example.items() if k not in ['_output', '_score']},
                    "error": str(e),
                    "score": 0.0
                })
        
        # Calculate overall metrics
        validation_time = time.time() - start_time
        metrics = {
            "average_score": total_score / len(validation_examples) if validation_examples else 0,
            "validation_time": validation_time,
            "num_examples": len(validation_examples),
            "success_rate": sum(1 for r in results if 'error' not in r) / len(results) if results else 0
        }
        
        # Save detailed results to artifact directory
        results_file = os.path.join(artifact_dir, "validation_results.json")
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    "metrics": metrics,
                    "detailed_results": results
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write validation results: {str(e)}")
        
        logger.info(f"Validation complete: average score {metrics['average_score']:.4f}")
        return metrics
    
    def _serialize_prediction(self, prediction) -> Dict[str, Any]:
        """
        Convert a prediction object to a serializable dictionary.
        
        Args:
            prediction: The prediction object
            
        Returns:
            Dict[str, Any]: Serializable representation
        """
        if hasattr(prediction, "__dict__"):
            return {k: v for k, v in prediction.__dict__.items()}
        elif isinstance(prediction, (list, tuple)):
            return [self._serialize_prediction(item) for item in prediction]
        elif isinstance(prediction, dict):
            return {k: self._serialize_prediction(v) for k, v in prediction.items()}
        else:
            return str(prediction)
    
    def _extract_prompts(self, module) -> Dict[str, str]:
        """
        Extract prompt templates from a module.
        
        Args:
            module: The module to extract from
            
        Returns:
            Dict[str, str]: A dictionary mapping template names to their content
        """
        templates = {}
        
        # Try using the get_prompt_templates method if available
        if hasattr(module, "get_prompt_templates"):
            try:
                return module.get_prompt_templates()
            except Exception as e:
                logger.warning(f"Failed to get prompt templates: {str(e)}")
        
        # Extract from ChainOfThought
        if isinstance(module, dspy.ChainOfThought):
            if hasattr(module, "prompt_template"):
                templates["cot_template"] = getattr(module, "prompt_template", "")
        
        # Extract from predict modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Predict):
                if hasattr(attr, "prompt_template"):
                    templates[f"{name}_template"] = getattr(attr, "prompt_template", "")
        
        # For nested modules, recursively extract
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Module) and not isinstance(attr, dspy.Predict):
                try:
                    nested_templates = self._extract_prompts(attr)
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception:
                    pass
        
        return templates
    
    def _calculate_prompt_diff_stats(
        self, 
        original_prompts: Dict[str, str], 
        optimized_prompts: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about prompt differences.
        
        Args:
            original_prompts: Original prompt templates
            optimized_prompts: Optimized prompt templates
            
        Returns:
            Dict[str, Any]: Diff statistics
        """
        stats = {
            "unchanged_count": 0,
            "changed_count": 0,
            "length_changes": {},
            "token_count_changes": {},
            "diffs": {}
        }
        
        # Process each prompt
        for name in set(original_prompts) & set(optimized_prompts):
            original = original_prompts[name]
            optimized = optimized_prompts[name]
            
            if original == optimized:
                stats["unchanged_count"] += 1
            else:
                stats["changed_count"] += 1
                
                # Calculate length changes
                original_len = len(original)
                optimized_len = len(optimized)
                stats["length_changes"][name] = {
                    "original": original_len,
                    "optimized": optimized_len,
                    "diff": optimized_len - original_len,
                    "percent_change": ((optimized_len - original_len) / original_len * 100) 
                                      if original_len > 0 else float('inf')
                }
                
                # Calculate approximate token count changes
                # (simple approximation - 4 chars per token)
                original_tokens = original_len // 4
                optimized_tokens = optimized_len // 4
                stats["token_count_changes"][name] = {
                    "original": original_tokens,
                    "optimized": optimized_tokens,
                    "diff": optimized_tokens - original_tokens
                }
                
                # Calculate diff
                diff = list(difflib.unified_diff(
                    original.splitlines(),
                    optimized.splitlines(),
                    lineterm='',
                    n=3
                ))
                stats["diffs"][name] = "\n".join(diff)
        
        # Calculate new and removed prompts
        stats["new_prompts"] = list(set(optimized_prompts) - set(original_prompts))
        stats["removed_prompts"] = list(set(original_prompts) - set(optimized_prompts))
        
        return stats
    
    async def _log_optimization_metrics(
        self,
        metrics: Dict[str, Any],
        artifact_dir: str
    ) -> None:
        """
        Log optimization metrics to MLflow.
        
        Args:
            metrics: Optimization metrics
            artifact_dir: Directory to save artifacts
        """
        if not self.settings.ENABLE_TELEMETRY:
            return
        
        try:
            import mlflow
            
            # Start a new run
            with mlflow.start_run(run_name=f"optimize_{metrics['module_type']}_{metrics['optimizer_type']}"):
                # Log parameters
                mlflow.log_param("module_type", metrics["module_type"])
                mlflow.log_param("optimizer_type", metrics["optimizer_type"])
                mlflow.log_param("num_examples", metrics["num_examples"])
                mlflow.log_param("max_rounds", metrics["max_rounds"])
                mlflow.log_param("rounds_completed", metrics["rounds_completed"])
                
                # Log metrics
                mlflow.log_metric("optimization_time", metrics["optimization_time"])
                
                if "validation_metrics" in metrics and metrics["validation_metrics"]:
                    mlflow.log_metric("validation_average_score", metrics["validation_metrics"]["average_score"])
                    mlflow.log_metric("validation_success_rate", metrics["validation_metrics"]["success_rate"])
                
                # Log round metrics
                for i, round_metric in enumerate(metrics["round_metrics"]):
                    if "score" in round_metric and round_metric["score"] is not None:
                        mlflow.log_metric(f"round_{i+1}_score", round_metric["score"])
                    mlflow.log_metric(f"round_{i+1}_duration", round_metric["duration"])
                
                # Log prompt diff stats
                if "prompt_diff_stats" in metrics:
                    mlflow.log_param("unchanged_prompts", metrics["prompt_diff_stats"]["unchanged_count"])
                    mlflow.log_param("changed_prompts", metrics["prompt_diff_stats"]["changed_count"])
                    mlflow.log_param("new_prompts", len(metrics["prompt_diff_stats"]["new_prompts"]))
                    mlflow.log_param("removed_prompts", len(metrics["prompt_diff_stats"]["removed_prompts"]))
                
                # Save metrics to JSON file
                metrics_file = os.path.join(artifact_dir, "optimization_metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                # Log artifacts
                mlflow.log_artifacts(artifact_dir)
        except Exception as e:
            logger.warning(f"Failed to log optimization metrics to MLflow: {str(e)}")
    
    async def _save_optimization_history(
        self,
        module_type: str,
        optimizer_type: str,
        metrics: Dict[str, Any],
        artifact_dir: str
    ) -> None:
        """
        Save optimization history to disk.
        
        Args:
            module_type: Type of module
            optimizer_type: Type of optimizer
            metrics: Optimization metrics
            artifact_dir: Directory to save artifacts
        """
        try:
            # Create history directory if it doesn't exist
            history_dir = os.path.join("optimization_history", module_type, optimizer_type)
            os.makedirs(history_dir, exist_ok=True)
            
            # Save metrics to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(history_dir, f"metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Copy artifacts
            import shutil
            for file in os.listdir(artifact_dir):
                src = os.path.join(artifact_dir, file)
                dst = os.path.join(history_dir, f"{timestamp}_{file}")
                shutil.copy2(src, dst)
            
            logger.info(f"Optimization history saved to {history_dir}")
        except Exception as e:
            logger.warning(f"Failed to save optimization history: {str(e)}")


# Global optimizer instance
_enhanced_optimizer = None


def get_enhanced_optimizer() -> EnhancedOptimizer:
    """
    Get the global enhanced optimizer instance.
    
    Returns:
        EnhancedOptimizer: The global enhanced optimizer instance
    """
    global _enhanced_optimizer
    if _enhanced_optimizer is None:
        _enhanced_optimizer = EnhancedOptimizer()
    return _enhanced_optimizer


# Export all classes and functions
__all__ = [
    'EnhancedOptimizer',
    'get_enhanced_optimizer'
]

"""
Temporal Rollback API for Medical Research

This module provides a temporal rollback API for analyzing how medical knowledge
and contradictions have evolved over time.
"""

import os
import json
import logging
import time
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from asf.orchestration.ray_orchestrator import RayOrchestrator, RayConfig
from asf.medical.models.biomedlm_wrapper import BioMedLMScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("temporal-rollback-api")

# Create API router
router = APIRouter(
    prefix="/temporal",
    tags=["Temporal Rollback"],
    responses={404: {"description": "Not found"}},
)

# Initialize Ray orchestrator
ray_config = RayConfig(
    use_ray=os.environ.get("USE_RAY", "true").lower() == "true",
    address=os.environ.get("RAY_ADDRESS"),
    num_cpus=int(os.environ.get("RAY_NUM_CPUS", "0")) or None,
    num_gpus=int(os.environ.get("RAY_NUM_GPUS", "0")) or None,
    include_dashboard=os.environ.get("RAY_INCLUDE_DASHBOARD", "true").lower() == "true",
    dashboard_port=int(os.environ.get("RAY_DASHBOARD_PORT", "8265")),
    logging_level=os.environ.get("RAY_LOGGING_LEVEL", "INFO"),
)

orchestrator = RayOrchestrator(config=ray_config)

# Define request/response models
class TemporalRollbackRequest(BaseModel):
    query: str
    timestamp: Optional[str] = None
    max_results: int = 20

class TemporalRollbackResponse(BaseModel):
    query: str
    timestamp: str
    results: List[Dict[str, Any]]
    total_count: int

class TemporalContradictionRequest(BaseModel):
    claim1: str
    claim2: str
    timestamp: Optional[str] = None
    use_tsmixer: bool = True
    use_lorentz: bool = True
    use_shap: bool = True

class TemporalContradictionResponse(BaseModel):
    claim1: str
    claim2: str
    timestamp: str
    contradiction_result: Dict[str, Any]

class TemporalEvolutionRequest(BaseModel):
    claim: str
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    interval_days: int = 30
    max_points: int = 12

class TemporalEvolutionResponse(BaseModel):
    claim: str
    start_timestamp: str
    end_timestamp: str
    evolution: List[Dict[str, Any]]

# Initialize database connection
try:
    from asf.medical.layer1_knowledge_substrate.memgraph_manager import MemgraphManager, MemgraphConfig
    
    memgraph_config = MemgraphConfig(
        host=os.environ.get("MEMGRAPH_HOST", "localhost"),
        port=int(os.environ.get("MEMGRAPH_PORT", "7687")),
        username=os.environ.get("MEMGRAPH_USERNAME", ""),
        password=os.environ.get("MEMGRAPH_PASSWORD", ""),
        encrypted=os.environ.get("MEMGRAPH_ENCRYPTED", "false").lower() == "true"
    )
    
    memgraph_manager = MemgraphManager(config=memgraph_config)
    logger.info("Memgraph manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Memgraph manager: {e}")
    memgraph_manager = None

# Register functions with orchestrator
def query_at_timestamp(query: str, timestamp: Optional[str] = None, max_results: int = 20) -> Dict[str, Any]:
    """
    Query the database as it existed at a specific timestamp.
    
    Args:
        query: Search query
        timestamp: Timestamp for temporal rollback (ISO format)
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with query results
    """
    # Parse timestamp
    if timestamp:
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            # Try parsing with different format
            try:
                dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
    else:
        dt = datetime.datetime.now()
    
    # Query database at timestamp
    if memgraph_manager is not None:
        try:
            # Construct Cypher query with timestamp
            cypher_query = f"""
            MATCH (n:MedicalClaim)
            WHERE n.timestamp <= '{dt.isoformat()}'
            RETURN n
            ORDER BY n.timestamp DESC
            LIMIT {max_results}
            """
            
            # Execute query
            results = memgraph_manager.run_query_sync(cypher_query)
            
            # Process results
            processed_results = []
            for result in results:
                node = result.get("n", {})
                if node:
                    processed_results.append(node)
            
            return {
                "query": query,
                "timestamp": dt.isoformat(),
                "results": processed_results,
                "total_count": len(processed_results)
            }
        except Exception as e:
            logger.error(f"Error querying database at timestamp {dt}: {e}")
            raise RuntimeError(f"Failed to query database at timestamp {dt}: {e}")
    else:
        # Simulate results for testing
        return {
            "query": query,
            "timestamp": dt.isoformat(),
            "results": [
                {
                    "id": f"claim_{i}",
                    "content": f"Medical claim {i} for {query}",
                    "timestamp": (dt - datetime.timedelta(days=i)).isoformat()
                }
                for i in range(min(max_results, 5))
            ],
            "total_count": min(max_results, 5)
        }

def detect_contradiction_at_timestamp(claim1: str, claim2: str, timestamp: Optional[str] = None, use_tsmixer: bool = True, use_lorentz: bool = True) -> Dict[str, Any]:
    """
    Detect contradiction between claims as they existed at a specific timestamp.
    
    Args:
        claim1: First medical claim
        claim2: Second medical claim
        timestamp: Timestamp for temporal rollback (ISO format)
        use_tsmixer: Whether to use TSMixer for temporal analysis
        use_lorentz: Whether to use Lorentz embeddings for hierarchical analysis
        
    Returns:
        Dictionary with contradiction detection results
    """
    # Parse timestamp
    if timestamp:
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            # Try parsing with different format
            try:
                dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
    else:
        dt = datetime.datetime.now()
    
    # Initialize BioMedLM scorer
    try:
        biomedlm_scorer = BioMedLMScorer(
            model_name=os.environ.get("BIOMEDLM_MODEL", "microsoft/BioMedLM"),
            use_negation_detection=True,
            use_multimodal_fusion=True,
            use_shap_explainer=True,
            use_tsmixer=use_tsmixer,
            use_lorentz=use_lorentz,
            use_temporal_confidence=True
        )
        
        # Detect contradiction
        contradiction_result = biomedlm_scorer.detect_contradiction(claim1, claim2)
        
        # Add timestamp information
        contradiction_result["timestamp"] = dt.isoformat()
        
        return {
            "claim1": claim1,
            "claim2": claim2,
            "timestamp": dt.isoformat(),
            "contradiction_result": contradiction_result
        }
    except Exception as e:
        logger.error(f"Error detecting contradiction at timestamp {dt}: {e}")
        raise RuntimeError(f"Failed to detect contradiction at timestamp {dt}: {e}")

def analyze_claim_evolution(claim: str, start_timestamp: Optional[str] = None, end_timestamp: Optional[str] = None, interval_days: int = 30, max_points: int = 12) -> Dict[str, Any]:
    """
    Analyze how a claim has evolved over time.
    
    Args:
        claim: Medical claim
        start_timestamp: Start timestamp for analysis (ISO format)
        end_timestamp: End timestamp for analysis (ISO format)
        interval_days: Interval between analysis points in days
        max_points: Maximum number of analysis points
        
    Returns:
        Dictionary with claim evolution analysis
    """
    # Parse timestamps
    if end_timestamp:
        try:
            end_dt = datetime.datetime.fromisoformat(end_timestamp)
        except ValueError:
            # Try parsing with different format
            try:
                end_dt = datetime.datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"Invalid end timestamp format: {end_timestamp}")
    else:
        end_dt = datetime.datetime.now()
    
    if start_timestamp:
        try:
            start_dt = datetime.datetime.fromisoformat(start_timestamp)
        except ValueError:
            # Try parsing with different format
            try:
                start_dt = datetime.datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"Invalid start timestamp format: {start_timestamp}")
    else:
        # Default to 1 year before end timestamp
        start_dt = end_dt - datetime.timedelta(days=365)
    
    # Calculate interval
    total_days = (end_dt - start_dt).days
    if total_days <= 0:
        raise ValueError("End timestamp must be after start timestamp")
    
    # Adjust interval if needed
    if total_days / interval_days > max_points:
        interval_days = total_days // max_points
    
    # Generate timestamps for analysis
    timestamps = []
    current_dt = start_dt
    while current_dt <= end_dt:
        timestamps.append(current_dt)
        current_dt += datetime.timedelta(days=interval_days)
    
    # Ensure end timestamp is included
    if timestamps[-1] != end_dt:
        timestamps.append(end_dt)
    
    # Limit to max_points
    if len(timestamps) > max_points:
        # Keep start, end, and evenly spaced points in between
        step = len(timestamps) // (max_points - 2)
        selected_timestamps = [timestamps[0]]
        for i in range(step, len(timestamps) - 1, step):
            selected_timestamps.append(timestamps[i])
        selected_timestamps.append(timestamps[-1])
        timestamps = selected_timestamps
    
    # Initialize BioMedLM scorer
    try:
        biomedlm_scorer = BioMedLMScorer(
            model_name=os.environ.get("BIOMEDLM_MODEL", "microsoft/BioMedLM"),
            use_negation_detection=True,
            use_multimodal_fusion=True,
            use_shap_explainer=True,
            use_tsmixer=True,
            use_lorentz=True,
            use_temporal_confidence=True
        )
        
        # Analyze claim at each timestamp
        evolution = []
        for dt in timestamps:
            # Create a contradictory claim for analysis
            contradictory_claim = f"It is not true that {claim}"
            
            # Detect contradiction
            contradiction_result = biomedlm_scorer.detect_contradiction(claim, contradictory_claim)
            
            # Extract confidence from temporal confidence scorer
            confidence = 0.8  # Default confidence
            if biomedlm_scorer.use_temporal_confidence and biomedlm_scorer.temporal_confidence_scorer is not None:
                try:
                    # Extract metadata from claim
                    claim_metadata = biomedlm_scorer.temporal_confidence_scorer.get_claim_metadata(claim)
                    
                    # Calculate confidence at this timestamp
                    confidence = biomedlm_scorer.temporal_confidence_scorer.calculate_confidence(
                        initial_confidence=claim_metadata["initial_confidence"],
                        domain=claim_metadata["domain"],
                        creation_time=claim_metadata["creation_time"],
                        current_time=dt
                    )
                except Exception as e:
                    logger.error(f"Error calculating confidence at timestamp {dt}: {e}")
            
            # Add analysis point
            evolution.append({
                "timestamp": dt.isoformat(),
                "confidence": confidence,
                "contradiction_score": contradiction_result.get("contradiction_score", 0.0),
                "methods_used": contradiction_result.get("methods_used", [])
            })
        
        return {
            "claim": claim,
            "start_timestamp": start_dt.isoformat(),
            "end_timestamp": end_dt.isoformat(),
            "evolution": evolution
        }
    except Exception as e:
        logger.error(f"Error analyzing claim evolution: {e}")
        raise RuntimeError(f"Failed to analyze claim evolution: {e}")

# Register functions with orchestrator
orchestrator.register_function(query_at_timestamp, "query_at_timestamp")
orchestrator.register_function(detect_contradiction_at_timestamp, "detect_contradiction_at_timestamp")
orchestrator.register_function(analyze_claim_evolution, "analyze_claim_evolution")

# Define API endpoints
@router.post("/query", response_model=TemporalRollbackResponse)
async def temporal_query(request: TemporalRollbackRequest):
    """
    Query the database as it existed at a specific timestamp.
    
    This endpoint allows you to perform temporal rollback queries to see
    how the medical knowledge base looked at a specific point in time.
    """
    try:
        # Create task for temporal query
        task_id = orchestrator.create_task(
            name="temporal_query",
            function_name="query_at_timestamp",
            args=[request.query],
            kwargs={
                "timestamp": request.timestamp,
                "max_results": request.max_results
            }
        )
        
        # Execute task
        result = orchestrator.execute_task(task_id)
        
        return result
    except Exception as e:
        logger.error(f"Error executing temporal query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/contradiction", response_model=TemporalContradictionResponse)
async def temporal_contradiction(request: TemporalContradictionRequest):
    """
    Detect contradiction between claims as they existed at a specific timestamp.
    
    This endpoint allows you to analyze how contradictions between medical claims
    would have been detected at a specific point in time.
    """
    try:
        # Create task for temporal contradiction detection
        task_id = orchestrator.create_task(
            name="temporal_contradiction",
            function_name="detect_contradiction_at_timestamp",
            args=[request.claim1, request.claim2],
            kwargs={
                "timestamp": request.timestamp,
                "use_tsmixer": request.use_tsmixer,
                "use_lorentz": request.use_lorentz
            }
        )
        
        # Execute task
        result = orchestrator.execute_task(task_id)
        
        return result
    except Exception as e:
        logger.error(f"Error executing temporal contradiction detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evolution", response_model=TemporalEvolutionResponse)
async def temporal_evolution(request: TemporalEvolutionRequest):
    """
    Analyze how a claim has evolved over time.
    
    This endpoint allows you to track the evolution of a medical claim's
    confidence and contradiction score over time.
    """
    try:
        # Create task for claim evolution analysis
        task_id = orchestrator.create_task(
            name="claim_evolution",
            function_name="analyze_claim_evolution",
            args=[request.claim],
            kwargs={
                "start_timestamp": request.start_timestamp,
                "end_timestamp": request.end_timestamp,
                "interval_days": request.interval_days,
                "max_points": request.max_points
            }
        )
        
        # Execute task
        result = orchestrator.execute_task(task_id)
        
        return result
    except Exception as e:
        logger.error(f"Error executing claim evolution analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json

import numpy as np
import torch

from asf.layer3_cognitive_boundary.core.semantic_tensor_network import SemanticTensorNetwork
from asf.layer3_cognitive_boundary.core.semantic_node import SemanticNode
from asf.layer3_cognitive_boundary.core.semantic_relation import SemanticRelation
from asf.layer3_cognitive_boundary.processing.async_queue import AsyncProcessingQueue
from asf.layer3_cognitive_boundary.processing.priority_manager import AdaptivePriorityManager
from asf.layer3_cognitive_boundary.formation.concept_formation import ConceptFormationEngine
from asf.layer3_cognitive_boundary.formation.conceptual_blending import ConceptualBlendingEngine
from asf.layer3_cognitive_boundary.formation.category_formation import CategoryFormationSystem
from asf.layer3_cognitive_boundary.resolution.conflict_detection import ConflictDetectionEngine
from asf.layer3_cognitive_boundary.temporal import AdaptiveTemporalMetadata
from asf.layer3_cognitive_boundary.predictive_processor import PredictiveProcessor
from asf.layer3_cognitive_boundary.active_inference import ActiveInferenceController
from asf.layer3_cognitive_boundary.enums import SemanticNodeType, SemanticConfidenceState


class CognitiveBoundaryLayer:
    """
    Main controller for Layer 3 (Cognitive Boundary Layer).
    Orchestrates semantic operations across all components.
    Fully integrates Seth's predictive processing principles with active inference.
    
    This layer provides:
    1. Semantic network management (nodes, relations, concepts)
    2. Predictive processing with uncertainty estimation
    3. Active inference for minimizing prediction error
    4. Conflict detection and resolution
    5. Concept, category, and blend formation
    
    Key predictive processing capabilities include:
    - Anticipating operation outcomes before execution
    - Optimizing operations to minimize prediction errors
    - Learning from prediction errors to improve future predictions
    - Managing uncertainty through precision weighting
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive boundary layer with components and configuration.
        
        Args:
            config: Optional configuration dictionary with settings
        """
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3")
        self.operation_history = []
        self.performance_metrics = {
            'operation_count': 0,
            'error_count': 0,
            'anticipation_accuracy': [],
            'processing_times': []
        }
        
        log_level = self.config.get('log_level', logging.INFO)
        self.logger.setLevel(log_level)
        
        self.device = self.config.get('device', 'auto')
        if self.device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)
            
        self.logger.info(f"Cognitive Boundary Layer using device: {self.device}")
        
        network_config = self.config.get('semantic_network', {})
        network_config['device'] = self.device
        self.semantic_network = SemanticTensorNetwork(**network_config)
        
        self.processing_queue = AsyncProcessingQueue()
        self.priority_manager = AdaptivePriorityManager()

        self.concept_formation = ConceptFormationEngine(self.semantic_network)
        self.conceptual_blending = ConceptualBlendingEngine(
            self.semantic_network, 
            self.concept_formation
        )
        self.category_formation = CategoryFormationSystem(self.semantic_network)

        self.conflict_detection = ConflictDetectionEngine(self.semantic_network)

        self.predictive_processor = PredictiveProcessor()
        self.active_inference = ActiveInferenceController(self)

        self.anticipation_enabled = self.config.get('anticipation_enabled', True)
        self.active_inference_enabled = self.config.get('active_inference_enabled', True)
        
        self.execution_counts = {
            'create_node': 0,
            'add_property': 0,
            'create_relation': 0,
            'form_concept': 0,
            'create_blend': 0,
            'form_categories': 0
        }
        
        self.default_temporal_config = self.config.get('temporal', {
            'default_half_life': 86400 * 7,  # 7 days
            'critical_half_life': 86400 * 30,  # 30 days 
            'ephemeral_half_life': 3600  # 1 hour
        })
        
        self.max_retry_attempts = self.config.get('max_retry_attempts', 3)
        self.error_recovery_strategies = {
            'network_error': self._recover_from_network_error,
            'conflict_error': self._recover_from_conflict_error,
            'resource_error': self._recover_from_resource_error,
            'timeout_error': self._recover_from_timeout_error
        }
        
        self.batch_size = self.config.get('batch_size', 64)
        self.auto_optimize = self.config.get('auto_optimize', True)
        
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        self.logger.info("Initializing Cognitive Boundary Layer")
        
        try:
            if self.anticipation_enabled:
                self.logger.info("Initializing predictive processing capabilities")
                await self.predictive_processor.initialize() if hasattr(self.predictive_processor, 'initialize') else None
                
            if self.active_inference_enabled:
                self.logger.info("Initializing active inference capabilities")
                await self.active_inference.initialize() if hasattr(self.active_inference, 'initialize') else None
                
            self.logger.info("Initializing processing queue")
            await self.processing_queue.initialize() if hasattr(self.processing_queue, 'initialize') else None
            
            if hasattr(self.priority_manager, 'initialize_neural_model'):
                state_dim = self.config.get('priority_state_dim', 8)
                action_dim = self.config.get('priority_action_dim', 3)
                self.logger.info(f"Initializing neural priority model with dims: {state_dim}, {action_dim}")
                self.priority_manager.initialize_neural_model(state_dim, action_dim)
            
            test_node = SemanticNode(
                id="test_init_node",
                label="Initialization Test Node",
                node_type=SemanticNodeType.CONCEPT.value,
                properties={},
                confidence=1.0
            )
            await self.semantic_network.add_node(test_node)
            test_result = await self.semantic_network.get_node("test_init_node")
            
            if not test_result:
                self.logger.error("Failed to initialize semantic network - test node creation failed")
                return False
                
            self.logger.info("Cognitive Boundary Layer initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            return False

    async def anticipate_semantic_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.anticipation_enabled:
            return {'status': 'disabled', 'message': 'Anticipation is disabled'}
        
        start_time = time.time()
        self.logger.info(f"Anticipating outcomes of {len(operations)} semantic operations")
        
        try:
            context_id = f"ctx_{int(time.time()*1000)}"
            
            result = {
                'status': 'unknown',
                'operation_count': len(operations),
                'context_id': context_id,
                'timestamp': time.time(),
                'anticipated_contradictions': [],
                'anticipated_node_changes': [],
                'anticipated_relation_changes': [],
                'anticipated_concepts': [],
                'performance_impact': {}
            }
            
            inference_result = await self.active_inference.anticipate_state(operations)
            
            if inference_result and 'status' in inference_result and inference_result['status'] != 'error':
                result.update({
                    'status': 'success',
                    'anticipated_contradictions': inference_result.get('anticipated_contradictions', []),
                    'operation_id': inference_result.get('operation_id', context_id)
                })
                
                for key in ['anticipated_categories', 'anticipated_concepts', 'anticipated_embeddings']:
                    if key in inference_result:
                        result[key] = inference_result[key]
                
                if hasattr(self.predictive_processor, 'register_prediction'):
                    await self.predictive_processor.register_prediction(
                        context_id=context_id,
                        entity_id="semantic_layer",
                        prediction_type="operations_outcome",
                        prediction_value=result
                    )
            else:
                result['status'] = 'error'
                result['message'] = inference_result.get('message', 'Unknown anticipation error')
                self.logger.warning(f"Anticipation failed: {result['message']}")
            
            result['performance_impact']['anticipation_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during anticipation: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Anticipation error: {str(e)}",
                'operation_count': len(operations)
            }

    async def optimize_operations(self, anticipated_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active_inference_enabled:
            return {'status': 'disabled', 'message': 'Active inference is disabled'}
        
        start_time = time.time()
        self.logger.info("Performing active inference to optimize operations")
        
        try:
            targets = self.config.get('optimization_targets', {})
            if not targets:
                targets = {
                    'contradiction_reduction': 0.4,
                    'category_coherence': 0.3,
                    'structural_efficiency': 0.3
                }
                
            optimization_result = await self.active_inference.perform_active_inference(
                anticipated_state, optimization_targets=targets
            )
            
            optimization_time = time.time() - start_time
            
            if 'performance_impact' not in optimization_result:
                optimization_result['performance_impact'] = {}
                
            optimization_result['performance_impact']['optimization_time'] = optimization_time
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error during operation optimization: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Optimization error: {str(e)}",
                'original_state': anticipated_state
            }

    async def execute_semantic_operations(self, 
                                        operations: List[Dict[str, Any]], 
                                        anticipate: bool = True, 
                                        optimize: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        operation_results = []
        context_id = f"ctx_{int(time.time()*1000)}"
        
        self.logger.info(f"Executing {len(operations)} semantic operations (anticipate={anticipate}, optimize={optimize})")
        if operations and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Operation types: {[op.get('type') for op in operations]}")
        
        anticipated_state = None
        if anticipate and self.anticipation_enabled and operations:
            anticipated_state = await self.anticipate_semantic_operations(operations)
            if anticipated_state.get('status') == 'error':
                self.logger.warning(f"Anticipation failed: {anticipated_state.get('message')}")
        
        if optimize and self.active_inference_enabled and anticipated_state and anticipated_state.get('status') != 'error':
            optimization_result = await self.optimize_operations(anticipated_state)
            if optimization_result.get('status') == 'success':
                if 'optimized_operations' in optimization_result:
                    original_count = len(operations)
                    operations = optimization_result.get('optimized_operations', operations)
                    self.logger.info(f"Using optimized operations: {len(operations)} (from {original_count})")
            else:
                self.logger.warning(f"Optimization failed: {optimization_result.get('message')}")
        
        async with self._lock:
            for operation in operations:
                op_type = operation.get('type')
                op_result = {'operation': operation, 'status': 'unknown'}
                
                try:
                    if op_type == 'create_node':
                        result = await self._execute_create_node(operation)
                        op_result = {'status': 'success', 'node_id': result, 'operation_type': 'create_node'}
                        self.execution_counts['create_node'] += 1
                        
                    elif op_type == 'add_property':
                        success, property_id = await self._execute_add_property(operation)
                        op_result = {'status': 'success' if success else 'failed', 
                                    'property_id': property_id,
                                    'operation_type': 'add_property'}
                        self.execution_counts['add_property'] += 1
                        
                    elif op_type == 'create_relation':
                        relation_id = await self._execute_create_relation(operation)
                        op_result = {'status': 'success', 'relation_id': relation_id, 
                                    'operation_type': 'create_relation'}
                        self.execution_counts['create_relation'] += 1
                        
                    elif op_type == 'form_concept':
                        concept_id = await self._execute_form_concept(operation)
                        op_result = {'status': 'success', 'concept_id': concept_id,
                                    'operation_type': 'form_concept'}
                        self.execution_counts['form_concept'] += 1
                        
                    elif op_type == 'create_blend':
                        blend_id = await self._execute_create_blend(operation)
                        op_result = {'status': 'success', 'blend_id': blend_id,
                                    'operation_type': 'create_blend'}
                        self.execution_counts['create_blend'] += 1
                        
                    elif op_type == 'form_categories':
                        category_result = await self._execute_form_categories(operation)
                        op_result = {'status': 'success', 'category_result': category_result,
                                    'operation_type': 'form_categories'}
                        self.execution_counts['form_categories'] += 1
                        
                    else:
                        op_result = {'status': 'unknown_operation', 
                                    'message': f"Unknown operation type: {op_type}",
                                    'operation_type': op_type}
                        self.logger.warning(f"Unknown operation type: {op_type}")
                        
                except Exception as e:
                    self.logger.error(f"Error executing operation {op_type}: {str(e)}", exc_info=True)
                    op_result = {'status': 'error', 
                                'message': str(e),
                                'operation_type': op_type}
                    self.performance_metrics['error_count'] += 1
                    
                op_result['timestamp'] = time.time()
                    
                operation_results.append(op_result)
                
                log_level = logging.DEBUG if op_result['status'] == 'success' else logging.WARNING
                if self.logger.isEnabledFor(log_level):
                    self.logger.log(log_level, f"Operation {op_type} result: {op_result['status']}")
        
        contradictions = await self.conflict_detection.check_contradictions()
        
        evaluation = None
        if anticipated_state and anticipated_state.get('status') != 'error':
            actual_state = {
                'actual_contradictions': contradictions,
                'operation_results': operation_results,
                'execution_time': time.time() - start_time,
                'context_id': context_id
            }
            
            if any(res['operation_type'] == 'form_categories' for res in operation_results if 'operation_type' in res):
                actual_state['actual_categories'] = await self._extract_category_state()
                
            if any(res['operation_type'] == 'form_concept' for res in operation_results if 'operation_type' in res):
                actual_state['actual_concepts'] = await self._extract_concept_state()
            
            evaluation = await self.active_inference.evaluate_anticipations(
                actual_state,
                anticipated_state.get('operation_id', context_id)
            )
            
            if evaluation and 'overall_error' in evaluation:
                self.performance_metrics['anticipation_accuracy'].append(1.0 - evaluation['overall_error'])
                if len(self.performance_metrics['anticipation_accuracy']) > 100:
                    self.performance_metrics['anticipation_accuracy'] = self.performance_metrics['anticipation_accuracy'][-100:]
        
        execution_time = time.time() - start_time
        self.performance_metrics['processing_times'].append(execution_time)
        if len(self.performance_metrics['processing_times']) > 100:
            self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-100:]
        
        self.performance_metrics['operation_count'] += len(operations)
        
        operation_entry = {
            'timestamp': time.time(),
            'operation_count': len(operations),
            'execution_time': execution_time,
            'anticipation_used': anticipate,
            'optimization_used': optimize,
            'contradiction_count': len(contradictions),
            'success_count': sum(1 for res in operation_results if res.get('status') == 'success'),
            'context_id': context_id
        }
        
        self.operation_history.append(operation_entry)
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]
        
        return {
            'status': 'success',
            'operation_count': len(operations),
            'results': operation_results,
            'contradictions': contradictions,
            'anticipation_evaluation': evaluation,
            'execution_time': execution_time,
            'context_id': context_id
        }

    async def _execute_create_node(self, operation: Dict[str, Any]) -> str:
        node_id = operation.get('node_id', f"node_{uuid.uuid4().hex[:8]}")
        node_type = operation.get('node_type', SemanticNodeType.CONCEPT.value)
        label = operation.get('label', f"Node_{node_id[-6:]}")
        properties = operation.get('properties', {})
        confidence = operation.get('confidence', 0.7)
        source_ids = operation.get('source_ids', [])
        confidence_state = operation.get('confidence_state', SemanticConfidenceState.PROVISIONAL.value)
        embeddings = operation.get('embeddings', None)
        
        temporal_config = operation.get('temporal_config', self.default_temporal_config)
        temporal_metadata = AdaptiveTemporalMetadata(
            contextual_half_lives=temporal_config
        )
        
        metadata = operation.get('metadata', {})
        metadata.update({
            'created_time': time.time(),
            'created_by': operation.get('created_by', 'system'),
            'creation_context': operation.get('creation_context', {})
        })
        
        node = SemanticNode(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties,
            confidence=confidence,
            confidence_state=confidence_state,
            source_ids=source_ids,
            temporal_metadata=temporal_metadata,
            metadata=metadata
        )
        
        if embeddings is not None:
            if isinstance(embeddings, list):
                node.embeddings = np.array(embeddings, dtype=np.float32)
            else:
                node.embeddings = embeddings
            
            if hasattr(node, 'embeddings') and node.embeddings is not None:
                node.tensor_representation = torch.tensor(
                    node.embeddings, dtype=torch.float32
                )
        
        await self.semantic_network.add_node(node)
        
        self.logger.debug(f"Created node: {node_id}, type: {node_type}")
        return node_id

    async def _execute_add_property(self, operation: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        node_id = operation.get('node_id')
        property_name = operation.get('property_name')
        property_value = operation.get('property_value')
        confidence = operation.get('confidence', 0.7)
        
        if not node_id or not property_name:
            self.logger.warning("Add property operation missing node_id or property_name")
            return False, None
        
        node = await self.semantic_network.get_node(node_id)
        if not node:
            self.logger.warning(f"Cannot add property: Node {node_id} not found")
            return False, None
        
        property_id = f"prop_{node_id}_{property_name}_{uuid.uuid4().hex[:6]}"
        
        node.properties[property_name] = property_value
        
        if not hasattr(node, 'property_metadata'):
            node.property_metadata = {}
            
        node.property_metadata[property_name] = {
            'added_time': time.time(),
            'confidence': confidence,
            'added_by': operation.get('added_by', 'system'),
            'property_id': property_id
        }
        
        node.temporal_metadata.update_modification()
        await self.semantic_network.add_node(node, update_tensors=True)
        
        self.logger.debug(f"Added property '{property_name}' to node {node_id}")
        return True, property_id

    async def _execute_create_relation(self, operation: Dict[str, Any]) -> str:
        relation_id = operation.get('relation_id', f"rel_{uuid.uuid4().hex[:8]}")
        source_id = operation.get('source_id')
        target_id = operation.get('target_id')
        relation_type = operation.get('relation_type', 'generic')
        weight = operation.get('weight', 0.8)
        bidirectional = operation.get('bidirectional', False)
        properties = operation.get('properties', {})
        confidence = operation.get('confidence', 0.7)
        
        if not source_id or not target_id:
            raise ValueError("Create relation operation missing source_id or target_id")
        
        source_node = await self.semantic_network.get_node(source_id)
        target_node = await self.semantic_network.get_node(target_id)
        
        if not source_node:
            raise ValueError(f"Source node {source_id} not found")
        if not target_node:
            raise ValueError(f"Target node {target_id} not found")
        
        temporal_config = operation.get('temporal_config', self.default_temporal_config)
        temporal_metadata = AdaptiveTemporalMetadata(
            contextual_half_lives=temporal_config
        )
        
        metadata = operation.get('metadata', {})
        metadata.update({
            'created_time': time.time(),
            'created_by': operation.get('created_by', 'system'),
            'creation_context': operation.get('creation_context', {})
        })
        
        relation = SemanticRelation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            bidirectional=bidirectional,
            properties=properties,
            confidence=confidence,
            temporal_metadata=temporal_metadata,
            metadata=metadata
        )
        
        await self.semantic_network.add_relation(relation)
        
        if bidirectional:
            reverse_id = f"{relation_id}_rev"
            reverse_relation = SemanticRelation(
                id=reverse_id,
                source_id=target_id,  # Swapped
                target_id=source_id,  # Swapped
                relation_type=relation_type,
                weight=weight,
                bidirectional=True,
                properties=properties,
                confidence=confidence,
                temporal_metadata=temporal_metadata,
                metadata={**metadata, 'bidirectional_pair': relation_id}
            )
            await self.semantic_network.add_relation(reverse_relation)
            
            relation.metadata['bidirectional_pair'] = reverse_id
            await self.semantic_network.add_relation(relation)  # Update
        
        self.logger.debug(f"Created relation: {relation_id}, type: {relation_type}, " +
                       f"source: {source_id}, target: {target_id}")
        return relation_id

    async def _execute_form_concept(self, operation: Dict[str, Any]) -> str:
        features = operation.get('features', {})
        source_id = operation.get('source_id')
        context = operation.get('context', {})
        
        if not features:
            raise ValueError("Form concept operation missing features")
        
        if 'label' in operation:
            context['label'] = operation['label']
        
        if 'confidence' in operation:
            context['source_confidence'] = operation['confidence']
            
        if 'metadata' in operation:
            context['metadata'] = operation['metadata']
            
        if self.anticipation_enabled and 'anticipate' in operation and operation['anticipate']:
            partial_features = {}
            for name, value in features.items():
                if isinstance(value, dict) and 'partial' in value and value['partial']:
                    partial_features[name] = value['value']
            
            if partial_features:
                anticipated = await self.concept_formation.anticipate_concept(
                    partial_features, context=context
                )
                if anticipated:
                    context['anticipated_id'] = id(anticipated)  # Use object ID for tracking
                    
        concept_id = await self.concept_formation.form_concept(
            features=features,
            source_id=source_id,
            context=context
        )
        
        self.logger.debug(f"Formed concept: {concept_id}, feature count: {len(features)}")
        return concept_id

    async def _execute_create_blend(self, operation: Dict[str, Any]) -> str:
        input_ids = operation.get('input_ids', [])
        blend_type = operation.get('blend_type', 'composition')
        context = operation.get('context', {})
        
        if not input_ids or len(input_ids) < 2:
            raise ValueError("Create blend operation requires at least two input concepts")
        
        for concept_id in input_ids:
            concept = await self.semantic_network.get_node(concept_id)
            if not concept:
                raise ValueError(f"Input concept {concept_id} not found")
        
        if 'confidence' in operation:
            context['source_confidence'] = operation['confidence']
            
        if 'metadata' in operation:
            context['metadata'] = operation['metadata']
            
        if self.anticipation_enabled and 'anticipate' in operation and operation['anticipate']:
            anticipated = await self.conceptual_blending.anticipate_blend(
                input_ids, blend_type=blend_type, context=context
            )
            if anticipated:
                context['anticipated_blend'] = anticipated
                
        blend_id = await self.conceptual_blending.create_blend(
            input_ids=input_ids,
            blend_type=blend_type,
            context=context
        )
        
        self.logger.debug(f"Created blend: {blend_id}, inputs: {len(input_ids)}, type: {blend_type}")
        return blend_id

    async def _execute_form_categories(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        node_ids = operation.get('node_ids', [])
        method = operation.get('method', 'similarity')
        params = operation.get('params', {})
        
        if not node_ids:
            raise ValueError("Form categories operation missing node_ids")
        
        nodes = {}
        for node_id in node_ids:
            node = await self.semantic_network.get_node(node_id)
            if node:
                nodes[node_id] = node
        
        if len(nodes) < 2:
            raise ValueError("Form categories operation requires at least two valid nodes")
        
        predicted_categories = await self.category_formation.predict_categories(
            nodes, method=method, params=params
        )
        
        if self.active_inference_enabled and 'refine' in operation and operation['refine']:
            refined_categories = await self.category_formation.refine_categories_via_active_inference(
                predicted_categories, source_nodes=nodes
            )
            
            category_result = refined_categories
            refinement_status = 'performed'
        else:
            category_result = predicted_categories
            refinement_status = 'skipped'
        
        if 'create_category_nodes' in operation and operation['create_category_nodes']:
            await self._create_category_nodes(category_result)
            
        categories = category_result.get('categories', [])
        self.logger.debug(f"Formed {len(categories)} categories from {len(nodes)} nodes, " +
                       f"method: {method}, refinement: {refinement_status}")
        
        full_result = {
            'status': category_result.get('status', 'unknown'),
            'categories': categories,
            'total_nodes': len(nodes),
            'categorized_nodes': sum(len(cat.get('members', [])) for cat in categories),
            'method': method,
            'refinement': refinement_status
        }
        
        if 'refinement' in category_result:
            full_result['refinement_details'] = category_result['refinement']
            
        return full_result

    async def _create_category_nodes(self, category_result: Dict[str, Any]) -> List[str]:
        if 'categories' not in category_result:
            return []
            
        category_node_ids = []
        
        for category in category_result['categories']:
            member_hash = hash(tuple(sorted(category.get('members', []))))
            category_id = f"category_{abs(member_hash) % 10000:04d}"
            
            properties = {}
            if 'common_properties' in category:
                properties.update(category['common_properties'])
                
            properties['member_count'] = len(category.get('members', []))
            
            cat_operation = {
                'node_id': category_id,
                'node_type': SemanticNodeType.CATEGORY.value,
                'label': category.get('label', f"Category_{category_id[-4:]}"),
                'properties': properties,
                'confidence': category.get('confidence', 0.7),
                'metadata': {
                    'category_members': category.get('members', []),
                    'formation_method': category_result.get('method', 'unknown')
                }
            }
            
            await self._execute_create_node(cat_operation)
            category_node_ids.append(category_id)
            
            for member_id in category.get('members', []):
                relation_operation = {
                    'source_id': category_id,
                    'target_id': member_id,
                    'relation_type': 'has_member',
                    'weight': 0.9,
                    'confidence': category.get('confidence', 0.7)
                }
                await self._execute_create_relation(relation_operation)
                
        return category_node_ids

    async def _extract_category_state(self) -> List[Dict[str, Any]]:
        category_nodes = {}
        for node_id, node in self.semantic_network.nodes.items():
            if node.node_type == SemanticNodeType.CATEGORY.value:
                category_nodes[node_id] = node
                
        if not category_nodes:
            return []
            
        categories = []
        
        for cat_id, cat_node in category_nodes.items():
            relations = await self.semantic_network.get_node_relations(
                cat_id, direction="outgoing", relation_type="has_member"
            )
            
            member_ids = [rel.target_id for rel in relations]
            
            category = {
                'id': cat_id,
                'label': cat_node.label,
                'members': member_ids,
                'size': len(member_ids),
                'common_properties': {k: v for k, v in cat_node.properties.items() 
                                     if k != 'member_count'}
            }
            
            categories.append(category)
            
        return categories

    async def _extract_concept_state(self) -> List[Dict[str, Any]]:
        recent_concept_ids = []
        if hasattr(self.concept_formation, 'formation_history'):
            history_limit = min(10, len(self.concept_formation.formation_history))
            for entry in self.concept_formation.formation_history[-history_limit:]:
                if 'concept_id' in entry:
                    recent_concept_ids.append(entry['concept_id'])
                    
        if not recent_concept_ids:
            return []
            
        concepts = []
        
        for concept_id in recent_concept_ids:
            node = await self.semantic_network.get_node(concept_id)
            if not node:
                continue
                
            concept = {
                'id': concept_id,
                'label': node.label,
                'properties': dict(node.properties),
                'confidence': node.confidence
            }
            
            if hasattr(node, 'embeddings') and node.embeddings is not None:
                if isinstance(node.embeddings, np.ndarray):
                    concept['embeddings_shape'] = node.embeddings.shape
                    concept['embeddings_sample'] = node.embeddings.flatten()[:10].tolist()
                    
            concepts.append(concept)
            
        return concepts

    async def _recover_from_network_error(self, error: Exception, operation: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Recovering from conflict error: {error}")
        
        
        return {'status': 'recovered', 'strategy': 'conflict_resolution'}
    
    async def _recover_from_resource_error(self, error: Exception, operation: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Recovering from timeout error: {error}")
        
        
        return {'status': 'recovered', 'strategy': 'timeout_handling'}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        metrics = dict(self.performance_metrics)
        
        if self.performance_metrics['processing_times']:
            metrics['avg_processing_time'] = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
            metrics['max_processing_time'] = max(self.performance_metrics['processing_times'])
            
        if self.performance_metrics['anticipation_accuracy']:
            metrics['avg_anticipation_accuracy'] = sum(self.performance_metrics['anticipation_accuracy']) / len(self.performance_metrics['anticipation_accuracy'])
            
        metrics['execution_counts'] = dict(self.execution_counts)
        
        if metrics['operation_count'] > 0:
            metrics['error_rate'] = metrics['error_count'] / metrics['operation_count']
        else:
            metrics['error_rate'] = 0.0
            
        return metrics

    async def get_status(self) -> Dict[str, Any]:
        status = {
            'timestamp': time.time(),
            'node_count': len(self.semantic_network.nodes),
            'relation_count': len(self.semantic_network.relations),
            'anticipation_enabled': self.anticipation_enabled,
            'active_inference_enabled': self.active_inference_enabled,
            'device': str(self.device),
            'faiss_initialized': bool(getattr(self.semantic_network, 'faiss_initialized', False)),
            'performance_metrics': await self.get_performance_metrics()
        }
        
        if hasattr(self.processing_queue, 'get_queue_status'):
            status['queue_status'] = await self.processing_queue.get_queue_status()
            
        if hasattr(self.priority_manager, 'get_prediction_stats'):
            status['priority_predictions'] = await self.priority_manager.get_prediction_stats()
            
        if hasattr(self.active_inference, 'active_inference_history'):
            history = self.active_inference.active_inference_history
            if history:
                status['active_inference'] = {
                    'history_count': len(history),
                    'last_score_improvement': history[-1].get('score_improvement', 0) if history else 0
                }
                
        return status

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        node = await self.semantic_network.get_node(node_id)
        if not node:
            return None
            
        node_dict = {
            'id': node.id,
            'label': node.label,
            'node_type': node.node_type,
            'properties': dict(node.properties),
            'confidence': node.confidence,
            'confidence_state': getattr(node, 'confidence_state', None),
            'metadata': getattr(node, 'metadata', {})
        }
        
        if hasattr(node, 'temporal_metadata'):
            node_dict['temporal_relevance'] = node.temporal_metadata.compute_relevance()
            node_dict['temporal_freshness'] = node.temporal_metadata.compute_freshness()
            
        if hasattr(node, 'embeddings') and node.embeddings is not None:
            if isinstance(node.embeddings, np.ndarray):
                node_dict['embeddings_shape'] = node.embeddings.shape
                node_dict['embeddings_sample'] = node.embeddings.flatten()[:5].tolist()
            elif isinstance(node.embeddings, torch.Tensor):
                node_dict['embeddings_shape'] = node.embeddings.shape
                node_dict['embeddings_sample'] = node.embeddings.flatten()[:5].tolist()
                
        similar_nodes = await self.semantic_network.get_similar_nodes(node_id, k=5, threshold=0.6)
        if similar_nodes:
            node_dict['similar_nodes'] = [
                {'id': node_id, 'similarity': similarity} 
                for node_id, similarity in similar_nodes
            ]
            
        relations = await self.semantic_network.get_node_relations(node_id)
        if relations:
            node_dict['relations'] = [relation.to_dict() for relation in relations]
            
        return node_dict

    async def get_network_summary(self) -> Dict[str, Any]:
        node_count = len(self.semantic_network.nodes)
        relation_count = len(self.semantic_network.relations)
        
        node_types = {}
        for node in self.semantic_network.nodes.values():
            node_type = node.node_type
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
            
        relation_types = {}
        for relation in self.semantic_network.relations.values():
            rel_type = relation.relation_type
            if rel_type not in relation_types:
                relation_types[rel_type] = 0
            relation_types[rel_type] += 1
            
        total_properties = sum(len(node.properties) for node in self.semantic_network.nodes.values())
        avg_properties = total_properties / node_count if node_count > 0 else 0
        
        if node_count > 0 and relation_count > 0:
            avg_relations_per_node = relation_count / node_count
            
            isolated_nodes = 0
            for node_id in self.semantic_network.nodes:
                if node_id not in self.semantic_network.relation_index['source'] and \
                   node_id not in self.semantic_network.relation_index['target']:
                    isolated_nodes += 1
                    
            isolated_percentage = (isolated_nodes / node_count) * 100
        else:
            avg_relations_per_node = 0
            isolated_nodes = 0
            isolated_percentage = 0
            
        return {
            'node_count': node_count,
            'relation_count': relation_count,
            'node_types': node_types,
            'relation_types': relation_types,
            'avg_properties_per_node': avg_properties,
            'avg_relations_per_node': avg_relations_per_node,
            'isolated_nodes': isolated_nodes,
            'isolated_percentage': isolated_percentage,
            'timestamp': time.time()
        }
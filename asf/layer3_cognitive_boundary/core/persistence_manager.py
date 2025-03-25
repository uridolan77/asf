"""
Knowledge Persistence Framework for Layer 3 Cognitive Boundary.
Provides durable storage for semantic network components with multiple backends,
efficient serialization, and incremental update capabilities.
"""

import asyncio
import json
import os
import time
import logging
import sqlite3
import pickle
import hashlib
import zlib
import gzip
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import aiofiles
import aiosqlite

from asf.layer3_cognitive_boundary.core.semantic_node import SemanticNode
from asf.layer3_cognitive_boundary.core.semantic_relation import SemanticRelation
from asf.layer3_cognitive_boundary.temporal import AdaptiveTemporalMetadata
from asf.layer3_cognitive_boundary.enums import SemanticNodeType, SemanticConfidenceState


class SerializationManager:
    """
    Handles serialization and deserialization of semantic network components.
    Supports multiple formats and efficient handling of tensor data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the serialization manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3.Persistence.Serialization")
        
        # Default serialization format
        self.default_format = self.config.get('serialization_format', 'json')
        self.compress_tensors = self.config.get('compress_tensors', True)
        self.compress_threshold = self.config.get('compress_threshold', 1024)  # Bytes
        
        # Serialization formats
        self.formats = {
            'json': self._serialize_json,
            'pickle': self._serialize_pickle,
            'binary': self._serialize_binary
        }
        
        # Deserialization formats
        self.deserialize_formats = {
            'json': self._deserialize_json,
            'pickle': self._deserialize_pickle,
            'binary': self._deserialize_binary
        }
    
    async def serialize_node(self, node: SemanticNode, format: Optional[str] = None) -> Dict[str, Any]:
        """
        Serialize a semantic node to the specified format.
        
        Args:
            node: SemanticNode to serialize
            format: Serialization format ('json', 'pickle', 'binary')
            
        Returns:
            Serialized node data
        """
        format = format or self.default_format
        
        # Base node data (common fields)
        node_data = {
            'id': node.id,
            'label': node.label,
            'node_type': node.node_type,
            'properties': node.properties,
            'confidence': node.confidence,
            'metadata': getattr(node, 'metadata', {}),
        }
        
        # Handle confidence state enum
        if hasattr(node, 'confidence_state'):
            if isinstance(node.confidence_state, SemanticConfidenceState):
                node_data['confidence_state'] = node.confidence_state.value
            else:
                node_data['confidence_state'] = node.confidence_state
        
        # Handle tensor representations separately for efficiency
        if hasattr(node, 'embeddings') and node.embeddings is not None:
            # Determine if we need to compress based on size
            if isinstance(node.embeddings, np.ndarray):
                tensor_data = await self._serialize_tensor(node.embeddings)
                node_data['embeddings'] = tensor_data
            elif isinstance(node.embeddings, torch.Tensor):
                # Convert torch tensor to numpy for serialization
                numpy_data = node.embeddings.detach().cpu().numpy()
                tensor_data = await self._serialize_tensor(numpy_data)
                node_data['embeddings'] = tensor_data
                
        # Handle temporal metadata
        if hasattr(node, 'temporal_metadata'):
            node_data['temporal_metadata'] = await self._serialize_temporal_metadata(
                node.temporal_metadata
            )
            
        # Additional fields specific to node types
        if hasattr(node, 'source_ids'):
            node_data['source_ids'] = node.source_ids
            
        if hasattr(node, 'parent_ids'):
            node_data['parent_ids'] = node.parent_ids
            
        if hasattr(node, 'child_ids'):
            node_data['child_ids'] = node.child_ids
            
        if hasattr(node, 'activation'):
            node_data['activation'] = float(node.activation)
            
        # Handle predictive fields
        predictive_fields = [
            'anticipated_activations', 'activation_errors', 
            'precision_values', 'anticipated_properties'
        ]
        for field in predictive_fields:
            if hasattr(node, field):
                node_data[field] = getattr(node, field)
            
        # Use the selected format for serialization
        if format in self.formats:
            return await self.formats[format](node_data)
        else:
            self.logger.warning(f"Unknown serialization format: {format}, using default")
            return await self.formats[self.default_format](node_data)
    
    async def deserialize_node(self, data: Union[Dict[str, Any], bytes, str], 
                              format: Optional[str] = None) -> SemanticNode:
        """
        Deserialize data into a semantic node.
        
        Args:
            data: Serialized node data
            format: Serialization format ('json', 'pickle', 'binary')
            
        Returns:
            Deserialized SemanticNode
        """
        format = format or self.default_format
        
        # Deserialize from the selected format
        if format in self.deserialize_formats:
            node_data = await self.deserialize_formats[format](data)
        else:
            self.logger.warning(f"Unknown deserialization format: {format}, using default")
            node_data = await self.deserialize_formats[self.default_format](data)
            
        # Handle embeddings if present
        embeddings = None
        if 'embeddings' in node_data:
            embeddings = await self._deserialize_tensor(node_data['embeddings'])
            del node_data['embeddings']  # Remove from dict to avoid duplicate
            
        # Handle temporal metadata if present
        temporal_metadata = None
        if 'temporal_metadata' in node_data:
            temporal_metadata = await self._deserialize_temporal_metadata(
                node_data['temporal_metadata']
            )
            del node_data['temporal_metadata']  # Remove from dict
            
        # Handle confidence state if present
        if 'confidence_state' in node_data:
            confidence_state = node_data['confidence_state']
            # Convert string to enum if needed
            if isinstance(confidence_state, str):
                try:
                    node_data['confidence_state'] = SemanticConfidenceState(confidence_state)
                except ValueError:
                    # Keep as string if not a valid enum value
                    pass
        
        # Extract fields that go directly to the constructor
        constructor_fields = {
            'id', 'label', 'node_type', 'properties', 'confidence', 
            'confidence_state', 'source_ids'
        }
        constructor_args = {k: v for k, v in node_data.items() if k in constructor_fields}
        
        # Create node with constructor arguments
        node = SemanticNode(**constructor_args)
        
        # Add embeddings
        if embeddings is not None:
            node.embeddings = embeddings
            # Create tensor representation if embeddings exist
            if isinstance(embeddings, np.ndarray):
                node.tensor_representation = torch.tensor(
                    embeddings, dtype=torch.float32
                )
        
        # Add temporal metadata
        if temporal_metadata is not None:
            node.temporal_metadata = temporal_metadata
        
        # Add remaining fields as attributes
        for key, value in node_data.items():
            if key not in constructor_fields and key not in {'embeddings', 'temporal_metadata'}:
                setattr(node, key, value)
                
        return node
    
    async def serialize_relation(self, relation: SemanticRelation, 
                                format: Optional[str] = None) -> Dict[str, Any]:
        """
        Serialize a semantic relation to the specified format.
        
        Args:
            relation: SemanticRelation to serialize
            format: Serialization format ('json', 'pickle', 'binary')
            
        Returns:
            Serialized relation data
        """
        format = format or self.default_format
        
        # Base relation data
        relation_data = {
            'id': relation.id,
            'source_id': relation.source_id,
            'target_id': relation.target_id,
            'relation_type': relation.relation_type,
            'weight': relation.weight,
            'bidirectional': relation.bidirectional,
            'properties': relation.properties,
            'confidence': relation.confidence,
            'attention_weight': relation.attention_weight,
            'metadata': relation.metadata,
        }
        
        # Handle relation embedding if present
        if relation.embedding is not None:
            if isinstance(relation.embedding, np.ndarray):
                tensor_data = await self._serialize_tensor(relation.embedding)
                relation_data['embedding'] = tensor_data
            elif isinstance(relation.embedding, torch.Tensor):
                # Convert torch tensor to numpy for serialization
                numpy_data = relation.embedding.detach().cpu().numpy()
                tensor_data = await self._serialize_tensor(numpy_data)
                relation_data['embedding'] = tensor_data
                
        # Handle temporal metadata
        if hasattr(relation, 'temporal_metadata'):
            relation_data['temporal_metadata'] = await self._serialize_temporal_metadata(
                relation.temporal_metadata
            )
            
        # Handle predictive fields
        if hasattr(relation, 'anticipated_weights'):
            relation_data['anticipated_weights'] = relation.anticipated_weights
            
        if hasattr(relation, 'weight_prediction_errors'):
            relation_data['weight_prediction_errors'] = dict(relation.weight_prediction_errors)
            
        if hasattr(relation, 'weight_precision'):
            relation_data['weight_precision'] = relation.weight_precision
            
        # Use the selected format for serialization
        if format in self.formats:
            return await self.formats[format](relation_data)
        else:
            self.logger.warning(f"Unknown serialization format: {format}, using default")
            return await self.formats[self.default_format](relation_data)
    
    async def deserialize_relation(self, data: Union[Dict[str, Any], bytes, str], 
                                  format: Optional[str] = None) -> SemanticRelation:
        """
        Deserialize data into a semantic relation.
        
        Args:
            data: Serialized relation data
            format: Serialization format ('json', 'pickle', 'binary')
            
        Returns:
            Deserialized SemanticRelation
        """
        format = format or self.default_format
        
        # Deserialize from the selected format
        if format in self.deserialize_formats:
            relation_data = await self.deserialize_formats[format](data)
        else:
            self.logger.warning(f"Unknown deserialization format: {format}, using default")
            relation_data = await self.deserialize_formats[self.default_format](data)
            
        # Handle embedding if present
        embedding = None
        if 'embedding' in relation_data:
            embedding = await self._deserialize_tensor(relation_data['embedding'])
            del relation_data['embedding']  # Remove from dict
            
        # Handle temporal metadata if present
        temporal_metadata = None
        if 'temporal_metadata' in relation_data:
            temporal_metadata = await self._deserialize_temporal_metadata(
                relation_data['temporal_metadata']
            )
            del relation_data['temporal_metadata']  # Remove from dict
            
        # Handle prediction data
        prediction_fields = [
            'anticipated_weights', 'weight_prediction_errors', 'weight_precision'
        ]
        prediction_data = {}
        for field in prediction_fields:
            if field in relation_data:
                prediction_data[field] = relation_data[field]
                del relation_data[field]  # Remove from main dict
                
        # Create relation with remaining data
        relation = SemanticRelation(**relation_data)
        
        # Add embedding
        if embedding is not None:
            relation.embedding = embedding
        
        # Add temporal metadata
        if temporal_metadata is not None:
            relation.temporal_metadata = temporal_metadata
            
        # Add prediction fields
        for field, value in prediction_data.items():
            setattr(relation, field, value)
            
        # Convert weight_prediction_errors from dict to defaultdict if needed
        if 'weight_prediction_errors' in prediction_data:
            relation.weight_prediction_errors = defaultdict(list, relation.weight_prediction_errors)
            
        return relation
    
    async def _serialize_tensor(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Serialize tensor data efficiently."""
        # Get tensor data and metadata
        tensor_bytes = tensor.tobytes()
        tensor_shape = tensor.shape
        tensor_dtype = str(tensor.dtype)
        
        # Determine if compression is beneficial
        should_compress = (
            self.compress_tensors and 
            len(tensor_bytes) > self.compress_threshold
        )
        
        if should_compress:
            # Compress tensor data
            compressed_bytes = zlib.compress(tensor_bytes)
            
            return {
                'data': compressed_bytes,
                'shape': tensor_shape,
                'dtype': tensor_dtype,
                'compressed': True
            }
        else:
            return {
                'data': tensor_bytes,
                'shape': tensor_shape,
                'dtype': tensor_dtype,
                'compressed': False
            }
    
    async def _deserialize_tensor(self, tensor_data: Dict[str, Any]) -> np.ndarray:
        """Deserialize tensor data."""
        # Extract tensor information
        data = tensor_data['data']
        shape = tensor_data['shape']
        dtype = tensor_data['dtype']
        compressed = tensor_data.get('compressed', False)
        
        # Decompress if needed
        if compressed:
            data = zlib.decompress(data)
            
        # Reconstruct numpy array
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    
    async def _serialize_temporal_metadata(self, 
                                          temporal_metadata: AdaptiveTemporalMetadata) -> Dict[str, Any]:
        """Serialize temporal metadata."""
        # Extract basic fields
        metadata_dict = {
            'creation_time': temporal_metadata.creation_time,
            'last_accessed': temporal_metadata.last_accessed,
            'last_modified': temporal_metadata.last_modified,
            'access_count': temporal_metadata.access_count,
            'modification_count': temporal_metadata.modification_count,
            'contextual_half_lives': dict(temporal_metadata.contextual_half_lives),
        }
        
        # Handle predictive fields
        metadata_dict['predicted_relevance'] = dict(temporal_metadata.predicted_relevance)
        metadata_dict['relevance_errors'] = {
            k: list(v) for k, v in temporal_metadata.relevance_errors.items()
        }
        metadata_dict['relevance_precision'] = dict(temporal_metadata.relevance_precision)
        
        return metadata_dict
    
    async def _deserialize_temporal_metadata(self, 
                                            data: Dict[str, Any]) -> AdaptiveTemporalMetadata:
        """Deserialize temporal metadata."""
        # Create basic temporal metadata
        temporal_metadata = AdaptiveTemporalMetadata(
            creation_time=data['creation_time'],
            last_accessed=data['last_accessed'],
            last_modified=data['last_modified'],
            access_count=data['access_count'],
            modification_count=data['modification_count'],
            contextual_half_lives=data['contextual_half_lives']
        )
        
        # Add predictive fields
        temporal_metadata.predicted_relevance = data['predicted_relevance']
        temporal_metadata.relevance_errors = defaultdict(list)
        for k, v in data['relevance_errors'].items():
            temporal_metadata.relevance_errors[k] = v
        temporal_metadata.relevance_precision = data['relevance_precision']
        
        return temporal_metadata
    
    async def _serialize_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize to JSON-compatible format."""
        # Convert non-JSON serializable types
        serializable_data = {}
        
        for key, value in data.items():
            if isinstance(value, (np.ndarray, bytes)):
                # Convert binary data to base64
                import base64
                if isinstance(value, np.ndarray):
                    value = value.tobytes()
                serializable_data[key] = {
                    'type': 'binary',
                    'data': base64.b64encode(value).decode('ascii')
                }
            elif isinstance(value, np.integer):
                serializable_data[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_data[key] = float(value)
            elif isinstance(value, defaultdict):
                serializable_data[key] = {
                    'type': 'defaultdict',
                    'data': dict(value)
                }
            elif isinstance(value, (set, tuple)):
                serializable_data[key] = {
                    'type': type(value).__name__,
                    'data': list(value)
                }
            elif isinstance(value, dict):
                # Recursively process dictionaries
                serializable_data[key] = await self._process_dict_for_json(value)
            elif isinstance(value, list):
                # Recursively process lists
                serializable_data[key] = await self._process_list_for_json(value)
            else:
                # Try to serialize directly
                try:
                    # Test if directly serializable
                    json.dumps(value)
                    serializable_data[key] = value
                except (TypeError, OverflowError):
                    # Fall back to string representation
                    serializable_data[key] = {
                        'type': 'str_repr',
                        'data': str(value)
                    }
                    
        return serializable_data
    
    async def _process_dict_for_json(self, d: Dict) -> Dict:
        """Process dictionary values for JSON serialization."""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = await self._process_dict_for_json(v)
            elif isinstance(v, list):
                result[k] = await self._process_list_for_json(v)
            elif isinstance(v, (np.ndarray, bytes)):
                # Convert binary data to base64
                import base64
                if isinstance(v, np.ndarray):
                    v = v.tobytes()
                result[k] = {
                    'type': 'binary',
                    'data': base64.b64encode(v).decode('ascii')
                }
            elif isinstance(v, np.integer):
                result[k] = int(v)
            elif isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, defaultdict):
                result[k] = {
                    'type': 'defaultdict',
                    'data': dict(v)
                }
            elif isinstance(v, (set, tuple)):
                result[k] = {
                    'type': type(v).__name__,
                    'data': list(v)
                }
            else:
                # Try direct serialization
                try:
                    # Test if directly serializable
                    json.dumps(v)
                    result[k] = v
                except (TypeError, OverflowError):
                    # Fall back to string representation
                    result[k] = {
                        'type': 'str_repr',
                        'data': str(v)
                    }
        return result
    
    async def _process_list_for_json(self, lst: List) -> List:
        """Process list values for JSON serialization."""
        result = []
        for v in lst:
            if isinstance(v, dict):
                result.append(await self._process_dict_for_json(v))
            elif isinstance(v, list):
                result.append(await self._process_list_for_json(v))
            elif isinstance(v, (np.ndarray, bytes)):
                # Convert binary data to base64
                import base64
                if isinstance(v, np.ndarray):
                    v = v.tobytes()
                result.append({
                    'type': 'binary',
                    'data': base64.b64encode(v).decode('ascii')
                })
            elif isinstance(v, np.integer):
                result.append(int(v))
            elif isinstance(v, np.floating):
                result.append(float(v))
            elif isinstance(v, defaultdict):
                result.append({
                    'type': 'defaultdict',
                    'data': dict(v)
                })
            elif isinstance(v, (set, tuple)):
                result.append({
                    'type': type(v).__name__,
                    'data': list(v)
                })
            else:
                # Try direct serialization
                try:
                    # Test if directly serializable
                    json.dumps(v)
                    result.append(v)
                except (TypeError, OverflowError):
                    # Fall back to string representation
                    result.append({
                        'type': 'str_repr',
                        'data': str(v)
                    })
        return result
    
    async def _deserialize_json(self, data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Deserialize from JSON format."""
        # Parse JSON if string
        if isinstance(data, str):
            parsed_data = json.loads(data)
        else:
            parsed_data = data
            
        # Reconstruct special types
        result = {}
        
        for key, value in parsed_data.items():
            if isinstance(value, dict) and 'type' in value and 'data' in value:
                # Special type encoding
                type_name = value['type']
                if type_name == 'binary':
                    # Convert base64 to bytes
                    import base64
                    result[key] = base64.b64decode(value['data'])
                elif type_name == 'defaultdict':
                    # Reconstruct defaultdict
                    result[key] = defaultdict(list, value['data'])
                elif type_name == 'set':
                    # Reconstruct set
                    result[key] = set(value['data'])
                elif type_name == 'tuple':
                    # Reconstruct tuple
                    result[key] = tuple(value['data'])
                elif type_name == 'str_repr':
                    # Just keep string representation
                    result[key] = value['data']
                else:
                    # Unknown type, keep as is
                    result[key] = value
            elif isinstance(value, dict):
                # Recursively process dictionaries
                result[key] = await self._process_dict_from_json(value)
            elif isinstance(value, list):
                # Recursively process lists
                result[key] = await self._process_list_from_json(value)
            else:
                # No special handling needed
                result[key] = value
                
        return result
    
    async def _process_dict_from_json(self, d: Dict) -> Dict:
        """Process dictionary from JSON deserialization."""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if 'type' in v and 'data' in v:
                    # Special type encoding
                    type_name = v['type']
                    if type_name == 'binary':
                        # Convert base64 to bytes
                        import base64
                        result[k] = base64.b64decode(v['data'])
                    elif type_name == 'defaultdict':
                        # Reconstruct defaultdict
                        result[k] = defaultdict(list, v['data'])
                    elif type_name == 'set':
                        # Reconstruct set
                        result[k] = set(v['data'])
                    elif type_name == 'tuple':
                        # Reconstruct tuple
                        result[k] = tuple(v['data'])
                    elif type_name == 'str_repr':
                        # Just keep string representation
                        result[k] = v['data']
                    else:
                        # Unknown type, keep as is
                        result[k] = v
                else:
                    # Regular dictionary
                    result[k] = await self._process_dict_from_json(v)
            elif isinstance(v, list):
                # Process list
                result[k] = await self._process_list_from_json(v)
            else:
                # Regular value
                result[k] = v
        return result
    
    async def _process_list_from_json(self, lst: List) -> List:
        """Process list from JSON deserialization."""
        result = []
        for v in lst:
            if isinstance(v, dict):
                if 'type' in v and 'data' in v:
                    # Special type encoding
                    type_name = v['type']
                    if type_name == 'binary':
                        # Convert base64 to bytes
                        import base64
                        result.append(base64.b64decode(v['data']))
                    elif type_name == 'defaultdict':
                        # Reconstruct defaultdict
                        result.append(defaultdict(list, v['data']))
                    elif type_name == 'set':
                        # Reconstruct set
                        result.append(set(v['data']))
                    elif type_name == 'tuple':
                        # Reconstruct tuple
                        result.append(tuple(v['data']))
                    elif type_name == 'str_repr':
                        # Just keep string representation
                        result.append(v['data'])
                    else:
                        # Unknown type, keep as is
                        result.append(v)
                else:
                    # Regular dictionary
                    result.append(await self._process_dict_from_json(v))
            elif isinstance(v, list):
                # Process list
                result.append(await self._process_list_from_json(v))
            else:
                # Regular value
                result.append(v)
        return result
    
    async def _serialize_pickle(self, data: Dict[str, Any]) -> bytes:
        """Serialize using pickle."""
        return pickle.dumps(data)
    
    async def _deserialize_pickle(self, data: bytes) -> Dict[str, Any]:
        """Deserialize using pickle."""
        return pickle.loads(data)
    
    async def _serialize_binary(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize to custom binary format.
        In a real implementation, this would be a more efficient binary encoding.
        """
        # For simplicity, we'll use pickle with compression
        pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(pickled_data)
    
    async def _deserialize_binary(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from custom binary format."""
        # Decompress and unpickle
        decompressed_data = zlib.decompress(data)
        return pickle.loads(decompressed_data)


class StorageBackend:
    """
    Abstract base class for storage backends.
    Defines interface that all storage backends must implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize storage backend.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3.Persistence.Storage")
    
    async def initialize(self) -> bool:
        """
        Initialize the storage backend.
        
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Storage backends must implement initialize()")
    
    async def store_node(self, node_id: str, node_data: Any) -> bool:
        """
        Store node data.
        
        Args:
            node_id: ID of the node
            node_data: Serialized node data
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Storage backends must implement store_node()")
    
    async def retrieve_node(self, node_id: str) -> Optional[Any]:
        """
        Retrieve node data.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[Any]: Node data if found, None otherwise
        """
        raise NotImplementedError("Storage backends must implement retrieve_node()")
    
    async def store_relation(self, relation_id: str, relation_data: Any) -> bool:
        """
        Store relation data.
        
        Args:
            relation_id: ID of the relation
            relation_data: Serialized relation data
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Storage backends must implement store_relation()")
    
    async def retrieve_relation(self, relation_id: str) -> Optional[Any]:
        """
        Retrieve relation data.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            Optional[Any]: Relation data if found, None otherwise
        """
        raise NotImplementedError("Storage backends must implement retrieve_relation()")
    
    async def store_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata about the semantic network.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Storage backends must implement store_metadata()")
    
    async def retrieve_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata about the semantic network.
        
        Returns:
            Optional[Dict[str, Any]]: Metadata if found, None otherwise
        """
        raise NotImplementedError("Storage backends must implement retrieve_metadata()")
    
    async def list_nodes(self) -> List[str]:
        """
        List all stored node IDs.
        
        Returns:
            List[str]: List of node IDs
        """
        raise NotImplementedError("Storage backends must implement list_nodes()")
    
    async def list_relations(self) -> List[str]:
        """
        List all stored relation IDs.
        
        Returns:
            List[str]: List of relation IDs
        """
        raise NotImplementedError("Storage backends must implement list_relations()")
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete node data.
        
        Args:
            node_id: ID of the node
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Storage backends must implement delete_node()")
    
    async def delete_relation(self, relation_id: str) -> bool:
        """
        Delete relation data.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Storage backends must implement delete_relation()")
    
    async def get_last_modified(self, entity_id: str) -> Optional[float]:
        """
        Get last modification time for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Optional[float]: Timestamp if found, None otherwise
        """
        raise NotImplementedError("Storage backends must implement get_last_modified()")
    
    async def create_backup(self) -> str:
        """
        Create a backup of the current storage.
        
        Returns:
            str: Backup identifier or path
        """
        raise NotImplementedError("Storage backends must implement create_backup()")
    
    async def close(self) -> None:
        """Close the storage backend and release resources."""
        raise NotImplementedError("Storage backends must implement close()")


class FileSystemBackend(StorageBackend):
    """
    File system-based storage backend.
    Stores nodes and relations as individual files in a directory structure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize file system backend.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.base_path = Path(self.config.get('storage_path', './semantic_data'))
        self.nodes_path = self.base_path / 'nodes'
        self.relations_path = self.base_path / 'relations'
        self.format = self.config.get('file_format', 'json')
        self.compression = self.config.get('compression', True)
        
        # Format to file extension mapping
        self.format_extensions = {
            'json': '.json',
            'pickle': '.pkl',
            'binary': '.bin'
        }
        
        # Cache for last modified times
        self.last_modified_cache = {}
        self.cache_expiry = 60  # Seconds
        self.last_cache_update = 0
    
    async def initialize(self) -> bool:
        """
        Initialize the file system backend.
        Creates necessary directories.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create base directories
            self.base_path.mkdir(exist_ok=True, parents=True)
            self.nodes_path.mkdir(exist_ok=True)
            self.relations_path.mkdir(exist_ok=True)
            
            # Create metadata file if it doesn't exist
            metadata_path = self.base_path / 'metadata.json'
            if not metadata_path.exists():
                async with aiofiles.open(metadata_path, 'w') as f:
                    await f.write(json.dumps({
                        'version': '1.0',
                        'created_at': time.time(),
                        'last_updated': time.time()
                    }))
                    
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize file system backend: {e}")
            return False
    
    async def store_node(self, node_id: str, node_data: Any) -> bool:
        """
        Store node data as a file.
        
        Args:
            node_id: ID of the node
            node_data: Serialized node data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create safe filename
            safe_id = self._safe_filename(node_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            # Ensure node directory exists
            node_dir = self.nodes_path / safe_id[0:2]
            node_dir.mkdir(exist_ok=True)
            
            # Determine file path and open file
            if self.compression:
                file_path = node_dir / f"{safe_id}{file_ext}.gz"
                async with aiofiles.open(file_path, 'wb') as f:
                    # Convert to bytes if necessary
                    if isinstance(node_data, (dict, list)):
                        data_bytes = json.dumps(node_data).encode('utf-8')
                    elif isinstance(node_data, str):
                        data_bytes = node_data.encode('utf-8')
                    else:
                        data_bytes = node_data
                        
                    # Compress data
                    compressed_data = gzip.compress(data_bytes)
                    await f.write(compressed_data)
            else:
                file_path = node_dir / f"{safe_id}{file_ext}"
                async with aiofiles.open(file_path, 'w' if self.format == 'json' else 'wb') as f:
                    if self.format == 'json':
                        # JSON format (text)
                        if isinstance(node_data, (dict, list)):
                            await f.write(json.dumps(node_data))
                        elif isinstance(node_data, str):
                            await f.write(node_data)
                        else:
                            await f.write(str(node_data))
                    else:
                        # Binary format
                        await f.write(node_data)
                        
            # Update last modified cache
            self.last_modified_cache[node_id] = time.time()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store node {node_id}: {e}")
            return False
    
    async def retrieve_node(self, node_id: str) -> Optional[Any]:
        """
        Retrieve node data from file.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[Any]: Node data if found, None otherwise
        """
        try:
            # Create safe filename
            safe_id = self._safe_filename(node_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            # Get node directory
            node_dir = self.nodes_path / safe_id[0:2]
            
            # Check for compressed or uncompressed file
            compressed_path = node_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = node_dir / f"{safe_id}{file_ext}"
            
            if compressed_path.exists():
                # Read compressed file
                async with aiofiles.open(compressed_path, 'rb') as f:
                    compressed_data = await f.read()
                    data = gzip.decompress(compressed_data)
                    
                    # Parse based on format
                    if self.format == 'json':
                        return json.loads(data.decode('utf-8'))
                    else:
                        return data
            elif uncompressed_path.exists():
                # Read uncompressed file
                if self.format == 'json':
                    async with aiofiles.open(uncompressed_path, 'r') as f:
                        data = await f.read()
                        return json.loads(data)
                else:
                    async with aiofiles.open(uncompressed_path, 'rb') as f:
                        return await f.read()
            else:
                # Node not found
                return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve node {node_id}: {e}")
            return None
    
    async def store_relation(self, relation_id: str, relation_data: Any) -> bool:
        """
        Store relation data as a file.
        
        Args:
            relation_id: ID of the relation
            relation_data: Serialized relation data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create safe filename
            safe_id = self._safe_filename(relation_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            # Ensure relation directory exists
            relation_dir = self.relations_path / safe_id[0:2]
            relation_dir.mkdir(exist_ok=True)
            
            # Determine file path and open file
            if self.compression:
                file_path = relation_dir / f"{safe_id}{file_ext}.gz"
                async with aiofiles.open(file_path, 'wb') as f:
                    # Convert to bytes if necessary
                    if isinstance(relation_data, (dict, list)):
                        data_bytes = json.dumps(relation_data).encode('utf-8')
                    elif isinstance(relation_data, str):
                        data_bytes = relation_data.encode('utf-8')
                    else:
                        data_bytes = relation_data
                        
                    # Compress data
                    compressed_data = gzip.compress(data_bytes)
                    await f.write(compressed_data)
            else:
                file_path = relation_dir / f"{safe_id}{file_ext}"
                async with aiofiles.open(file_path, 'w' if self.format == 'json' else 'wb') as f:
                    if self.format == 'json':
                        # JSON format (text)
                        if isinstance(relation_data, (dict, list)):
                            await f.write(json.dumps(relation_data))
                        elif isinstance(relation_data, str):
                            await f.write(relation_data)
                        else:
                            await f.write(str(relation_data))
                    else:
                        # Binary format
                        await f.write(relation_data)
                        
            # Update last modified cache
            self.last_modified_cache[relation_id] = time.time()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store relation {relation_id}: {e}")
            return False
    
    async def retrieve_relation(self, relation_id: str) -> Optional[Any]:
        """
        Retrieve relation data from file.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            Optional[Any]: Relation data if found, None otherwise
        """
        try:
            # Create safe filename
            safe_id = self._safe_filename(relation_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            # Get relation directory
            relation_dir = self.relations_path / safe_id[0:2]
            
            # Check for compressed or uncompressed file
            compressed_path = relation_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = relation_dir / f"{safe_id}{file_ext}"
            
            if compressed_path.exists():
                # Read compressed file
                async with aiofiles.open(compressed_path, 'rb') as f:
                    compressed_data = await f.read()
                    data = gzip.decompress(compressed_data)
                    
                    # Parse based on format
                    if self.format == 'json':
                        return json.loads(data.decode('utf-8'))
                    else:
                        return data
            elif uncompressed_path.exists():
                # Read uncompressed file
                if self.format == 'json':
                    async with aiofiles.open(uncompressed_path, 'r') as f:
                        data = await f.read()
                        return json.loads(data)
                else:
                    async with aiofiles.open(uncompressed_path, 'rb') as f:
                        return await f.read()
            else:
                # Relation not found
                return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve relation {relation_id}: {e}")
            return None
    
    async def store_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata about the semantic network.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update last_updated timestamp
            metadata['last_updated'] = time.time()
            
            # Write metadata file
            metadata_path = self.base_path / 'metadata.json'
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to store metadata: {e}")
            return False
    
    async def retrieve_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata about the semantic network.
        
        Returns:
            Optional[Dict[str, Any]]: Metadata if found, None otherwise
        """
        try:
            metadata_path = self.base_path / 'metadata.json'
            if not metadata_path.exists():
                return None
                
            async with aiofiles.open(metadata_path, 'r') as f:
                data = await f.read()
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to retrieve metadata: {e}")
            return None
    
    async def list_nodes(self) -> List[str]:
        """
        List all stored node IDs.
        
        Returns:
            List[str]: List of node IDs
        """
        nodes = []
        
        try:
            # For each subdirectory in nodes path
            for subdir in self.nodes_path.iterdir():
                if subdir.is_dir():
                    # For each file in subdirectory
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            # Extract node ID from filename
                            filename = file_path.name
                            
                            # Remove extensions
                            for ext in ['.gz'] + list(self.format_extensions.values()):
                                if filename.endswith(ext):
                                    filename = filename[:-len(ext)]
                                    
                            nodes.append(filename)
                            
            return nodes
        except Exception as e:
            self.logger.error(f"Failed to list nodes: {e}")
            return []
    
    async def list_relations(self) -> List[str]:
        """
        List all stored relation IDs.
        
        Returns:
            List[str]: List of relation IDs
        """
        relations = []
        
        try:
            # For each subdirectory in relations path
            for subdir in self.relations_path.iterdir():
                if subdir.is_dir():
                    # For each file in subdirectory
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            # Extract relation ID from filename
                            filename = file_path.name
                            
                            # Remove extensions
                            for ext in ['.gz'] + list(self.format_extensions.values()):
                                if filename.endswith(ext):
                                    filename = filename[:-len(ext)]
                                    
                            relations.append(filename)
                            
            return relations
        except Exception as e:
            self.logger.error(f"Failed to list relations: {e}")
            return []
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete node data file.
        
        Args:
            node_id: ID of the node
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create safe filename
            safe_id = self._safe_filename(node_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            # Get node directory
            node_dir = self.nodes_path / safe_id[0:2]
            
            # Check for compressed or uncompressed file
            compressed_path = node_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = node_dir / f"{safe_id}{file_ext}"
            
            deleted = False
            
            if compressed_path.exists():
                compressed_path.unlink()
                deleted = True
                
            if uncompressed_path.exists():
                uncompressed_path.unlink()
                deleted = True
                
            # Remove from cache
            if node_id in self.last_modified_cache:
                del self.last_modified_cache[node_id]
                
            return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    async def delete_relation(self, relation_id: str) -> bool:
        """
        Delete relation data file.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create safe filename
            safe_id = self._safe_filename(relation_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            # Get relation directory
            relation_dir = self.relations_path / safe_id[0:2]
            
            # Check for compressed or uncompressed file
            compressed_path = relation_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = relation_dir / f"{safe_id}{file_ext}"
            
            deleted = False
            
            if compressed_path.exists():
                compressed_path.unlink()
                deleted = True
                
            if uncompressed_path.exists():
                uncompressed_path.unlink()
                deleted = True
                
            # Remove from cache
            if relation_id in self.last_modified_cache:
                del self.last_modified_cache[relation_id]
                
            return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete relation {relation_id}: {e}")
            return False
    
    async def get_last_modified(self, entity_id: str) -> Optional[float]:
        """
        Get last modification time for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Optional[float]: Timestamp if found, None otherwise
        """
        # Check cache first
        if entity_id in self.last_modified_cache:
            return self.last_modified_cache[entity_id]
            
        try:
            # Determine if it's a node or relation
            safe_id = self._safe_filename(entity_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            # Check node files
            node_dir = self.nodes_path / safe_id[0:2]
            node_compressed = node_dir / f"{safe_id}{file_ext}.gz"
            node_uncompressed = node_dir / f"{safe_id}{file_ext}"
            
            # Check relation files
            relation_dir = self.relations_path / safe_id[0:2]
            relation_compressed = relation_dir / f"{safe_id}{file_ext}.gz"
            relation_uncompressed = relation_dir / f"{safe_id}{file_ext}"
            
            # Check each path
            for path in [node_compressed, node_uncompressed, relation_compressed, relation_uncompressed]:
                if path.exists():
                    mtime = path.stat().st_mtime
                    # Cache the result
                    self.last_modified_cache[entity_id] = mtime
                    return mtime
                    
            return None
        except Exception as e:
            self.logger.error(f"Failed to get last modified time for {entity_id}: {e}")
            return None
    
    async def create_backup(self) -> str:
        """
        Create a backup of the current storage.
        
        Returns:
            str: Backup path
        """
        try:
            # Create backup directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.base_path.parent / f"semantic_backup_{timestamp}"
            backup_dir.mkdir(parents=True)
            
            # Copy nodes directory
            nodes_backup_dir = backup_dir / 'nodes'
            nodes_backup_dir.mkdir()
            
            # Copy relations directory
            relations_backup_dir = backup_dir / 'relations'
            relations_backup_dir.mkdir()
            
            # Copy nodes
            for subdir in self.nodes_path.iterdir():
                if subdir.is_dir():
                    # Create corresponding backup subdirectory
                    backup_subdir = nodes_backup_dir / subdir.name
                    backup_subdir.mkdir()
                    
                    # Copy files
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            import shutil
                            shutil.copy2(file_path, backup_subdir / file_path.name)
            
            # Copy relations
            for subdir in self.relations_path.iterdir():
                if subdir.is_dir():
                    # Create corresponding backup subdirectory
                    backup_subdir = relations_backup_dir / subdir.name
                    backup_subdir.mkdir()
                    
                    # Copy files
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            import shutil
                            shutil.copy2(file_path, backup_subdir / file_path.name)
            
            # Copy metadata
            metadata_path = self.base_path / 'metadata.json'
            if metadata_path.exists():
                import shutil
                shutil.copy2(metadata_path, backup_dir / 'metadata.json')
                
            return str(backup_dir)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return ""
    
    async def close(self) -> None:
        """Close the storage backend (no-op for file system)."""
        pass
    
    def _safe_filename(self, entity_id: str) -> str:
        """Convert entity ID to safe filename."""
        # Remove unsafe characters
        safe_id = ''.join(c if c.isalnum() or c in '-_' else '_' for c in entity_id)
        
        # Ensure not too long
        if len(safe_id) > 255:
            # Use a hash in the filename to avoid collisions
            hash_part = hashlib.md5(entity_id.encode('utf-8')).hexdigest()[:16]
            safe_id = f"{safe_id[:230]}_{hash_part}"
            
        return safe_id


class SqliteBackend(StorageBackend):
    """
    SQLite-based storage backend.
    Stores nodes and relations in a SQLite database.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SQLite backend.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.db_path = Path(self.config.get('db_path', './semantic_data/network.db'))
        self.format = self.config.get('serialization_format', 'binary')
        self.conn = None
        self.last_modified_cache = {}
        self.cache_expiry = 60  # Seconds
        self.last_cache_update = 0
    
    async def initialize(self) -> bool:
        """
        Initialize the SQLite backend.
        Creates database tables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Connect to database
            self.conn = await aiosqlite.connect(self.db_path)
            
            # Enable WAL mode for better concurrency
            await self.conn.execute("PRAGMA journal_mode=WAL")
            
            # Create tables
            await self.conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    format TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    modified_at REAL NOT NULL
                )
            ''')
            
            await self.conn.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    data BLOB NOT NULL,
                    format TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    modified_at REAL NOT NULL
                )
            ''')
            
            await self.conn.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    modified_at REAL NOT NULL
                )
            ''')
            
            # Create indices
            await self.conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)')
            await self.conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)')
            
            # Commit changes
            await self.conn.commit()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite backend: {e}")
            return False
    
    async def store_node(self, node_id: str, node_data: Any) -> bool:
        """
        Store node data in SQLite.
        
        Args:
            node_id: ID of the node
            node_data: Serialized node data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Convert data to appropriate format
            if self.format == 'json':
                if isinstance(node_data, (dict, list)):
                    data_bytes = json.dumps(node_data).encode('utf-8')
                elif isinstance(node_data, str):
                    data_bytes = node_data.encode('utf-8')
                else:
                    data_bytes = node_data
            else:
                # Binary data
                if isinstance(node_data, (dict, list)):
                    data_bytes = pickle.dumps(node_data)
                elif isinstance(node_data, str):
                    data_bytes = node_data.encode('utf-8')
                else:
                    data_bytes = node_data
                    
            # Check if node exists
            now = time.time()
            async with self.conn.execute("SELECT id FROM nodes WHERE id=?", (node_id,)) as cursor:
                if await cursor.fetchone():
                    # Update existing node
                    await self.conn.execute(
                        "UPDATE nodes SET data=?, format=?, modified_at=? WHERE id=?",
                        (data_bytes, self.format, now, node_id)
                    )
                else:
                    # Insert new node
                    await self.conn.execute(
                        "INSERT INTO nodes (id, data, format, created_at, modified_at) VALUES (?, ?, ?, ?, ?)",
                        (node_id, data_bytes, self.format, now, now)
                    )
                    
            # Commit changes
            await self.conn.commit()
            
            # Update cache
            self.last_modified_cache[node_id] = now
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store node {node_id} in SQLite: {e}")
            return False
    
    async def retrieve_node(self, node_id: str) -> Optional[Any]:
        """
        Retrieve node data from SQLite.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[Any]: Node data if found, None otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Query database
            async with self.conn.execute(
                "SELECT data, format FROM nodes WHERE id=?", (node_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row is None:
                    return None
                    
                data_bytes, format_type = row
                
                # Parse based on format
                if format_type == 'json':
                    return json.loads(data_bytes.decode('utf-8'))
                else:
                    # Binary format - usually pickle
                    return data_bytes
        except Exception as e:
            self.logger.error(f"Failed to retrieve node {node_id} from SQLite: {e}")
            return None
    
    async def store_relation(self, relation_id: str, relation_data: Any) -> bool:
        """
        Store relation data in SQLite.
        
        Args:
            relation_id: ID of the relation
            relation_data: Serialized relation data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Convert data to appropriate format
            if self.format == 'json':
                if isinstance(relation_data, (dict, list)):
                    data_bytes = json.dumps(relation_data).encode('utf-8')
                elif isinstance(relation_data, str):
                    data_bytes = relation_data.encode('utf-8')
                else:
                    data_bytes = relation_data
            else:
                # Binary data
                if isinstance(relation_data, (dict, list)):
                    data_bytes = pickle.dumps(relation_data)
                elif isinstance(relation_data, str):
                    data_bytes = relation_data.encode('utf-8')
                else:
                    data_bytes = relation_data
                    
            # Extract source and target IDs if present
            source_id = None
            target_id = None
            
            if isinstance(relation_data, dict):
                source_id = relation_data.get('source_id')
                target_id = relation_data.get('target_id')
            
            # Check if relation exists
            now = time.time()
            async with self.conn.execute("SELECT id FROM relations WHERE id=?", (relation_id,)) as cursor:
                if await cursor.fetchone():
                    # Update existing relation
                    await self.conn.execute(
                        "UPDATE relations SET data=?, format=?, modified_at=?, source_id=?, target_id=? WHERE id=?",
                        (data_bytes, self.format, now, source_id, target_id, relation_id)
                    )
                else:
                    # Insert new relation
                    await self.conn.execute(
                        "INSERT INTO relations (id, source_id, target_id, data, format, created_at, modified_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (relation_id, source_id, target_id, data_bytes, self.format, now, now)
                    )
                    
            # Commit changes
            await self.conn.commit()
            
            # Update cache
            self.last_modified_cache[relation_id] = now
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store relation {relation_id} in SQLite: {e}")
            return False
    
    async def retrieve_relation(self, relation_id: str) -> Optional[Any]:
        """
        Retrieve relation data from SQLite.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            Optional[Any]: Relation data if found, None otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Query database
            async with self.conn.execute(
                "SELECT data, format FROM relations WHERE id=?", (relation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row is None:
                    return None
                    
                data_bytes, format_type = row
                
                # Parse based on format
                if format_type == 'json':
                    return json.loads(data_bytes.decode('utf-8'))
                else:
                    # Binary format - usually pickle
                    return data_bytes
        except Exception as e:
            self.logger.error(f"Failed to retrieve relation {relation_id} from SQLite: {e}")
            return None
    
    async def store_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata in SQLite.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Convert metadata to JSON
            now = time.time()
            value_bytes = json.dumps(metadata).encode('utf-8')
            
            # Store in metadata table with 'network' key
            await self.conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value, modified_at) VALUES (?, ?, ?)",
                ('network', value_bytes, now)
            )
            
            # Commit changes
            await self.conn.commit()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store metadata in SQLite: {e}")
            return False
    
    async def retrieve_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata from SQLite.
        
        Returns:
            Optional[Dict[str, Any]]: Metadata if found, None otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Query database
            async with self.conn.execute(
                "SELECT value FROM metadata WHERE key=?", ('network',)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row is None:
                    return None
                    
                value_bytes = row[0]
                
                # Parse JSON
                return json.loads(value_bytes.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"Failed to retrieve metadata from SQLite: {e}")
            return None
    
    async def list_nodes(self) -> List[str]:
        """
        List all stored node IDs.
        
        Returns:
            List[str]: List of node IDs
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Query database
            async with self.conn.execute("SELECT id FROM nodes") as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to list nodes from SQLite: {e}")
            return []
    
    async def list_relations(self) -> List[str]:
        """
        List all stored relation IDs.
        
        Returns:
            List[str]: List of relation IDs
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Query database
            async with self.conn.execute("SELECT id FROM relations") as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to list relations from SQLite: {e}")
            return []
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete node from SQLite.
        
        Args:
            node_id: ID of the node
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Delete from database
            await self.conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
            
            # Commit changes
            await self.conn.commit()
            
            # Remove from cache
            if node_id in self.last_modified_cache:
                del self.last_modified_cache[node_id]
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete node {node_id} from SQLite: {e}")
            return False
    
    async def delete_relation(self, relation_id: str) -> bool:
        """
        Delete relation from SQLite.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Delete from database
            await self.conn.execute("DELETE FROM relations WHERE id=?", (relation_id,))
            
            # Commit changes
            await self.conn.commit()
            
            # Remove from cache
            if relation_id in self.last_modified_cache:
                del self.last_modified_cache[relation_id]
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete relation {relation_id} from SQLite: {e}")
            return False
    
    async def get_last_modified(self, entity_id: str) -> Optional[float]:
        """
        Get last modification time for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Optional[float]: Timestamp if found, None otherwise
        """
        # Check cache first
        if entity_id in self.last_modified_cache:
            return self.last_modified_cache[entity_id]
            
        if self.conn is None:
            await self.initialize()
            
        try:
            # Query database for node
            async with self.conn.execute(
                "SELECT modified_at FROM nodes WHERE id=?", (entity_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row is not None:
                    # Cache and return result
                    self.last_modified_cache[entity_id] = row[0]
                    return row[0]
                    
            # Query database for relation
            async with self.conn.execute(
                "SELECT modified_at FROM relations WHERE id=?", (entity_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row is not None:
                    # Cache and return result
                    self.last_modified_cache[entity_id] = row[0]
                    return row[0]
                    
            return None
        except Exception as e:
            self.logger.error(f"Failed to get last modified time for {entity_id} from SQLite: {e}")
            return None
    
    async def create_backup(self) -> str:
        """
        Create a backup of the SQLite database.
        
        Returns:
            str: Backup file path
        """
        if self.conn is None:
            await self.initialize()
            
        try:
            # Create backup file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.db_path.with_suffix(f'.{timestamp}.bak')
            
            # Create backup using SQLite's backup API
            await self.conn.execute(f"VACUUM INTO '{backup_path}'")
            
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return ""
    
    async def close(self) -> None:
        """Close the SQLite connection."""
        if self.conn is not None:
            await self.conn.close()
            self.conn = None


class SemanticPersistenceManager:
    """
    Manages persistence of semantic network data across sessions.
    Supports multiple storage backends and incremental updates.
    """
    
    def __init__(self, semantic_network, config=None):
        """
        Initialize the persistence manager.
        
        Args:
            semantic_network: SemanticTensorNetwork instance
            config: Configuration dictionary
        """
        self.semantic_network = semantic_network
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3.Persistence")
        
        # Initialize serialization manager
        self.serialization_manager = SerializationManager(self.config.get('serialization', {}))
        
        # Initialize storage backend
        self.storage_type = self.config.get('storage_type', 'file')
        self.storage_backend = self._create_storage_backend()
        
        # Track sync state
        self.last_sync_time = 0
        self.nodes_sync_state = {}  # node_id -> last_sync_time
        self.relations_sync_state = {}  # relation_id -> last_sync_time
        
        # Background synchronization task
        self.sync_task = None
        self.sync_interval = self.config.get('sync_interval', 300)  # 5 minutes
        self.sync_running = False
        
        # Error handling
        self.max_retry_attempts = self.config.get('max_retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # seconds
        
        # Change tracking - track which entities have changed since last save
        self.changed_nodes = set()
        self.changed_relations = set()
        self.sync_lock = asyncio.Lock()
    
    def _create_storage_backend(self) -> StorageBackend:
        """Create and return appropriate storage backend based on configuration."""
        if self.storage_type == 'file':
            return FileSystemBackend(self.config.get('file_storage', {}))
        elif self.storage_type == 'sqlite':
            return SqliteBackend(self.config.get('sqlite_storage', {}))
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")
    
    async def initialize(self) -> bool:
        """
        Initialize the persistence manager.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize storage backend
            success = await self.storage_backend.initialize()
            if not success:
                self.logger.error("Failed to initialize storage backend")
                return False
                
            # Get or create metadata
            metadata = await self.storage_backend.retrieve_metadata()
            if metadata is None:
                # Create initial metadata
                metadata = {
                    'version': '1.0',
                    'created_at': time.time(),
                    'last_updated': time.time(),
                    'node_count': 0,
                    'relation_count': 0
                }
                await self.storage_backend.store_metadata(metadata)
                
            self.logger.info(f"Persistence manager initialized with {metadata.get('node_count', 0)} nodes and {metadata.get('relation_count', 0)} relations")
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing persistence manager: {e}")
            return False
    
    async def save_state(self, incremental=True):
        """
        Save current semantic network state.
        
        Args:
            incremental: Whether to do incremental update or full save
            
        Returns:
            Dict with save statistics
        """
        async with self.sync_lock:
            start_time = time.time()
            
            try:
                if not incremental:
                    # Full save - save all nodes and relations
                    return await self._perform_full_save()
                else:
                    # Incremental save - save only changed entities
                    return await self._perform_incremental_save()
            except Exception as e:
                self.logger.error(f"Error saving state: {e}")
                return {
                    'status': 'error',
                    'message': str(e),
                    'duration': time.time() - start_time
                }
    
    async def _perform_full_save(self):
        """Perform full save of all nodes and relations."""
        saved_nodes = 0
        saved_relations = 0
        errors = 0
        start_time = time.time()
        
        # Save all nodes
        for node_id, node in self.semantic_network.nodes.items():
            try:
                # Serialize node
                node_data = await self.serialization_manager.serialize_node(node)
                
                # Store serialized data
                success = await self.storage_backend.store_node(node_id, node_data)
                
                if success:
                    saved_nodes += 1
                    self.nodes_sync_state[node_id] = time.time()
                else:
                    errors += 1
            except Exception as e:
                self.logger.error(f"Error saving node {node_id}: {e}")
                errors += 1
                
        # Save all relations
        for relation_id, relation in self.semantic_network.relations.items():
            try:
                # Serialize relation
                relation_data = await self.serialization_manager.serialize_relation(relation)
                
                # Store serialized data
                success = await self.storage_backend.store_relation(relation_id, relation_data)
                
                if success:
                    saved_relations += 1
                    self.relations_sync_state[relation_id] = time.time()
                else:
                    errors += 1
            except Exception as e:
                self.logger.error(f"Error saving relation {relation_id}: {e}")
                errors += 1
                
        # Update metadata
        duration = time.time() - start_time
        metadata = await self.storage_backend.retrieve_metadata() or {}
        metadata.update({
            'last_updated': time.time(),
            'node_count': len(self.semantic_network.nodes),
            'relation_count': len(self.semantic_network.relations),
            'last_full_save': time.time(),
            'last_save_duration': duration
        })
        await self.storage_backend.store_metadata(metadata)
        
        # Update sync time and clear change tracking
        self.last_sync_time = time.time()
        self.changed_nodes.clear()
        self.changed_relations.clear()
        
        return {
            'status': 'success',
            'saved_nodes': saved_nodes,
            'saved_relations': saved_relations,
            'errors': errors,
            'duration': duration,
            'full_save': True
        }
    
    async def _perform_incremental_save(self):
        """Perform incremental save of changed nodes and relations."""
        # Get changes since last sync
        changed_nodes = set(self.changed_nodes)
        changed_relations = set(self.changed_relations)
        
        # Add nodes whose activation changed significantly
        for node_id, node in self.semantic_network.nodes.items():
            # Skip already changed nodes
            if node_id in changed_nodes:
                continue
                
            # Check if node was modified since last sync
            if node_id in self.nodes_sync_state:
                last_sync = self.nodes_sync_state[node_id]
                if hasattr(node, 'temporal_metadata') and node.temporal_metadata.last_modified > last_sync:
                    changed_nodes.add(node_id)
                    
        # Add relations whose weights changed significantly
        for relation_id, relation in self.semantic_network.relations.items():
            # Skip already changed relations
            if relation_id in changed_relations:
                continue
                
            # Check if relation was modified since last sync
            if relation_id in self.relations_sync_state:
                last_sync = self.relations_sync_state[relation_id]
                if hasattr(relation, 'temporal_metadata') and relation.temporal_metadata.last_modified > last_sync:
                    changed_relations.add(relation_id)
                    
        # If no changes, return early
        if not changed_nodes and not changed_relations:
            return {
                'status': 'success',
                'saved_nodes': 0,
                'saved_relations': 0,
                'errors': 0,
                'duration': 0,
                'full_save': False
            }
            
        # Perform incremental save
        saved_nodes = 0
        saved_relations = 0
        errors = 0
        start_time = time.time()
        
        # Save changed nodes
        for node_id in changed_nodes:
            node = self.semantic_network.nodes.get(node_id)
            if node is None:
                continue
                
            try:
                # Serialize node
                node_data = await self.serialization_manager.serialize_node(node)
                
                # Store serialized data
                success = await self.storage_backend.store_node(node_id, node_data)
                
                if success:
                    saved_nodes += 1
                    self.nodes_sync_state[node_id] = time.time()
                else:
                    errors += 1
            except Exception as e:
                self.logger.error(f"Error saving node {node_id}: {e}")
                errors += 1
                
        # Save changed relations
        for relation_id in changed_relations:
            relation = self.semantic_network.relations.get(relation_id)
            if relation is None:
                continue
                
            try:
                # Serialize relation
                relation_data = await self.serialization_manager.serialize_relation(relation)
                
                # Store serialized data
                success = await self.storage_backend.store_relation(relation_id, relation_data)
                
                if success:
                    saved_relations += 1
                    self.relations_sync_state[relation_id] = time.time()
                else:
                    errors += 1
            except Exception as e:
                self.logger.error(f"Error saving relation {relation_id}: {e}")
                errors += 1
                
        # Update metadata
        duration = time.time() - start_time
        metadata = await self.storage_backend.retrieve_metadata() or {}
        metadata.update({
            'last_updated': time.time(),
            'node_count': len(self.semantic_network.nodes),
            'relation_count': len(self.semantic_network.relations),
            'last_incremental_save': time.time(),
            'last_save_duration': duration
        })
        await self.storage_backend.store_metadata(metadata)
        
        # Update sync time
        self.last_sync_time = time.time()
        
        # Clear saved entities from change tracking
        self.changed_nodes -= set(node_id for node_id in changed_nodes if node_id in self.nodes_sync_state)
        self.changed_relations -= set(relation_id for relation_id in changed_relations if relation_id in self.relations_sync_state)
        
        return {
            'status': 'success',
            'saved_nodes': saved_nodes,
            'saved_relations': saved_relations,
            'errors': errors,
            'duration': duration,
            'full_save': False
        }
    
    async def load_state(self):
        """
        Load semantic network state from storage.
        
        Returns:
            Dict with load statistics
        """
        start_time = time.time()
        
        try:
            # Check if storage has valid data
            metadata = await self.storage_backend.retrieve_metadata()
            if metadata is None:
                self.logger.warning("No stored state found")
                return {
                    'status': 'error',
                    'message': 'No stored state found',
                    'duration': time.time() - start_time
                }
                
            # Clear existing network state
            self.semantic_network.nodes = {}
            self.semantic_network.relations = {}
            self.semantic_network.node_index = {}
            self.semantic_network.index_to_node = {}
            self.semantic_network.node_count = 0
            self.semantic_network.relation_index = {
                'source': defaultdict(list),
                'target': defaultdict(list),
                'type': defaultdict(list)
            }
            
            # Load nodes
            loaded_nodes = 0
            node_errors = 0
            
            node_ids = await self.storage_backend.list_nodes()
            for node_id in node_ids:
                try:
                    # Load serialized data
                    node_data = await self.storage_backend.retrieve_node(node_id)
                    if node_data is None:
                        continue
                        
                    # Deserialize node
                    node = await self.serialization_manager.deserialize_node(node_data)
                    
                    # Add to network
                    if node is not None:
                        await self.semantic_network.add_node(node, update_tensors=False)
                        loaded_nodes += 1
                        
                        # Update sync state
                        self.nodes_sync_state[node_id] = time.time()
                except Exception as e:
                    self.logger.error(f"Error loading node {node_id}: {e}")
                    node_errors += 1
                    
            # Load relations
            loaded_relations = 0
            relation_errors = 0
            
            relation_ids = await self.storage_backend.list_relations()
            for relation_id in relation_ids:
                try:
                    # Load serialized data
                    relation_data = await self.storage_backend.retrieve_relation(relation_id)
                    if relation_data is None:
                        continue
                        
                    # Deserialize relation
                    relation = await self.serialization_manager.deserialize_relation(relation_data)
                    
                    # Add to network
                    if relation is not None:
                        await self.semantic_network.add_relation(relation)
                        loaded_relations += 1
                        
                        # Update sync state
                        self.relations_sync_state[relation_id] = time.time()
                except Exception as e:
                    self.logger.error(f"Error loading relation {relation_id}: {e}")
                    relation_errors += 1
                    
            # Update tensors
            self.semantic_network.rebuild_needed = True
            
            # Update sync time
            self.last_sync_time = time.time()
            
            # Clear change tracking
            self.changed_nodes.clear()
            self.changed_relations.clear()
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'loaded_nodes': loaded_nodes,
                'loaded_relations': loaded_relations,
                'node_errors': node_errors,
                'relation_errors': relation_errors,
                'duration': duration
            }
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'duration': time.time() - start_time
            }
    
    async def track_node_change(self, node_id):
        """
        Track node change for incremental save.
        
        Args:
            node_id: ID of changed node
        """
        self.changed_nodes.add(node_id)
        
    async def track_relation_change(self, relation_id):
        """
        Track relation change for incremental save.
        
        Args:
            relation_id: ID of changed relation
        """
        self.changed_relations.add(relation_id)
    
    async def schedule_regular_backups(self, interval=3600):
        """
        Schedule regular state backups.
        
        Args:
            interval: Backup interval in seconds
        """
        self.sync_interval = interval
        
        # Cancel existing task if running
        if self.sync_task is not None:
            self.sync_task.cancel()
            
        # Start new background sync task
        self.sync_running = True
        self.sync_task = asyncio.create_task(self._background_sync())
    
    async def _background_sync(self):
        """Background task for regular synchronization."""
        try:
            while self.sync_running:
                # Wait for sync interval
                await asyncio.sleep(self.sync_interval)
                
                # Perform incremental save
                try:
                    self.logger.info("Performing scheduled incremental save")
                    async with self.sync_lock:
                        result = await self._perform_incremental_save()
                    
                    # Create periodic backup every 10 syncs
                    if result.get('saved_nodes', 0) > 0 or result.get('saved_relations', 0) > 0:
                        metadata = await self.storage_backend.retrieve_metadata() or {}
                        backup_count = metadata.get('backup_count', 0) + 1
                        
                        if backup_count % 10 == 0:
                            self.logger.info("Creating periodic backup")
                            backup_path = await self.storage_backend.create_backup()
                            metadata['last_backup'] = time.time()
                            metadata['last_backup_path'] = backup_path
                            
                        metadata['backup_count'] = backup_count
                        await self.storage_backend.store_metadata(metadata)
                except Exception as e:
                    self.logger.error(f"Error in background sync: {e}")
        except asyncio.CancelledError:
            self.logger.info("Background sync task cancelled")
        except Exception as e:
            self.logger.error(f"Unexpected error in background sync: {e}")
    
    async def create_backup(self):
        """
        Create a backup of the current state.
        
        Returns:
            str: Backup identifier or path
        """
        try:
            # First ensure all changes are saved
            await self.save_state(incremental=True)
            
            # Create backup using storage backend
            backup_path = await self.storage_backend.create_backup()
            
            if backup_path:
                # Update metadata
                metadata = await self.storage_backend.retrieve_metadata() or {}
                metadata['last_backup'] = time.time()
                metadata['last_backup_path'] = backup_path
                await self.storage_backend.store_metadata(metadata)
                
                self.logger.info(f"Created backup at {backup_path}")
                
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return ""
    
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
        
        self.default_format = self.config.get('serialization_format', 'json')
        self.compress_tensors = self.config.get('compress_tensors', True)
        self.compress_threshold = self.config.get('compress_threshold', 1024)  # Bytes
        
        self.formats = {
            'json': self._serialize_json,
            'pickle': self._serialize_pickle,
            'binary': self._serialize_binary
        }
        
        self.deserialize_formats = {
            'json': self._deserialize_json,
            'pickle': self._deserialize_pickle,
            'binary': self._deserialize_binary
        }
    
    async def serialize_node(self, node: SemanticNode, format: Optional[str] = None) -> Dict[str, Any]:
        format = format or self.default_format
        
        node_data = {
            'id': node.id,
            'label': node.label,
            'node_type': node.node_type,
            'properties': node.properties,
            'confidence': node.confidence,
            'metadata': getattr(node, 'metadata', {}),
        }
        
        if hasattr(node, 'confidence_state'):
            if isinstance(node.confidence_state, SemanticConfidenceState):
                node_data['confidence_state'] = node.confidence_state.value
            else:
                node_data['confidence_state'] = node.confidence_state
        
        if hasattr(node, 'embeddings') and node.embeddings is not None:
            if isinstance(node.embeddings, np.ndarray):
                tensor_data = await self._serialize_tensor(node.embeddings)
                node_data['embeddings'] = tensor_data
            elif isinstance(node.embeddings, torch.Tensor):
                numpy_data = node.embeddings.detach().cpu().numpy()
                tensor_data = await self._serialize_tensor(numpy_data)
                node_data['embeddings'] = tensor_data
                
        if hasattr(node, 'temporal_metadata'):
            node_data['temporal_metadata'] = await self._serialize_temporal_metadata(
                node.temporal_metadata
            )
            
        if hasattr(node, 'source_ids'):
            node_data['source_ids'] = node.source_ids
            
        if hasattr(node, 'parent_ids'):
            node_data['parent_ids'] = node.parent_ids
            
        if hasattr(node, 'child_ids'):
            node_data['child_ids'] = node.child_ids
            
        if hasattr(node, 'activation'):
            node_data['activation'] = float(node.activation)
            
        predictive_fields = [
            'anticipated_activations', 'activation_errors', 
            'precision_values', 'anticipated_properties'
        ]
        for field in predictive_fields:
            if hasattr(node, field):
                node_data[field] = getattr(node, field)
            
        if format in self.formats:
            return await self.formats[format](node_data)
        else:
            self.logger.warning(f"Unknown serialization format: {format}, using default")
            return await self.formats[self.default_format](node_data)
    
    async def deserialize_node(self, data: Union[Dict[str, Any], bytes, str], 
                              format: Optional[str] = None) -> SemanticNode:
        format = format or self.default_format
        
        if format in self.deserialize_formats:
            node_data = await self.deserialize_formats[format](data)
        else:
            self.logger.warning(f"Unknown deserialization format: {format}, using default")
            node_data = await self.deserialize_formats[self.default_format](data)
            
        embeddings = None
        if 'embeddings' in node_data:
            embeddings = await self._deserialize_tensor(node_data['embeddings'])
            del node_data['embeddings']  # Remove from dict to avoid duplicate
            
        temporal_metadata = None
        if 'temporal_metadata' in node_data:
            temporal_metadata = await self._deserialize_temporal_metadata(
                node_data['temporal_metadata']
            )
            del node_data['temporal_metadata']  # Remove from dict
            
        if 'confidence_state' in node_data:
            confidence_state = node_data['confidence_state']
            if isinstance(confidence_state, str):
                try:
                    node_data['confidence_state'] = SemanticConfidenceState(confidence_state)
                except ValueError:
                    pass
        
        constructor_fields = {
            'id', 'label', 'node_type', 'properties', 'confidence', 
            'confidence_state', 'source_ids'
        }
        constructor_args = {k: v for k, v in node_data.items() if k in constructor_fields}
        
        node = SemanticNode(**constructor_args)
        
        if embeddings is not None:
            node.embeddings = embeddings
            if isinstance(embeddings, np.ndarray):
                node.tensor_representation = torch.tensor(
                    embeddings, dtype=torch.float32
                )
        
        if temporal_metadata is not None:
            node.temporal_metadata = temporal_metadata
        
        for key, value in node_data.items():
            if key not in constructor_fields and key not in {'embeddings', 'temporal_metadata'}:
                setattr(node, key, value)
                
        return node
    
    async def serialize_relation(self, relation: SemanticRelation, 
                                format: Optional[str] = None) -> Dict[str, Any]:
        format = format or self.default_format
        
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
        
        if relation.embedding is not None:
            if isinstance(relation.embedding, np.ndarray):
                tensor_data = await self._serialize_tensor(relation.embedding)
                relation_data['embedding'] = tensor_data
            elif isinstance(relation.embedding, torch.Tensor):
                numpy_data = relation.embedding.detach().cpu().numpy()
                tensor_data = await self._serialize_tensor(numpy_data)
                relation_data['embedding'] = tensor_data
                
        if hasattr(relation, 'temporal_metadata'):
            relation_data['temporal_metadata'] = await self._serialize_temporal_metadata(
                relation.temporal_metadata
            )
            
        if hasattr(relation, 'anticipated_weights'):
            relation_data['anticipated_weights'] = relation.anticipated_weights
            
        if hasattr(relation, 'weight_prediction_errors'):
            relation_data['weight_prediction_errors'] = dict(relation.weight_prediction_errors)
            
        if hasattr(relation, 'weight_precision'):
            relation_data['weight_precision'] = relation.weight_precision
            
        if format in self.formats:
            return await self.formats[format](relation_data)
        else:
            self.logger.warning(f"Unknown serialization format: {format}, using default")
            return await self.formats[self.default_format](relation_data)
    
    async def deserialize_relation(self, data: Union[Dict[str, Any], bytes, str], 
                                  format: Optional[str] = None) -> SemanticRelation:
        format = format or self.default_format
        
        if format in self.deserialize_formats:
            relation_data = await self.deserialize_formats[format](data)
        else:
            self.logger.warning(f"Unknown deserialization format: {format}, using default")
            relation_data = await self.deserialize_formats[self.default_format](data)
            
        embedding = None
        if 'embedding' in relation_data:
            embedding = await self._deserialize_tensor(relation_data['embedding'])
            del relation_data['embedding']  # Remove from dict
            
        temporal_metadata = None
        if 'temporal_metadata' in relation_data:
            temporal_metadata = await self._deserialize_temporal_metadata(
                relation_data['temporal_metadata']
            )
            del relation_data['temporal_metadata']  # Remove from dict
            
        prediction_fields = [
            'anticipated_weights', 'weight_prediction_errors', 'weight_precision'
        ]
        prediction_data = {}
        for field in prediction_fields:
            if field in relation_data:
                prediction_data[field] = relation_data[field]
                del relation_data[field]  # Remove from main dict
                
        relation = SemanticRelation(**relation_data)
        
        if embedding is not None:
            relation.embedding = embedding
        
        if temporal_metadata is not None:
            relation.temporal_metadata = temporal_metadata
            
        for field, value in prediction_data.items():
            setattr(relation, field, value)
            
        if 'weight_prediction_errors' in prediction_data:
            relation.weight_prediction_errors = defaultdict(list, relation.weight_prediction_errors)
            
        return relation
    
    async def _serialize_tensor(self, tensor: np.ndarray) -> Dict[str, Any]:
        data = tensor_data['data']
        shape = tensor_data['shape']
        dtype = tensor_data['dtype']
        compressed = tensor_data.get('compressed', False)
        
        if compressed:
            data = zlib.decompress(data)
            
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    
    async def _serialize_temporal_metadata(self, 
                                          temporal_metadata: AdaptiveTemporalMetadata) -> Dict[str, Any]:
        temporal_metadata = AdaptiveTemporalMetadata(
            creation_time=data['creation_time'],
            last_accessed=data['last_accessed'],
            last_modified=data['last_modified'],
            access_count=data['access_count'],
            modification_count=data['modification_count'],
            contextual_half_lives=data['contextual_half_lives']
        )
        
        temporal_metadata.predicted_relevance = data['predicted_relevance']
        temporal_metadata.relevance_errors = defaultdict(list)
        for k, v in data['relevance_errors'].items():
            temporal_metadata.relevance_errors[k] = v
        temporal_metadata.relevance_precision = data['relevance_precision']
        
        return temporal_metadata
    
    async def _serialize_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = await self._process_dict_for_json(v)
            elif isinstance(v, list):
                result[k] = await self._process_list_for_json(v)
            elif isinstance(v, (np.ndarray, bytes)):
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
                try:
                    json.dumps(v)
                    result[k] = v
                except (TypeError, OverflowError):
                    result[k] = {
                        'type': 'str_repr',
                        'data': str(v)
                    }
        return result
    
    async def _process_list_for_json(self, lst: List) -> List:
        if isinstance(data, str):
            parsed_data = json.loads(data)
        else:
            parsed_data = data
            
        result = {}
        
        for key, value in parsed_data.items():
            if isinstance(value, dict) and 'type' in value and 'data' in value:
                type_name = value['type']
                if type_name == 'binary':
                    import base64
                    result[key] = base64.b64decode(value['data'])
                elif type_name == 'defaultdict':
                    result[key] = defaultdict(list, value['data'])
                elif type_name == 'set':
                    result[key] = set(value['data'])
                elif type_name == 'tuple':
                    result[key] = tuple(value['data'])
                elif type_name == 'str_repr':
                    result[key] = value['data']
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = await self._process_dict_from_json(value)
            elif isinstance(value, list):
                result[key] = await self._process_list_from_json(value)
            else:
                result[key] = value
                
        return result
    
    async def _process_dict_from_json(self, d: Dict) -> Dict:
        result = []
        for v in lst:
            if isinstance(v, dict):
                if 'type' in v and 'data' in v:
                    type_name = v['type']
                    if type_name == 'binary':
                        import base64
                        result.append(base64.b64decode(v['data']))
                    elif type_name == 'defaultdict':
                        result.append(defaultdict(list, v['data']))
                    elif type_name == 'set':
                        result.append(set(v['data']))
                    elif type_name == 'tuple':
                        result.append(tuple(v['data']))
                    elif type_name == 'str_repr':
                        result.append(v['data'])
                    else:
                        result.append(v)
                else:
                    result.append(await self._process_dict_from_json(v))
            elif isinstance(v, list):
                result.append(await self._process_list_from_json(v))
            else:
                result.append(v)
        return result
    
    async def _serialize_pickle(self, data: Dict[str, Any]) -> bytes:
        return pickle.loads(data)
    
    async def _serialize_binary(self, data: Dict[str, Any]) -> bytes:
        pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(pickled_data)
    
    async def _deserialize_binary(self, data: bytes) -> Dict[str, Any]:
    Abstract base class for storage backends.
    Defines interface that all storage backends must implement.
        Initialize storage backend.
        
        Args:
            config: Configuration dictionary
        Initialize the storage backend.
        
        Returns:
            bool: True if successful, False otherwise
        Store node data.
        
        Args:
            node_id: ID of the node
            node_data: Serialized node data
            
        Returns:
            bool: True if successful, False otherwise
        Retrieve node data.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[Any]: Node data if found, None otherwise
        Store relation data.
        
        Args:
            relation_id: ID of the relation
            relation_data: Serialized relation data
            
        Returns:
            bool: True if successful, False otherwise
        Retrieve relation data.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            Optional[Any]: Relation data if found, None otherwise
        Store metadata about the semantic network.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            bool: True if successful, False otherwise
        Retrieve metadata about the semantic network.
        
        Returns:
            Optional[Dict[str, Any]]: Metadata if found, None otherwise
        List all stored node IDs.
        
        Returns:
            List[str]: List of node IDs
        List all stored relation IDs.
        
        Returns:
            List[str]: List of relation IDs
        Delete node data.
        
        Args:
            node_id: ID of the node
            
        Returns:
            bool: True if successful, False otherwise
        Delete relation data.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            bool: True if successful, False otherwise
        Get last modification time for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Optional[float]: Timestamp if found, None otherwise
        Create a backup of the current storage.
        
        Returns:
            str: Backup identifier or path
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
        
        self.format_extensions = {
            'json': '.json',
            'pickle': '.pkl',
            'binary': '.bin'
        }
        
        self.last_modified_cache = {}
        self.cache_expiry = 60  # Seconds
        self.last_cache_update = 0
    
    async def initialize(self) -> bool:
        try:
            self.base_path.mkdir(exist_ok=True, parents=True)
            self.nodes_path.mkdir(exist_ok=True)
            self.relations_path.mkdir(exist_ok=True)
            
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
        try:
            safe_id = self._safe_filename(node_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            node_dir = self.nodes_path / safe_id[0:2]
            node_dir.mkdir(exist_ok=True)
            
            if self.compression:
                file_path = node_dir / f"{safe_id}{file_ext}.gz"
                async with aiofiles.open(file_path, 'wb') as f:
                    if isinstance(node_data, (dict, list)):
                        data_bytes = json.dumps(node_data).encode('utf-8')
                    elif isinstance(node_data, str):
                        data_bytes = node_data.encode('utf-8')
                    else:
                        data_bytes = node_data
                        
                    compressed_data = gzip.compress(data_bytes)
                    await f.write(compressed_data)
            else:
                file_path = node_dir / f"{safe_id}{file_ext}"
                async with aiofiles.open(file_path, 'w' if self.format == 'json' else 'wb') as f:
                    if self.format == 'json':
                        if isinstance(node_data, (dict, list)):
                            await f.write(json.dumps(node_data))
                        elif isinstance(node_data, str):
                            await f.write(node_data)
                        else:
                            await f.write(str(node_data))
                    else:
                        await f.write(node_data)
                        
            self.last_modified_cache[node_id] = time.time()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store node {node_id}: {e}")
            return False
    
    async def retrieve_node(self, node_id: str) -> Optional[Any]:
        try:
            safe_id = self._safe_filename(node_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            node_dir = self.nodes_path / safe_id[0:2]
            
            compressed_path = node_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = node_dir / f"{safe_id}{file_ext}"
            
            if compressed_path.exists():
                async with aiofiles.open(compressed_path, 'rb') as f:
                    compressed_data = await f.read()
                    data = gzip.decompress(compressed_data)
                    
                    if self.format == 'json':
                        return json.loads(data.decode('utf-8'))
                    else:
                        return data
            elif uncompressed_path.exists():
                if self.format == 'json':
                    async with aiofiles.open(uncompressed_path, 'r') as f:
                        data = await f.read()
                        return json.loads(data)
                else:
                    async with aiofiles.open(uncompressed_path, 'rb') as f:
                        return await f.read()
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve node {node_id}: {e}")
            return None
    
    async def store_relation(self, relation_id: str, relation_data: Any) -> bool:
        try:
            safe_id = self._safe_filename(relation_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            relation_dir = self.relations_path / safe_id[0:2]
            relation_dir.mkdir(exist_ok=True)
            
            if self.compression:
                file_path = relation_dir / f"{safe_id}{file_ext}.gz"
                async with aiofiles.open(file_path, 'wb') as f:
                    if isinstance(relation_data, (dict, list)):
                        data_bytes = json.dumps(relation_data).encode('utf-8')
                    elif isinstance(relation_data, str):
                        data_bytes = relation_data.encode('utf-8')
                    else:
                        data_bytes = relation_data
                        
                    compressed_data = gzip.compress(data_bytes)
                    await f.write(compressed_data)
            else:
                file_path = relation_dir / f"{safe_id}{file_ext}"
                async with aiofiles.open(file_path, 'w' if self.format == 'json' else 'wb') as f:
                    if self.format == 'json':
                        if isinstance(relation_data, (dict, list)):
                            await f.write(json.dumps(relation_data))
                        elif isinstance(relation_data, str):
                            await f.write(relation_data)
                        else:
                            await f.write(str(relation_data))
                    else:
                        await f.write(relation_data)
                        
            self.last_modified_cache[relation_id] = time.time()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store relation {relation_id}: {e}")
            return False
    
    async def retrieve_relation(self, relation_id: str) -> Optional[Any]:
        try:
            safe_id = self._safe_filename(relation_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            relation_dir = self.relations_path / safe_id[0:2]
            
            compressed_path = relation_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = relation_dir / f"{safe_id}{file_ext}"
            
            if compressed_path.exists():
                async with aiofiles.open(compressed_path, 'rb') as f:
                    compressed_data = await f.read()
                    data = gzip.decompress(compressed_data)
                    
                    if self.format == 'json':
                        return json.loads(data.decode('utf-8'))
                    else:
                        return data
            elif uncompressed_path.exists():
                if self.format == 'json':
                    async with aiofiles.open(uncompressed_path, 'r') as f:
                        data = await f.read()
                        return json.loads(data)
                else:
                    async with aiofiles.open(uncompressed_path, 'rb') as f:
                        return await f.read()
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve relation {relation_id}: {e}")
            return None
    
    async def store_metadata(self, metadata: Dict[str, Any]) -> bool:
        try:
            metadata['last_updated'] = time.time()
            
            metadata_path = self.base_path / 'metadata.json'
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to store metadata: {e}")
            return False
    
    async def retrieve_metadata(self) -> Optional[Dict[str, Any]]:
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
        nodes = []
        
        try:
            for subdir in self.nodes_path.iterdir():
                if subdir.is_dir():
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            filename = file_path.name
                            
                            for ext in ['.gz'] + list(self.format_extensions.values()):
                                if filename.endswith(ext):
                                    filename = filename[:-len(ext)]
                                    
                            nodes.append(filename)
                            
            return nodes
        except Exception as e:
            self.logger.error(f"Failed to list nodes: {e}")
            return []
    
    async def list_relations(self) -> List[str]:
        relations = []
        
        try:
            for subdir in self.relations_path.iterdir():
                if subdir.is_dir():
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            filename = file_path.name
                            
                            for ext in ['.gz'] + list(self.format_extensions.values()):
                                if filename.endswith(ext):
                                    filename = filename[:-len(ext)]
                                    
                            relations.append(filename)
                            
            return relations
        except Exception as e:
            self.logger.error(f"Failed to list relations: {e}")
            return []
    
    async def delete_node(self, node_id: str) -> bool:
        try:
            safe_id = self._safe_filename(node_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            node_dir = self.nodes_path / safe_id[0:2]
            
            compressed_path = node_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = node_dir / f"{safe_id}{file_ext}"
            
            deleted = False
            
            if compressed_path.exists():
                compressed_path.unlink()
                deleted = True
                
            if uncompressed_path.exists():
                uncompressed_path.unlink()
                deleted = True
                
            if node_id in self.last_modified_cache:
                del self.last_modified_cache[node_id]
                
            return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    async def delete_relation(self, relation_id: str) -> bool:
        try:
            safe_id = self._safe_filename(relation_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            relation_dir = self.relations_path / safe_id[0:2]
            
            compressed_path = relation_dir / f"{safe_id}{file_ext}.gz"
            uncompressed_path = relation_dir / f"{safe_id}{file_ext}"
            
            deleted = False
            
            if compressed_path.exists():
                compressed_path.unlink()
                deleted = True
                
            if uncompressed_path.exists():
                uncompressed_path.unlink()
                deleted = True
                
            if relation_id in self.last_modified_cache:
                del self.last_modified_cache[relation_id]
                
            return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete relation {relation_id}: {e}")
            return False
    
    async def get_last_modified(self, entity_id: str) -> Optional[float]:
        if entity_id in self.last_modified_cache:
            return self.last_modified_cache[entity_id]
            
        try:
            safe_id = self._safe_filename(entity_id)
            file_ext = self.format_extensions.get(self.format, '.json')
            
            node_dir = self.nodes_path / safe_id[0:2]
            node_compressed = node_dir / f"{safe_id}{file_ext}.gz"
            node_uncompressed = node_dir / f"{safe_id}{file_ext}"
            
            relation_dir = self.relations_path / safe_id[0:2]
            relation_compressed = relation_dir / f"{safe_id}{file_ext}.gz"
            relation_uncompressed = relation_dir / f"{safe_id}{file_ext}"
            
            for path in [node_compressed, node_uncompressed, relation_compressed, relation_uncompressed]:
                if path.exists():
                    mtime = path.stat().st_mtime
                    self.last_modified_cache[entity_id] = mtime
                    return mtime
                    
            return None
        except Exception as e:
            self.logger.error(f"Failed to get last modified time for {entity_id}: {e}")
            return None
    
    async def create_backup(self) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.base_path.parent / f"semantic_backup_{timestamp}"
            backup_dir.mkdir(parents=True)
            
            nodes_backup_dir = backup_dir / 'nodes'
            nodes_backup_dir.mkdir()
            
            relations_backup_dir = backup_dir / 'relations'
            relations_backup_dir.mkdir()
            
            for subdir in self.nodes_path.iterdir():
                if subdir.is_dir():
                    backup_subdir = nodes_backup_dir / subdir.name
                    backup_subdir.mkdir()
                    
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            import shutil
                            shutil.copy2(file_path, backup_subdir / file_path.name)
            
            for subdir in self.relations_path.iterdir():
                if subdir.is_dir():
                    backup_subdir = relations_backup_dir / subdir.name
                    backup_subdir.mkdir()
                    
                    for file_path in subdir.iterdir():
                        if file_path.is_file():
                            import shutil
                            shutil.copy2(file_path, backup_subdir / file_path.name)
            
            metadata_path = self.base_path / 'metadata.json'
            if metadata_path.exists():
                import shutil
                shutil.copy2(metadata_path, backup_dir / 'metadata.json')
                
            return str(backup_dir)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return ""
    
    async def close(self) -> None:
        safe_id = ''.join(c if c.isalnum() or c in '-_' else '_' for c in entity_id)
        
        if len(safe_id) > 255:
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
        try:
            self.db_path.parent.mkdir(exist_ok=True, parents=True)
            
            self.conn = await aiosqlite.connect(self.db_path)
            
            await self.conn.execute("PRAGMA journal_mode=WAL")
            
            await self.conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    format TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    modified_at REAL NOT NULL
                )
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    data BLOB NOT NULL,
                    format TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    modified_at REAL NOT NULL
                )
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    modified_at REAL NOT NULL
                )
        Store node data in SQLite.
        
        Args:
            node_id: ID of the node
            node_data: Serialized node data
            
        Returns:
            bool: True if successful, False otherwise
        Retrieve node data from SQLite.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[Any]: Node data if found, None otherwise
        Store relation data in SQLite.
        
        Args:
            relation_id: ID of the relation
            relation_data: Serialized relation data
            
        Returns:
            bool: True if successful, False otherwise
        Retrieve relation data from SQLite.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            Optional[Any]: Relation data if found, None otherwise
        Store metadata in SQLite.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            bool: True if successful, False otherwise
        Retrieve metadata from SQLite.
        
        Returns:
            Optional[Dict[str, Any]]: Metadata if found, None otherwise
        List all stored node IDs.
        
        Returns:
            List[str]: List of node IDs
        List all stored relation IDs.
        
        Returns:
            List[str]: List of relation IDs
        Delete node from SQLite.
        
        Args:
            node_id: ID of the node
            
        Returns:
            bool: True if successful, False otherwise
        Delete relation from SQLite.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            bool: True if successful, False otherwise
        Get last modification time for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Optional[float]: Timestamp if found, None otherwise
        Create a backup of the SQLite database.
        
        Returns:
            str: Backup file path
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
        
        self.serialization_manager = SerializationManager(self.config.get('serialization', {}))
        
        self.storage_type = self.config.get('storage_type', 'file')
        self.storage_backend = self._create_storage_backend()
        
        self.last_sync_time = 0
        self.nodes_sync_state = {}  # node_id -> last_sync_time
        self.relations_sync_state = {}  # relation_id -> last_sync_time
        
        self.sync_task = None
        self.sync_interval = self.config.get('sync_interval', 300)  # 5 minutes
        self.sync_running = False
        
        self.max_retry_attempts = self.config.get('max_retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # seconds
        
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
        Save current semantic network state.
        
        Args:
            incremental: Whether to do incremental update or full save
            
        Returns:
            Dict with save statistics
        saved_nodes = 0
        saved_relations = 0
        errors = 0
        start_time = time.time()
        
        for node_id, node in self.semantic_network.nodes.items():
            try:
                node_data = await self.serialization_manager.serialize_node(node)
                
                success = await self.storage_backend.store_node(node_id, node_data)
                
                if success:
                    saved_nodes += 1
                    self.nodes_sync_state[node_id] = time.time()
                else:
                    errors += 1
            except Exception as e:
                self.logger.error(f"Error saving node {node_id}: {e}")
                errors += 1
                
        for relation_id, relation in self.semantic_network.relations.items():
            try:
                relation_data = await self.serialization_manager.serialize_relation(relation)
                
                success = await self.storage_backend.store_relation(relation_id, relation_data)
                
                if success:
                    saved_relations += 1
                    self.relations_sync_state[relation_id] = time.time()
                else:
                    errors += 1
            except Exception as e:
                self.logger.error(f"Error saving relation {relation_id}: {e}")
                errors += 1
                
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
        Load semantic network state from storage.
        
        Returns:
            Dict with load statistics
        Track node change for incremental save.
        
        Args:
            node_id: ID of changed node
        Track relation change for incremental save.
        
        Args:
            relation_id: ID of changed relation
        Schedule regular state backups.
        
        Args:
            interval: Backup interval in seconds
        try:
            while self.sync_running:
                await asyncio.sleep(self.sync_interval)
                
                try:
                    self.logger.info("Performing scheduled incremental save")
                    async with self.sync_lock:
                        result = await self._perform_incremental_save()
                    
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
        try:
            await self.save_state(incremental=True)
            
            backup_path = await self.storage_backend.create_backup()
            
            if backup_path:
                metadata = await self.storage_backend.retrieve_metadata() or {}
                metadata['last_backup'] = time.time()
                metadata['last_backup_path'] = backup_path
                await self.storage_backend.store_metadata(metadata)
                
                self.logger.info(f"Created backup at {backup_path}")
                
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return ""
    
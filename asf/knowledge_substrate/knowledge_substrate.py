# asf/knowledge_substrate/knowledge_substrate.py
import time
from asf.core.enums import PerceptualInputType, PerceptualEventType
from asf.knowledge_substrate.temporal.processing_engine import TemporalProcessingEngine
from asf.knowledge_substrate.perception.entity import PerceptualEntity
from asf.knowledge_substrate.confidence.bayesian_updater import BayesianConfidenceUpdater
from asf.knowledge_substrate.linking.entity_gnn import EntityLinkingGNN
from asf.knowledge_substrate.causal.representation import CausalRepresentationLearner
from asf.knowledge_substrate.memory.energy_based import EnergyBasedMemoryManager
from asf.knowledge_substrate.extraction.text import TextFeatureExtractor
from asf.knowledge_substrate.extraction.image import ImageFeatureExtractor

class KnowledgeSubstrateLayer:
    """
    Main controller class for the Knowledge Substrate Layer (Layer 1).
    Manages integration of all components and interfaces with higher layers.
    """
    def __init__(self, config=None):
        # Initialize configuration
        self.config = config or {}
        
        # Core components
        self.entities = {}  # entity_id -> PerceptualEntity
        self.entity_relations = []  # (source_idx, target_idx, relation_type)
        
        # Feature extractors for different modalities
        self.feature_extractors = {
            PerceptualInputType.TEXT: TextFeatureExtractor(),
            PerceptualInputType.IMAGE: ImageFeatureExtractor(),
            # Other modalities would be initialized here
        }
        
        # Enhanced components
        self.bayesian_updater = BayesianConfidenceUpdater()
        self.entity_linking_gnn = EntityLinkingGNN()
        self.temporal_processor = TemporalProcessingEngine()
        self.causal_learner = CausalRepresentationLearner()
        self.memory_manager = EnergyBasedMemoryManager()
        
        # Processing statistics
        self.stats = {
            "entities_processed": 0,
            "relations_discovered": 0,
            "confidence_updates": 0,
            "causal_relations": 0,
            "temporal_patterns": 0
        }
    
    def process_input(self, input_data, input_type, context=None):
        """
        Process a new perceptual input and create/update entities
        
        Parameters:
        - input_data: Raw input data (text, image, etc.)
        - input_type: PerceptualInputType indicating modality
        - context: Optional processing context
        
        Returns the entity_id of the processed entity
        """
        # Generate a unique ID for this entity
        entity_id = f"{input_type.value}_{int(time.time()*1000)}"
        
        # Extract features using appropriate extractor
        if input_type in self.feature_extractors:
            extractor = self.feature_extractors[input_type]
            
            # Extract features with semiotic weighting
            weighted_features = extractor.extract_weighted_features(input_data, context)
            
            # Create entity with extracted features
            entity = PerceptualEntity(entity_id, input_type)
            
            # Add weighted features
            for name, feature_info in weighted_features.items():
                entity.add_feature(
                    name, 
                    feature_info['value'], 
                    confidence=feature_info['importance']
                )
            
            # Store entity
            self.entities[entity_id] = entity
            
            # Add to memory manager
            self.memory_manager.add_entity(entity, context)
            
            # Update statistics
            self.stats["entities_processed"] += 1
            
            # Process temporal aspects
            self._process_temporal_aspects(entity, input_type)
            
            # Update causal model
            self._update_causal_model()
            
            # Return the entity ID
            return entity_id
        
        # Return None if input type not supported
        return None
    
    def update_entity_confidence(self, entity_id, relevant, context=None):
        """
        Update entity confidence using Bayesian updater
        
        Parameters:
        - entity_id: ID of entity to update
        - relevant: Boolean indicating if entity was relevant in current context
        - context: Optional context for prediction
        """
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            
            # Convert context to vector if needed
            context_vector = None
            if context is not None:
                if hasattr(context, 'get_feature_vector'):
                    context_vector = context.get_feature_vector()
                elif isinstance(context, dict) and 'feature_vector' in context:
                    context_vector = context['feature_vector']
            
            # Update confidence using Bayesian updater
            new_confidence = self.bayesian_updater.update_confidence(
                entity_id, 
                relevant, 
                context_vector
            )
            
            # Update entity confidence state
            new_state = self.bayesian_updater.get_confidence_state(new_confidence)
            entity.update_confidence_state(new_state, new_confidence)
            
            # Update statistics
            self.stats["confidence_updates"] += 1
            
            return new_confidence
        return None
    
    def predict_entity_relevance(self, entity_id, context):
        """
        Predict entity relevance in a given context
        
        Parameters:
        - entity_id: ID of entity to evaluate
        - context: Context for prediction
        
        Returns relevance probability between 0 and 1
        """
        # Convert context to vector if needed
        context_vector = None
        if context is not None:
            if hasattr(context, 'get_feature_vector'):
                context_vector = context.get_feature_vector()
            elif isinstance(context, dict) and 'feature_vector' in context:
                context_vector = context['feature_vector']
        
        # Use Bayesian updater to predict relevance
        return self.bayesian_updater.predict_relevance(entity_id, context_vector)
    
    def find_similar_entities(self, entity_id, modality=None, top_k=5):
        """
        Find entities similar to the specified entity
        
        Parameters:
        - entity_id: ID of query entity
        - modality: Optional filter by modality
        - top_k: Maximum number of results to return
        
        Returns list of (entity, similarity_score) tuples
        """
        if entity_id not in self.entities:
            return []
        
        query_entity = self.entities[entity_id]
        
        # Filter entities by modality if specified
        if modality is not None:
            filtered_entities = [e for e in self.entities.values() if e.input_type == modality]
        else:
            filtered_entities = list(self.entities.values())
        
        # Remove query entity from candidates
        filtered_entities = [e for e in filtered_entities if e.id != entity_id]
        
        # Update entity embeddings if needed
        self._ensure_entity_embeddings()
        
        # Use GNN to find similar entities
        similar_entities = self.entity_linking_gnn.find_similar_entities(
            query_entity,
            filtered_entities,
            top_k
        )
        
        return similar_entities
    
    def get_entity(self, entity_id):
        """Get entity by ID, using memory manager"""
        return self.memory_manager.get_entity(entity_id)
    
    def add_cross_modal_link(self, source_id, target_id):
        """
        Create a cross-modal link between two entities
        
        Parameters:
        - source_id: ID of source entity
        - target_id: ID of target entity
        
        Returns True if link created, False otherwise
        """
        if source_id in self.entities and target_id in self.entities:
            source = self.entities[source_id]
            target = self.entities[target_id]
            
            # Only link across different modalities
            if source.input_type != target.input_type:
                # Add cross-modal links in both directions
                source.add_cross_modal_link(target_id)
                target.add_cross_modal_link(source_id)
                
                # Add to relations list for GNN
                source_idx = list(self.entities.keys()).index(source_id)
                target_idx = list(self.entities.keys()).index(target_id)
                rel_type = self.entity_linking_gnn._get_relation_type(
                    source.input_type,
                    target.input_type
                )
                
                self.entity_relations.append((source_idx, target_idx, rel_type))
                
                # Update statistics
                self.stats["relations_discovered"] += 1
                
                return True
        
        return False
    
    def suggest_cross_modal_links(self, similarity_threshold=0.7):
        """
        Suggest potential cross-modal links between entities
        
        Parameters:
        - similarity_threshold: Minimum similarity score for suggested links
        
        Returns list of (entity1, entity2, similarity) tuples
        """
        # Update entity embeddings if needed
        self._ensure_entity_embeddings()
        
        # Get suggestions from GNN
        return self.entity_linking_gnn.suggest_cross_modal_links(
            list(self.entities.values()),
            similarity_threshold
        )
    
    def add_temporal_event(self, entity_id, event_type, event_data):
        """
        Add a temporal event for an entity
        
        Parameters:
        - entity_id: ID of entity
        - event_type: Type of event
        - event_data: Event data
        
        Returns pattern detection result if a pattern is found
        """
        if entity_id in self.entities:
            # Get event type as string
            if isinstance(event_type, PerceptualEventType):
                sequence_type = event_type.value
            else:
                sequence_type = str(event_type)
            
            # Register sequence if needed
            self.temporal_processor.register_sequence(entity_id, sequence_type)
            
            # Add event and check for patterns
            pattern = self.temporal_processor.add_event(entity_id, sequence_type, event_data)
            
            if pattern:
                # Update statistics
                self.stats["temporal_patterns"] += 1
            
            return pattern
        
        return None
    
    def perform_causal_intervention(self, entity_id, feature_name, new_value):
        """
        Perform a causal intervention to test causal relationships
        
        Parameters:
        - entity_id: ID of entity to modify
        - feature_name: Name of feature to intervene on
        - new_value: New value to set
        
        Returns True if intervention successful, False otherwise
        """
        return self.causal_learner.perform_causal_intervention(
            entity_id,
            feature_name,
            new_value,
            self.entities
        )
    
    def get_causal_explanation(self, entity_id, feature_name):
        """
        Get causal explanation for a feature
        
        Parameters:
        - entity_id: ID of entity
        - feature_name: Name of feature to explain
        
        Returns textual explanation of causal influences
        """
        return self.causal_learner.generate_causal_explanation(entity_id, feature_name)
    
    def sample_entities_by_energy(self, context=None, n=5):
        """
        Sample entities using energy-based model
        
        Parameters:
        - context: Optional context for relevance calculation
        - n: Number of entities to sample
        
        Returns list of sampled entities
        """
        return self.memory_manager.sample_entities(context, n)
    
    def get_layer_statistics(self):
        """Get statistics about layer operations"""
        # Combine stats from various components
        memory_stats = self.memory_manager.get_memory_statistics()
        
        return {
            **self.stats,
            "memory_usage": memory_stats
        }
    
    def _process_temporal_aspects(self, entity, input_type):
        """Process temporal aspects of a new entity"""
        # Add NEW_INPUT event
        self.add_temporal_event(
            entity.id,
            PerceptualEventType.NEW_INPUT,
            entity
        )
        
        # Get entities of same type for temporal context
        same_type_entities = [
            e for e in self.entities.values() 
            if e.input_type == input_type and e.id != entity.id
        ]
        
        # Sort by recency (most recent first)
        same_type_entities.sort(
            key=lambda e: e.temporal_metadata.last_access_time,
            reverse=True
        )
        
        # Use recent entities for context
        if same_type_entities:
            # Maintain temporal context for this modality
            self.temporal_processor.maintain_temporal_context(
                f"recent_{input_type.value}",
                same_type_entities[:5]  # Keep 5 most recent
            )
    
    def _update_causal_model(self):
        """Update causal model with current entities"""
        entity_features = {
            entity_id: entity.features
            for entity_id, entity in self.entities.items()
        }
        
        self.causal_learner.update_from_observations(entity_features)
    
    def _ensure_entity_embeddings(self):
        """Ensure all entities have GNN embeddings"""
        # Check if we need to update entity embeddings
        entities_list = list(self.entities.values())
        
        # Only update if we have entities and relations
        if entities_list and self.entity_relations:
            self.entity_linking_gnn.update_entity_embeddings(
                entities_list,
                self.entity_relations
            )

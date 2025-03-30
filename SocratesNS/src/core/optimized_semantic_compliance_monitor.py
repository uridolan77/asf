import hashlib
import numpy as np
from src.core.utils import LRUCache
from src.core.utils.utils import EntityTracker, TopicDetector
from src.core.utils.utils import OptimizedSemanticAnalyzer
from src.core.utils.utils import ComplianceFramework, ComplianceState
class OptimizedSemanticComplianceMonitor:
    """
    Optimized semantic monitoring system that efficiently tracks compliance
    state during text generation.
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        
        # Initialize semantic analysis components
        self.semantic_analyzer = OptimizedSemanticAnalyzer(compliance_config)
        self.entity_tracker = EntityTracker(compliance_config)
        self.topic_detector = TopicDetector(compliance_config)
        
        # Configure semantic state management
        self.state_dim = compliance_config.get("semantic_state_dim", 64)
        self.sliding_window_size = compliance_config.get("sliding_window", 200)
        
        # Initialize concept sensitivity mapping
        self.concept_sensitivity = compliance_config.get("concept_sensitivity", {})
        
        # Caching and optimization
        self.update_trigger_words = set(compliance_config.get("update_trigger_words", []))
        self.full_update_interval = compliance_config.get("full_update_interval", 20)
        self.state_cache = LRUCache(maxsize=100)
        
    def initialize_optimized(self, prompt, frameworks):
        """
        Initialize semantic state with optimized analysis
        
        Args:
            prompt: Input prompt text
            frameworks: Applicable regulatory frameworks
            
        Returns:
            Initial semantic state
        """
        # Get semantic representation of prompt
        semantic_analysis = self.semantic_analyzer.analyze(prompt)
        
        # Extract topics with optimized detector
        topics = self.topic_detector.detect_topics(prompt)
        
        # Track relevant entities
        entities = self.entity_tracker.extract_entities(prompt)
        
        # Get sensitive concepts from frameworks
        sensitive_concepts = self._get_framework_sensitive_concepts(frameworks)
        
        # Create initial state with optimized structure
        state = {
            "semantic_embedding": semantic_analysis["embedding"],
            "text_fingerprint": self._get_text_fingerprint(prompt),
            "topics": {topic: score for topic, score in topics.items() if score > 0.1},
            "entities": entities,
            "sensitive_concepts": sensitive_concepts,
            "concept_scores": self._score_concepts(semantic_analysis, sensitive_concepts),
            "token_count": len(prompt.split()),
            "text_buffer": prompt[-self.sliding_window_size:] if len(prompt) > self.sliding_window_size else prompt,
            "risk_score": self._calculate_initial_risk(semantic_analysis, frameworks, topics),
            "frameworks": [f.id for f in frameworks],
            "last_full_update": 0,
            "warnings": []
        }
        
        # Cache initial state
        state_key = self._get_state_key(prompt, frameworks)
        self.state_cache[state_key] = state
        
        return state
        
    def update_optimized(self, state, token_text, generated_text, frameworks):
        """
        Update semantic state with optimized processing
        
        Args:
            state: Current semantic state
            token_text: New token text
            generated_text: Full generated text
            frameworks: Applicable frameworks
            
        Returns:
            Updated semantic state
        """
        # Create copy of state to avoid modifying original
        updated_state = state.copy()
        
        # Update token count and text buffer
        updated_state["token_count"] += 1
        updated_state["text_buffer"] += token_text
        if len(updated_state["text_buffer"]) > self.sliding_window_size:
            updated_state["text_buffer"] = updated_state["text_buffer"][-self.sliding_window_size:]
        
        # Check if full update is needed
        needs_full_update = self._needs_full_update(
            token_text, 
            updated_state["token_count"], 
            updated_state["last_full_update"]
        )
        
        if needs_full_update:
            # Perform full semantic update
            updated_state = self._perform_full_update(
                updated_state,
                generated_text,
                frameworks
            )
        else:
            # Perform lightweight update for efficiency
            updated_state = self._perform_lightweight_update(
                updated_state,
                token_text,
                generated_text
            )
        
        return updated_state
        
    def get_fingerprint(self, state):
        """Get compact fingerprint of semantic state for caching"""
        risk_fp = f"{state['risk_score']:.2f}"
        topics_fp = "-".join(sorted(state["topics"].keys()))
        return f"{risk_fp}|{topics_fp}"
        
    def _get_framework_sensitive_concepts(self, frameworks):
        """Extract sensitive concepts from frameworks"""
        sensitive_concepts = {}
        
        for framework in frameworks:
            # Get framework-specific concepts
            if hasattr(framework, "get_sensitive_concepts"):
                framework_concepts = framework.get_sensitive_concepts()
                
                # Add to overall concept map with framework attribution
                for concept, data in framework_concepts.items():
                    if concept not in sensitive_concepts:
                        sensitive_concepts[concept] = {
                            "frameworks": [framework.id],
                            "severity": data.get("severity", "medium"),
                            "threshold": data.get("threshold", 0.7)
                        }
                    else:
                        # Add framework to existing concept
                        sensitive_concepts[concept]["frameworks"].append(framework.id)
                        
                        # Use highest severity if multiple frameworks define it
                        if self._severity_rank(data.get("severity")) > self._severity_rank(sensitive_concepts[concept]["severity"]):
                            sensitive_concepts[concept]["severity"] = data.get("severity")
                            
                        # Use lowest threshold (most conservative)
                        if data.get("threshold", 1.0) < sensitive_concepts[concept]["threshold"]:
                            sensitive_concepts[concept]["threshold"] = data.get("threshold")
        
        return sensitive_concepts
        
    def _score_concepts(self, semantic_analysis, sensitive_concepts):
        """Score content against sensitive concepts"""
        scores = {}
        
        for concept in sensitive_concepts:
            # Use topic scores if concept matches a topic
            if concept in semantic_analysis.get("topics", {}):
                scores[concept] = semantic_analysis["topics"][concept]
            else:
                # Use embedding similarity for concepts not directly in topics
                # This is a placeholder - would use real concept embeddings
                scores[concept] = 0.1
                
        return scores
        
    def _calculate_initial_risk(self, semantics, frameworks, topics):
        """Calculate initial risk based on semantics and frameworks"""
        # Base risk from topics
        topic_risk = sum(
            score * self.concept_sensitivity.get(topic, 0.5)
            for topic, score in topics.items()
        ) / max(len(topics), 1)
        
        # Risk from framework-specific factors
        framework_risk = 0.0
        for framework in frameworks:
            if hasattr(framework, "calculate_risk"):
                framework_risk = max(framework_risk, framework.calculate_risk(semantics))
        
        # Combine risks (weighted average)
        combined_risk = 0.7 * topic_risk + 0.3 * framework_risk
        
        # Ensure in valid range
        return max(0.0, min(combined_risk, 1.0))
        
    def _needs_full_update(self, token_text, token_count, last_full_update):
        """Determine if full semantic update is needed"""
        # Update on regular intervals
        if token_count - last_full_update >= self.full_update_interval:
            return True
            
        # Update on sentence boundaries
        if any(p in token_text for p in ['.', '!', '?', '\n']):
            return True
            
        # Update on trigger words
        if token_text.lower().strip() in self.update_trigger_words:
            return True
            
        # Otherwise, no full update needed
        return False
        
    def _perform_full_update(self, state, generated_text, frameworks):
        """Perform full semantic state update"""
        # Update text fingerprint
        state["text_fingerprint"] = self._get_text_fingerprint(state["text_buffer"])
        
        # Perform semantic analysis on current buffer
        semantic_analysis = self.semantic_analyzer.analyze(state["text_buffer"])
        
        # Update semantic embedding with weighted average
        alpha = 0.7  # Weight for new information
        state["semantic_embedding"] = self._weighted_combine(
            state["semantic_embedding"],
            semantic_analysis["embedding"],
            alpha
        )
        
        # Update topics
        new_topics = self.topic_detector.detect_topics(state["text_buffer"])
        state["topics"] = self._update_topics(state["topics"], new_topics)
        
        # Update entities
        new_entities = self.entity_tracker.extract_entities(state["text_buffer"])
        state["entities"] = self._merge_entities(state["entities"], new_entities)
        
        # Update concept scores
        state["concept_scores"] = self._score_concepts(
            semantic_analysis, state["sensitive_concepts"]
        )
        
        # Update risk score
        state["risk_score"] = self._calculate_updated_risk(state, frameworks)
        
        # Add warnings if needed
        self._update_warnings(state, generated_text)
        
        # Update last full update counter
        state["last_full_update"] = state["token_count"]
        
        return state
        
    def _perform_lightweight_update(self, state, token_text, generated_text):
        """Perform lightweight semantic update"""
        # Update risk score with minimal computation
        if token_text.strip():  # Non-whitespace token
            # Simple approximation based on token
            risk_adjustment = self._estimate_token_risk(token_text, state["topics"])
            state["risk_score"] = min(1.0, max(0.0, state["risk_score"] + risk_adjustment))
            
            # Check if risk crossed threshold and add warning if so
            if state["risk_score"] > 0.8 and not any(w["type"] == "high_risk" for w in state["warnings"]):
                state["warnings"].append({
                    "type": "high_risk",
                    "message": "Content is approaching compliance boundaries",
                    "position": len(generated_text)
                })
                
        return state
        
    def _weighted_combine(self, old_embedding, new_embedding, alpha):
        """Combine embeddings with weighted average"""
        if not isinstance(old_embedding, np.ndarray):
            old_embedding = np.array(old_embedding)
        if not isinstance(new_embedding, np.ndarray):
            new_embedding = np.array(new_embedding)
            
        # Weighted combination
        combined = (1 - alpha) * old_embedding + alpha * new_embedding
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined
        
    def _update_topics(self, old_topics, new_topics):
        """Update topic scores with new information"""
        updated_topics = old_topics.copy()
        
        # Update existing topics with exponential moving average
        for topic, score in new_topics.items():
            if score > 0.1:  # Filter low-confidence topics
                if topic in updated_topics:
                    updated_topics[topic] = 0.7 * updated_topics[topic] + 0.3 * score
                else:
                    updated_topics[topic] = score
                    
        # Remove topics with very low scores
        updated_topics = {k: v for k, v in updated_topics.items() if v > 0.05}
        
        return updated_topics
        
    def _merge_entities(self, old_entities, new_entities):
        """Merge entity lists with deduplication"""
        merged = old_entities.copy()
        
        # Index existing entities by ID or name
        entity_map = {
            e.get("id", e.get("name", "")): e
            for e in merged
        }
        
        # Add or update entities
        for entity in new_entities:
            entity_id = entity.get("id", entity.get("name", ""))
            
            if entity_id in entity_map:
                # Update existing entity
                entity_map[entity_id].update(entity)
            else:
                # Add new entity
                merged.append(entity)
                
        return merged
        
    def _calculate_updated_risk(self, state, frameworks):
        """Calculate updated risk score"""
        # Factor in sensitive concept scores
        concept_risk = 0.0
        num_concepts = 0
        
        for concept, score in state["concept_scores"].items():
            threshold = state["sensitive_concepts"][concept]["threshold"]
            if score > threshold:
                # Weight by concept severity
                severity_weight = self._severity_weight(state["sensitive_concepts"][concept]["severity"])
                concept_risk += (score - threshold) * severity_weight
                num_concepts += 1
                
        # Normalize concept risk
        if num_concepts > 0:
            concept_risk /= num_concepts
            
        # Apply decay to current risk (risk diminishes if no new issues)
        decay_factor = 0.95
        decayed_risk = state["risk_score"] * decay_factor
        
        # Combine risks
        topic_weight = 0.3
        concept_weight = 0.4
        prev_weight = 0.3
        
        # Get topic risk
        topic_risk = sum(
            score * self.concept_sensitivity.get(topic, 0.5)
            for topic, score in state["topics"].items()
        ) / max(len(state["topics"]), 1)
        
        # Combine component risks
        updated_risk = (
            prev_weight * decayed_risk +
            topic_weight * topic_risk +
            concept_weight * concept_risk
        )
        
        # Ensure in valid range
        return max(0.0, min(updated_risk, 1.0))
        
    def _estimate_token_risk(self, token, topics):
        """Estimate risk contribution from a single token"""
        # Simple heuristic for token risk
        token_lower = token.lower()
        
        # Check if token is related to sensitive topics
        for topic in topics:
            if topic in self.concept_sensitivity and token_lower in topic:
                return 0.01 * self.concept_sensitivity[topic]
                
        # Default small adjustment
        return 0.001
        
    def _update_warnings(self, state, generated_text):
        """Update warnings based on current state"""
        # Check risk thresholds
        if state["risk_score"] > 0.8 and not any(w["type"] == "high_risk" for w in state["warnings"]):
            state["warnings"].append({
                "type": "high_risk",
                "message": "Content is approaching compliance boundaries",
                "position": len(generated_text)
            })
            
        # Check concept thresholds
        for concept, score in state["concept_scores"].items():
            threshold = state["sensitive_concepts"][concept]["threshold"]
            if score > threshold:
                concept_warning = {
                    "type": "concept_threshold",
                    "concept": concept,
                    "message": f"Content contains sensitive concept: {concept}",
                    "severity": state["sensitive_concepts"][concept]["severity"],
                    "position": len(generated_text)
                }
                
                # Add warning if not already present
                if not any(w["type"] == "concept_threshold" and w["concept"] == concept for w in state["warnings"]):
                    state["warnings"].append(concept_warning)
                    
        return state
        
    def _get_text_fingerprint(self, text):
        """Generate fingerprint of text for efficient comparison"""
        return hashlib.md5(text.encode()).hexdigest()
        
    def _get_state_key(self, text, frameworks):
        """Generate key for state caching"""
        text_prefix = text[:50] if len(text) > 50 else text
        frameworks_str = "-".join(sorted(f.id for f in frameworks))
        return f"{text_prefix}|{frameworks_str}"
        
    def _severity_rank(self, severity):
        """Convert severity to numeric rank"""
        ranks = {"low": 1, "medium": 2, "high": 3}
        return ranks.get(severity, 1)
        
    def _severity_weight(self, severity):
        """Convert severity to weight factor"""
        weights = {"low": 0.5, "medium": 1.0, "high": 2.0}
        return weights.get(severity, 1.0)
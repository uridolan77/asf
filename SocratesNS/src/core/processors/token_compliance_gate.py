import numpy as np
import logging
import re
from dataclasses import dataclass, field
from src.core.utils.utils import TokenConfidenceTracker
from src.core.utils.utils import SensitiveTokenDetector

class TokenLevelComplianceGate:
    """
    Filters token probabilities during generation to ensure compliance with
    regulatory constraints at each generation step.
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.token_blacklist = set(compliance_config.get("token_blacklist", []))
        self.special_token_handling = compliance_config.get("special_token_handling", {})
        self.sensitive_token_detector = SensitiveTokenDetector(compliance_config)
        self.batch_size = compliance_config.get("batch_size", 128)
        self.probability_threshold = compliance_config.get("probability_threshold", 0.01)
        
        # Initialize pattern detectors for faster matching
        self.patterns = self._compile_patterns(compliance_config.get("token_patterns", []))
        
        # Token confidence tracking for uncertainty-aware filtering
        self.confidence_tracker = TokenConfidenceTracker(
            window_size=compliance_config.get("confidence_window", 10)
        )
        
    def filter(self, logits, generated_text, semantic_state, constraints, compliance_mode):
        """
        Filter token logits based on compliance constraints.
        
        Args:
            logits: Token probability logits from language model
            generated_text: Text generated so far
            semantic_state: Current semantic compliance state
            constraints: Applicable compliance constraints
            compliance_mode: Strictness level for filtering
            
        Returns:
            Filtered logits with prohibited tokens masked
        """
        # Clone logits to avoid modifying the original
        filtered_logits = logits.copy()
        
        # Get top tokens for efficient processing (focus on tokens with non-negligible probability)
        top_tokens = self._get_top_tokens(filtered_logits, threshold=self.probability_threshold)
        
        # Filter tokens in batches for efficiency
        for i in range(0, len(top_tokens), self.batch_size):
            batch = top_tokens[i:i+self.batch_size]
            batch_results = self._check_token_batch_compliance(
                batch, generated_text, semantic_state, constraints
            )
            
            # Apply batch results
            for token_id, is_compliant in batch_results.items():
                if not is_compliant:
                    filtered_logits[token_id] = float('-inf')  # Mask prohibited tokens
        
        # Apply special handling for mode-specific adjustments
        self._apply_mode_specific_filtering(filtered_logits, compliance_mode, semantic_state)
        
        # Update token confidence tracking
        self.confidence_tracker.update(filtered_logits, generated_text)
        
        # Ensure we haven't blocked all tokens
        if self._all_tokens_blocked(filtered_logits):
            # Fallback to allow safe continuation
            filtered_logits = self._apply_safe_fallback(logits)
            
        return filtered_logits
    
    def _check_token_batch_compliance(self, token_batch, generated_text, semantic_state, constraints):
        """Check compliance for a batch of tokens in parallel"""
        results = {}
        
        # Check blacklisted tokens (fast path)
        for token_id in token_batch:
            if token_id in self.token_blacklist:
                results[token_id] = False
                continue
        
        # Tokens not immediately rejected need further analysis
        remaining_tokens = [t for t in token_batch if t not in results]
        
        # Check pattern-based constraints
        for token_id in remaining_tokens:
            # Generate hypothetical continuations
            potential_text = generated_text + self._decode_token(token_id)
            
            # Check against pattern constraints
            if self._violates_patterns(potential_text):
                results[token_id] = False
                continue
                
            # Check against semantic constraints by updating hypothetical state
            hypothetical_state = semantic_state.copy()
            hypothetical_state = self._update_semantic_state(
                hypothetical_state,
                token_id,
                potential_text
            )
            
            # Check if hypothetical state violates any constraints
            violates_constraints = False
            for constraint in constraints:
                if self._violates_constraint(hypothetical_state, constraint):
                    violates_constraints = True
                    break
                    
            results[token_id] = not violates_constraints
            
        return results
    
    def _get_top_tokens(self, logits, threshold):
        """Get tokens with probability above threshold"""
        # Convert logits to probabilities
        probs = self._logits_to_probs(logits)
        
        # Find tokens above threshold
        top_tokens = [i for i, p in enumerate(probs) if p >= threshold]
        
        return top_tokens
    
    def _logits_to_probs(self, logits):
        """Convert logits to probabilities using softmax"""
        # Simplified softmax implementation
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    def _violates_patterns(self, text):
        """Check if text violates any compiled patterns"""
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False
    
    def _compile_patterns(self, patterns):
        """Compile regex patterns for efficient matching"""
        compiled_patterns = []
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except Exception as e:
                logging.warning(f"Failed to compile pattern '{pattern}': {str(e)}")
        return compiled_patterns
    
    def _update_semantic_state(self, state, token_id, potential_text):
        """Update semantic state with new token"""
        # This would integrate with semantic monitoring
        # Simplified placeholder implementation
        return state
    
    def _violates_constraint(self, state, constraint):
        """Check if semantic state violates a constraint"""
        # Simplified placeholder implementation
        return False
    
    def _apply_mode_specific_filtering(self, logits, mode, semantic_state):
        """Apply mode-specific filtering adjustments"""
        if mode == "strict":
            # In strict mode, reduce probabilities of uncertain tokens
            uncertain_tokens = self.confidence_tracker.get_uncertain_tokens()
            for token_id in uncertain_tokens:
                logits[token_id] -= 2.0  # Reduce probability significantly
                
        elif mode == "relaxed":
            # In relaxed mode, allow more tokens through
            # Only filter definite violations
            pass
    
    def _all_tokens_blocked(self, logits):
        """Check if all tokens have been blocked"""
        return np.all(np.isinf(logits)) or np.all(logits == float('-inf'))
    
    def _apply_safe_fallback(self, original_logits):
        """Apply fallback strategy when all tokens are blocked"""
        # Create a safe subset of tokens to allow generation to continue
        # This prevents the model from getting stuck
        safe_logits = original_logits.copy()
        
        # Allow only very safe tokens like punctuation and common words
        for i in range(len(safe_logits)):
            if i not in self.get_safe_token_ids():
                safe_logits[i] = float('-inf')
                
        # If still all blocked, allow top-5 safest tokens
        if self._all_tokens_blocked(safe_logits):
            top_5 = np.argsort(original_logits)[-5:]
            safe_logits = np.full_like(original_logits, float('-inf'))
            safe_logits[top_5] = original_logits[top_5]
            
        return safe_logits
    
    def get_safe_token_ids(self):
        """Get a set of inherently safe token IDs"""
        # This would be populated with known safe tokens
        # Simplified implementation returns a small set of token IDs
        return {9, 10, 11, 12, 13}  # Example safe token IDs
    
    def _decode_token(self, token_id):
        """Decode token ID to text (placeholder implementation)"""
        # In a real implementation, this would use the tokenizer
        return f"<token_{token_id}>"
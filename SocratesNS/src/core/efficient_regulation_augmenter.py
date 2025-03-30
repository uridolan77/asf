class EfficientRegulationAugmenter:
    """
    System for augmenting prompts with relevant regulatory context
    to improve compliance during generation.
    """
    def __init__(self, regulatory_retriever, language_model):
        self.retriever = regulatory_retriever
        self.language_model = language_model
        
        # Augmentation configuration
        self.max_regulations = 5
        self.max_tokens_per_regulation = 200
        self.min_relevance_score = 0.65
        
        # Tracking statistics
        self.augmentation_stats = {
            "avg_regulations_used": 0,
            "avg_total_tokens_added": 0,
            "augmentation_count": 0
        }
        
    def augment_prompt(self, prompt, context=None, available_tokens=1000):
        """
        Augment prompt with relevant regulatory content
        
        Args:
            prompt: Original prompt text
            context: Optional context information
            available_tokens: Maximum tokens available for augmentation
            
        Returns:
            Tuple of (augmented_prompt, used_regulations)
        """
        # Skip augmentation if no tokens available
        if available_tokens <= 0:
            return prompt, []
            
        # Analyze prompt to identify regulatory concepts
        relevant_concepts = self._extract_relevant_concepts(prompt)
        
        # Retrieve relevant regulatory content
        retrieved_regulations = self._retrieve_relevant_regulations(
            prompt, context, relevant_concepts
        )
        
        # Filter regulations based on relevance and token budget
        selected_regulations = self._select_regulations(
            retrieved_regulations, 
            available_tokens,
            self.max_regulations
        )
        
        # Format regulations for insertion
        formatted_regulations = self._format_regulations(selected_regulations)
        
        # Create augmented prompt
        augmented_prompt = self._create_augmented_prompt(prompt, formatted_regulations)
        
        # Update statistics
        self._update_augmentation_stats(selected_regulations)
        
        # Return augmented prompt and regulation metadata
        regulation_metadata = [
            {
                "id": reg["document_id"],
                "framework": reg["framework_id"],
                "relevance": reg["score"]
            }
            for reg in selected_regulations
        ]
        
        return augmented_prompt, regulation_metadata
        
    def _extract_relevant_concepts(self, prompt):
        """Extract relevant regulatory concepts from prompt"""
        # This would use a model to extract concepts in a real system
        # Placeholder implementation
        return ["privacy", "data_protection", "consent"]
        
    def _retrieve_relevant_regulations(self, prompt, context, concepts):
        """Retrieve relevant regulatory content"""
        # Use retriever to get relevant regulations
        retrieved_regulations = self.retriever.retrieve(
            prompt, context=context, top_k=self.max_regulations * 2
        )
        
        # Also search by specific concepts
        concept_regulations = []
        for concept in concepts:
            concept_results = self.retriever.retrieve_by_concept(
                concept, top_k=3
            )
            concept_regulations.extend(concept_results)
            
        # Combine results, removing duplicates
        combined_regulations = retrieved_regulations.copy()
        seen_doc_ids = {reg["document_id"] for reg in retrieved_regulations}
        
        for reg in concept_regulations:
            if reg["document_id"] not in seen_doc_ids:
                combined_regulations.append(reg)
                seen_doc_ids.add(reg["document_id"])
                
        # Filter by minimum relevance
        filtered_regulations = [
            reg for reg in combined_regulations
            if reg["score"] >= self.min_relevance_score
        ]
        
        # Sort by relevance
        filtered_regulations.sort(key=lambda x: x["score"], reverse=True)
        
        return filtered_regulations
        
    def _select_regulations(self, regulations, available_tokens, max_count):
        """Select regulations within token budget"""
        selected = []
        tokens_used = 0
        
        for reg in regulations:
            # Estimate token count
            reg_tokens = self._estimate_tokens(reg["excerpt"])
            
            # Truncate if needed
            if reg_tokens > self.max_tokens_per_regulation:
                reg["excerpt"] = self._truncate_text(
                    reg["excerpt"], 
                    self.max_tokens_per_regulation
                )
                reg_tokens = self.max_tokens_per_regulation
                
            # Check if we can add this regulation
            if tokens_used + reg_tokens <= available_tokens and len(selected) < max_count:
                selected.append(reg)
                tokens_used += reg_tokens
            else:
                # Stop if we've reached token limit or max count
                break
                
        return selected
        
    def _format_regulations(self, regulations):
        """Format regulations for insertion into prompt"""
        if not regulations:
            return ""
            
        formatted_text = "\nRELEVANT REGULATORY GUIDELINES:\n"
        
        for i, reg in enumerate(regulations, 1):
            formatted_text += f"[{i}] Framework: {reg['framework_id']}\n"
            formatted_text += f"{reg['excerpt']}\n\n"
            
        return formatted_text
        
    def _create_augmented_prompt(self, prompt, formatted_regulations):
        """Create augmented prompt with regulations"""
        if not formatted_regulations:
            return prompt
            
        # Add regulations before user instructions
        augmented_prompt = formatted_regulations + "\n" + prompt
        
        return augmented_prompt
        
    def _estimate_tokens(self, text):
        """Estimate token count for text"""
        # Simple approximation: words ÷ 0.75
        return int(len(text.split()) / 0.75)
        
    def _truncate_text(self, text, max_tokens):
        """Truncate text to fit within token limit"""
        # Simple approximation: truncate to 0.75 * max_tokens words
        words = text.split()
        max_words = int(max_tokens * 0.75)
        
        if len(words) <= max_words:
            return text
            
        # Truncate to max words
        truncated = " ".join(words[:max_words])
        
        # Add ellipsis
        return truncated + "..."
        
    def _update_augmentation_stats(self, selected_regulations):
        """Update augmentation statistics"""
        # Calculate tokens added
        total_tokens = sum(self._estimate_tokens(reg["excerpt"]) for reg in selected_regulations)
        
        # Update moving averages
        count = self.augmentation_stats["augmentation_count"]
        if count > 0:
            # Update with exponential moving average
            self.augmentation_stats["avg_regulations_used"] = (
                0.9 * self.augmentation_stats["avg_regulations_used"] +
                0.1 * len(selected_regulations)
            )
            self.augmentation_stats["avg_total_tokens_added"] = (
                0.9 * self.augmentation_stats["avg_total_tokens_added"] +
                0.1 * total_tokens
            )
        else:
            # First update
            self.augmentation_stats["avg_regulations_used"] = len(selected_regulations)
            self.augmentation_stats["avg_total_tokens_added"] = total_tokens
            
        # Increment count
        self.augmentation_stats["augmentation_count"] += 1
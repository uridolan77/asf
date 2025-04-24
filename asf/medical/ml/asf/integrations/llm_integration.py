"""
LLM Integration Module

This module provides integration between the ASF framework and Large Language Models.
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class LLMIntegration:
    """
    Integrates ASF with Large Language Models.
    
    This class provides a bridge between the ASF framework and various Large Language Models,
    enabling the use of LLMs for knowledge extraction, contradiction detection, and other tasks.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize LLM integration.
        
        Args:
            model_name: Name of the LLM model (optional)
            device: Device to use (optional)
        """
        self.model_name = model_name
        self.device = device or "cuda" if self._is_cuda_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        self.last_used = 0
        self.total_tokens_processed = 0
        self.total_generations = 0
        self.generation_times = []
    
    async def initialize(self, model_name: Optional[str] = None) -> bool:
        """
        Initialize the LLM.
        
        Args:
            model_name: Name of the LLM model (optional)
            
        Returns:
            Success flag
        """
        if model_name:
            self.model_name = model_name
        
        if not self.model_name:
            return False
        
        try:
            # Import here to avoid dependency if not used
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text or list of generated texts
        """
        if not self.is_initialized:
            return "LLM not initialized"
        
        try:
            # Import here to avoid dependency if not used
            import torch
            
            # Record start time
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_tokens = inputs.input_ids.shape[1]
            
            # Set up generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Add any additional kwargs
            gen_kwargs.update(generation_kwargs)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    **gen_kwargs
                )
            
            # Decode output
            if num_return_sequences == 1:
                decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = decoded_output
            else:
                decoded_outputs = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                result = decoded_outputs
            
            # Record end time
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Update stats
            self.last_used = end_time
            self.total_tokens_processed += input_tokens + (outputs.shape[1] - input_tokens) * num_return_sequences
            self.total_generations += 1
            self.generation_times.append(generation_time)
            
            # Trim generation times list if it gets too long
            if len(self.generation_times) > 100:
                self.generation_times = self.generation_times[-100:]
            
            return result
        except Exception as e:
            return f"Error generating text: {e}"
    
    async def get_embeddings(self, text: Union[str, List[str]]) -> Optional[Any]:
        """
        Get embeddings for text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Text embeddings
        """
        if not self.is_initialized:
            return None
        
        try:
            # Import here to avoid dependency if not used
            import torch
            
            # Handle single text or list of texts
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Tokenize input
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Use last hidden state as embeddings
            # Shape: (batch_size, sequence_length, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]
            
            # Average over sequence length to get a single embedding per text
            # Shape: (batch_size, hidden_size)
            embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Return single embedding or list of embeddings
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None
    
    async def extract_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract knowledge claims from text using the LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted knowledge claims
        """
        if not self.is_initialized:
            return []
        
        try:
            # Create a prompt for knowledge extraction
            prompt = f"""
            Extract factual claims from the following text. For each claim, provide:
            1. The claim text
            2. Confidence level (0-1)
            3. Entities mentioned
            
            Text: {text}
            
            Claims:
            """
            
            # Generate extraction
            extraction = await self.generate(
                prompt,
                max_length=512,
                temperature=0.3,  # Lower temperature for more deterministic output
                top_p=0.9,
                top_k=50
            )
            
            # Parse extraction (this is a simplified parser)
            claims = []
            current_claim = {}
            
            for line in extraction.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                    
                if line.startswith('Claim:') or line.startswith('- Claim:') or line.startswith('1.'):
                    # Save previous claim if it exists
                    if current_claim and 'text' in current_claim:
                        claims.append(current_claim)
                        
                    # Start new claim
                    current_claim = {'text': line.split(':', 1)[1].strip() if ':' in line else line}
                elif line.startswith('Confidence:') or line.startswith('- Confidence:') or line.startswith('2.'):
                    if 'text' in current_claim:
                        try:
                            conf_text = line.split(':', 1)[1].strip() if ':' in line else line
                            # Extract numeric value
                            import re
                            conf_match = re.search(r'(\d+(\.\d+)?)', conf_text)
                            if conf_match:
                                conf_value = float(conf_match.group(1))
                                # Normalize to 0-1 range
                                if conf_value > 1:
                                    conf_value /= 10
                                current_claim['confidence'] = min(1.0, max(0.0, conf_value))
                        except:
                            current_claim['confidence'] = 0.5
                elif line.startswith('Entities:') or line.startswith('- Entities:') or line.startswith('3.'):
                    if 'text' in current_claim:
                        entities_text = line.split(':', 1)[1].strip() if ':' in line else line
                        entities = [e.strip() for e in entities_text.split(',')]
                        current_claim['entities'] = entities
            
            # Add the last claim if it exists
            if current_claim and 'text' in current_claim:
                claims.append(current_claim)
            
            # Add metadata
            for claim in claims:
                claim['source'] = 'llm_extraction'
                claim['timestamp'] = time.time()
                claim['model'] = self.model_name
                
                # Set default confidence if not present
                if 'confidence' not in claim:
                    claim['confidence'] = 0.5
                    
                # Set default entities if not present
                if 'entities' not in claim:
                    claim['entities'] = []
            
            return claims
        except Exception as e:
            print(f"Error extracting knowledge: {e}")
            return []
    
    async def detect_contradiction(
        self, 
        text_a: str, 
        text_b: str
    ) -> Tuple[bool, float]:
        """
        Detect contradiction between two texts using the LLM.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Tuple of (is_contradiction, confidence)
        """
        if not self.is_initialized:
            return False, 0.0
        
        try:
            # Create a prompt for contradiction detection
            prompt = f"""
            Determine if the following two statements contradict each other.
            
            Statement 1: {text_a}
            Statement 2: {text_b}
            
            Do these statements contradict each other? Answer with "Yes" or "No" followed by your confidence level (0-1).
            """
            
            # Generate detection
            detection = await self.generate(
                prompt,
                max_length=100,
                temperature=0.3,  # Lower temperature for more deterministic output
                top_p=0.9,
                top_k=50
            )
            
            # Parse detection
            is_contradiction = "yes" in detection.lower()
            
            # Extract confidence
            import re
            conf_match = re.search(r'(\d+(\.\d+)?)', detection)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            
            # Normalize confidence to 0-1 range
            if confidence > 1:
                confidence /= 10
            confidence = min(1.0, max(0.0, confidence))
            
            return is_contradiction, confidence
        except Exception as e:
            print(f"Error detecting contradiction: {e}")
            return False, 0.0
    
    def _is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            Whether CUDA is available
        """
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM.
        
        Returns:
            Dictionary of model information
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        # Calculate average generation time
        avg_gen_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "model_type": type(self.model).__name__ if self.model else "unknown",
            "total_tokens_processed": self.total_tokens_processed,
            "total_generations": self.total_generations,
            "average_generation_time": avg_gen_time,
            "last_used": self.last_used
        }

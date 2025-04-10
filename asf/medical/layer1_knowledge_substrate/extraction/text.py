# asf/medical/layer1_knowledge_substrate/extraction/text.py
import numpy as np
from transformers import BertModel
from asf.medical.layer1_knowledge_substrate.extraction.base import FeatureExtractor

class TextFeatureExtractor(FeatureExtractor):
    """
    Extracts features from text inputs using BERT.
    """
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = None  # Would initialize tokenizer in real implementation

    def extract_features(self, text_input):
        """Extract features from text input"""
        # Check cache first
        cache_key = self.get_cache_key(text_input)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # In real implementation, would tokenize and pass through BERT
        # For prototype, we'll simulate the output
        embedding_dim = 768  # BERT base embedding dimension
        simulated_embedding = np.random.randn(embedding_dim)

        # Normalize embedding to unit length
        norm = np.linalg.norm(simulated_embedding)
        if norm > 0:
            simulated_embedding = simulated_embedding / norm

        # Create feature dictionary
        features = {
            'text_embedding': simulated_embedding,
            'text_length': len(text_input),
            'word_count': len(text_input.split()),
        }

        # Cache for future use
        self.feature_cache[cache_key] = features

        return features

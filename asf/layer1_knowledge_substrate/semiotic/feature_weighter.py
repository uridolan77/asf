import torch
import torch.nn as nn
import torch.nn.functional as F

class SemioticFeatureWeighter:
    """
    Weights features based on their semiotic significance using transformer attention.
    Philosophical Influence: Saussure's semiotics, Peirce's sign theory
    """
    def __init__(self, feature_dim=768, num_heads=8):
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.context_projection = nn.Linear(feature_dim, feature_dim)
        self.importance_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def extract_attention_weights(self, model_output, input_tokens):
        """
        Extracts attention weights from transformer models like BERT or ViT
        """
        if hasattr(model_output, 'attentions') and model_output.attentions:
            attention_weights = model_output.attentions[-1]
            avg_attention = torch.mean(attention_weights, dim=1)
            return avg_attention
        elif hasattr(model_output, 'attention_weights'):
            return model_output.attention_weights
        return torch.ones((len(input_tokens), len(input_tokens)))
    
    def weight_features(self, features, context_vector=None):
        """
        Applies semiotic weighting to features based on context
        
        Parameters:
        - features: Tensor of feature vectors [num_features, feature_dim]
        - context_vector: Optional context representation
        """
        features_t = features.unsqueeze(1)  # [num_features, 1, feature_dim]
        
        if context_vector is not None:
            context_key = self.context_projection(context_vector).unsqueeze(0)
            
            attn_output, attn_weights = self.attention(
                context_key,
                features_t,
                features_t
            )
            importance_scores = attn_weights.squeeze()
        else:
            attn_output, attn_weights = self.attention(
                features_t,
                features_t,
                features_t
            )
            importance_scores = attn_weights.mean(dim=1).squeeze()
        
        importance_scores = F.softmax(importance_scores, dim=0)
        
        weighted_features = features * importance_scores.unsqueeze(1)
        
        return weighted_features, importance_scores
    
    def extract_key_features(self, feature_dict, top_k=5):
        """
        Identifies most semantically relevant features
        """
        feature_keys = list(feature_dict.keys())
        feature_values = torch.stack([
            torch.tensor(feature_dict[k].value, dtype=torch.float32)
            for k in feature_keys if hasattr(feature_dict[k], 'value')
        ])
        
        _, importance_scores = self.weight_features(feature_values)
        
        _, top_indices = torch.topk(importance_scores, min(top_k, len(feature_keys)))
        
        return [feature_keys[i] for i in top_indices]

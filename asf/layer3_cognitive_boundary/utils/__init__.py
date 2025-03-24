# Layer 3: Semantic Organization Layer
# Utility functions for tensor operations and other helpers

from asf.layer3_cognitive_boundary.utils.tensor_utils import (
    normalize_embeddings, cosine_similarity, batch_cosine_similarity,
    tensor_max_pool, project_to_hyperplane, soft_attention_weighted_sum
)

__all__ = [
    'normalize_embeddings', 'cosine_similarity', 'batch_cosine_similarity',
    'tensor_max_pool', 'project_to_hyperplane', 'soft_attention_weighted_sum'
]

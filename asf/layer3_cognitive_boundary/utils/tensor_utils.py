import torch
import torch.nn.functional as F
import numpy as np

def normalize_embeddings(embeddings, dim=1, eps=1e-8):
    """
    Normalize embeddings to unit length along specified dimension.
    
    Args:
        embeddings: Numpy array or torch tensor of embeddings
        dim: Dimension along which to normalize
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized embeddings
    """
    if isinstance(embeddings, np.ndarray):
        norms = np.linalg.norm(embeddings, axis=dim, keepdims=True)
        norms[norms < eps] = 1.0  # Avoid division by zero
        return embeddings / norms
        
    elif isinstance(embeddings, torch.Tensor):
        return F.normalize(embeddings, p=2, dim=dim, eps=eps)
        
    else:
        raise TypeError("Embeddings must be numpy array or torch tensor")

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between vectors.
    
    Args:
        a: First vector (numpy array or torch tensor)
        b: Second vector (numpy array or torch tensor)
        
    Returns:
        Cosine similarity
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
            
        return np.dot(a, b) / (a_norm * b_norm)
        
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a_normalized = F.normalize(a, p=2, dim=0)
        b_normalized = F.normalize(b, p=2, dim=0)
        
        return torch.dot(a_normalized, b_normalized).item()
        
    else:
        raise TypeError("Vectors must be either both numpy arrays or both torch tensors")

def batch_cosine_similarity(query, database):
    """
    Calculate cosine similarity between a query and all vectors in a database.
    
    Args:
        query: Query vector [D] or batch [B, D]
        database: Database of vectors [N, D]
        
    Returns:
        Similarities [N] or [B, N]
    """
    if isinstance(query, np.ndarray) and isinstance(database, np.ndarray):
        query_norm = normalize_embeddings(query, dim=0 if query.ndim == 1 else 1)
        db_norm = normalize_embeddings(database, dim=1)
        
        if query.ndim == 1:
            return np.dot(db_norm, query_norm)
        else:
            return np.dot(query_norm, db_norm.T)
            
    elif isinstance(query, torch.Tensor) and isinstance(database, torch.Tensor):
        query_norm = F.normalize(query, p=2, dim=0 if query.dim() == 1 else 1)
        db_norm = F.normalize(database, p=2, dim=1)
        
        if query.dim() == 1:
            return torch.matmul(db_norm, query_norm)
        else:
            return torch.matmul(query_norm, db_norm.t())
            
    else:
        raise TypeError("Vectors must be either both numpy arrays or both torch tensors")

def tensor_max_pool(tensor_list, dim=0):
    """
    Apply max pooling across a list of tensors.
    Implements Seth's principle of MGC-based maximization.
    
    Args:
        tensor_list: List of tensors with same shape
        dim: Dimension along which to stack tensors
        
    Returns:
        Tensor with maximum values
    """
    if not tensor_list:
        return None
        
    if isinstance(tensor_list[0], np.ndarray):
        stacked = np.stack(tensor_list, axis=dim)
        return np.max(stacked, axis=dim)
        
    elif isinstance(tensor_list[0], torch.Tensor):
        stacked = torch.stack(tensor_list, dim=dim)
        return torch.max(stacked, dim=dim)[0]
        
    else:
        raise TypeError("Tensors must be either numpy arrays or torch tensors")

def project_to_hyperplane(vectors, normal, bias=0.0):
    """
    Project vectors onto a hyperplane defined by normal and bias.
    Useful for conceptual blending and dimension reduction.
    
    Args:
        vectors: Vectors to project [N, D]
        normal: Normal vector of hyperplane [D]
        bias: Hyperplane bias
        
    Returns:
        Projected vectors [N, D]
    """
    if isinstance(vectors, np.ndarray) and isinstance(normal, np.ndarray):
        normal = normal / np.linalg.norm(normal)
        
        distances = np.dot(vectors, normal) - bias
        projections = vectors - np.outer(distances, normal)
        
        return projections
        
    elif isinstance(vectors, torch.Tensor) and isinstance(normal, torch.Tensor):
        normal = F.normalize(normal, p=2, dim=0)
        
        distances = torch.matmul(vectors, normal) - bias
        projections = vectors - torch.outer(distances, normal)
        
        return projections
        
    else:
        raise TypeError("Vectors must be either both numpy arrays or both torch tensors")

def soft_attention_weighted_sum(query, keys, values, temperature=1.0):
    """
    Calculate attention-weighted sum of values.
    Important for implementing selective attention in Seth's framework.
    
    Args:
        query: Query vector [D]
        keys: Key vectors [N, D]
        values: Value vectors [N, V]
        temperature: Controls softmax temperature
        
    Returns:
        Weighted sum of values [V]
    """
    if isinstance(query, np.ndarray):
        similarities = batch_cosine_similarity(query, keys)
        
        similarities = similarities / temperature
        weights = np.exp(similarities)
        weights = weights / np.sum(weights)
        
        weighted_sum = np.sum(values * weights.reshape(-1, 1), axis=0)
        
        return weighted_sum
        
    elif isinstance(query, torch.Tensor):
        similarities = batch_cosine_similarity(query, keys)
        
        similarities = similarities / temperature
        weights = F.softmax(similarities, dim=0)
        
        weighted_sum = torch.sum(values * weights.unsqueeze(1), dim=0)
        
        return weighted_sum
        
    else:
        raise TypeError("Query must be numpy array or torch tensor")

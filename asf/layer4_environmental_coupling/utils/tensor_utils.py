import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional

def normalize_tensor(tensor, dim=0, eps=1e-8):
    """
    Normalize a tensor along a specific dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized tensor
    """
    if isinstance(tensor, np.ndarray):
        norm = np.linalg.norm(tensor, axis=dim, keepdims=True)
        return tensor / np.maximum(norm, eps)
    elif isinstance(tensor, torch.Tensor):
        return F.normalize(tensor, p=2, dim=dim, eps=eps)
    else:
        raise TypeError("Input must be numpy array or torch tensor")

def cosine_similarity_batch(queries, corpus):
    """
    Calculate cosine similarity between queries and corpus.
    
    Args:
        queries: Tensor of shape [Q, D] for Q queries of dimension D
        corpus: Tensor of shape [C, D] for C corpus items of dimension D
        
    Returns:
        Similarities tensor of shape [Q, C]
    """
    if isinstance(queries, np.ndarray) and isinstance(corpus, np.ndarray):
        # Normalize
        queries_norm = normalize_tensor(queries, dim=1)
        corpus_norm = normalize_tensor(corpus, dim=1)
        
        # Calculate similarity
        return np.matmul(queries_norm, corpus_norm.T)
    
    elif isinstance(queries, torch.Tensor) and isinstance(corpus, torch.Tensor):
        # Normalize
        queries_norm = F.normalize(queries, p=2, dim=1)
        corpus_norm = F.normalize(corpus, p=2, dim=1)
        
        # Calculate similarity
        return torch.matmul(queries_norm, corpus_norm.t())
    
    else:
        raise TypeError("Both inputs must be of the same type (numpy array or torch tensor)")

def sparse_tensor_to_dense(sparse_tensor, shape=None):
    """
    Convert sparse tensor to dense tensor.
    
    Args:
        sparse_tensor: Sparse tensor (either torch.sparse or scipy.sparse)
        shape: Optional shape for the dense tensor
        
    Returns:
        Dense tensor
    """
    if isinstance(sparse_tensor, torch.Tensor) and sparse_tensor.is_sparse:
        return sparse_tensor.to_dense()
    
    elif hasattr(sparse_tensor, 'toarray'):  # scipy.sparse matrices
        return sparse_tensor.toarray()
    
    else:
        raise TypeError("Input must be a sparse tensor")

def batched_matrix_operation(operation, matrices, batch_size=64):
    """
    Apply an operation to a list of matrices in batches.
    
    Args:
        operation: Function to apply to each batch
        matrices: List of matrices
        batch_size: Size of each batch
        
    Returns:
        List of operation results
    """
    results = []
    for i in range(0, len(matrices), batch_size):
        batch = matrices[i:i+batch_size]
        batch_results = operation(batch)
        results.extend(batch_results)
    return results

def adaptive_precision_tensor(tensor, precision='auto'):
    """
    Convert tensor to appropriate precision based on content.
    
    Args:
        tensor: Input tensor
        precision: 'auto', 'float32', 'float16', or 'bfloat16'
        
    Returns:
        Tensor with appropriate precision
    """
    if not isinstance(tensor, torch.Tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        else:
            tensor = torch.tensor(tensor)
    
    # Determine precision
    if precision == 'auto':
        # Check value range
        max_val = torch.max(torch.abs(tensor)).item()
        
        if max_val > 65504 or max_val < 6e-5:
            # Values too large or small for float16
            precision = 'float32'
        else:
            precision = 'float16'
    
    # Convert to specified precision
    if precision == 'float32':
        return tensor.float()
    elif precision == 'float16':
        return tensor.half()
    elif precision == 'bfloat16' and hasattr(torch, 'bfloat16'):
        return tensor.to(torch.bfloat16)
    else:
        return tensor.float()  # Default to float32
def mixed_precision_matmul(a, b, precision='auto'):
    """
    Perform matrix multiplication with mixed precision.
    
    Args:
        a: First tensor
        b: Second tensor
        precision: Precision to use for computation
        
    Returns:
        Result of matrix multiplication
    """
    # Determine if we should use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert inputs to tensors if needed
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device=device)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=device)
    
    # Ensure tensors are on the same device
    a = a.to(device)
    b = b.to(device)
    
    # Apply precision settings
    a_mp = adaptive_precision_tensor(a, precision)
    b_mp = adaptive_precision_tensor(b, precision)
    
    # Perform matrix multiplication
    result = torch.matmul(a_mp, b_mp)
    
    # Convert back to float32 for stability
    return result.float()

def sparse_tensor_add(a, b, inplace=False):
    """
    Add two sparse tensors efficiently.
    
    Args:
        a: First sparse tensor
        b: Second sparse tensor
        inplace: Whether to modify a in-place
        
    Returns:
        Result of addition
    """
    if not a.is_sparse or not b.is_sparse:
        raise ValueError("Both tensors must be sparse")
    
    if inplace:
        # Add values at corresponding indices
        indices_a = a.indices()
        values_a = a.values()
        
        indices_b = b.indices()
        values_b = b.values()
        
        # Handle overlapping indices
        result = torch.sparse_coo_tensor(
            torch.cat([indices_a, indices_b], dim=1),
            torch.cat([values_a, values_b]),
            a.size()
        ).coalesce()
        
        a._indices().copy_(result._indices())
        a._values().copy_(result._values())
        return a
    else:
        # Use built-in addition
        return a + b

def convert_to_csr(sparse_tensor):
    """
    Convert a sparse COO tensor to CSR format for efficient row operations.
    
    Args:
        sparse_tensor: Sparse tensor in COO format
        
    Returns:
        Sparse tensor in CSR format
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Make sure it's coalesced (no duplicate indices)
    sparse_tensor = sparse_tensor.coalesce()
    
    # For PyTorch 1.9+ we can use to_sparse_csr directly
    if hasattr(sparse_tensor, 'to_sparse_csr'):
        return sparse_tensor.to_sparse_csr()
    
    # For older PyTorch versions, we need to convert to scipy and back
    indices = sparse_tensor._indices().cpu().numpy()
    values = sparse_tensor._values().cpu().numpy()
    shape = sparse_tensor.size()
    
    # Convert to scipy CSR
    import scipy.sparse as sp
    scipy_csr = sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)
    
    # Convert back to PyTorch
    indptr = torch.tensor(scipy_csr.indptr, dtype=torch.long)
    indices = torch.tensor(scipy_csr.indices, dtype=torch.long)
    data = torch.tensor(scipy_csr.data, dtype=torch.float)
    
    # Create csr tensor - implementation depends on PyTorch version
    return torch.sparse_coo_tensor(
        torch.stack([indptr[:-1], indices]),
        data,
        shape
    )

def sparse_slice(sparse_tensor, dim, start, end):
    """
    Slice a sparse tensor along a dimension.
    
    Args:
        sparse_tensor: Sparse tensor to slice
        dim: Dimension to slice along
        start: Start index
        end: End index
        
    Returns:
        Sliced sparse tensor
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Get indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    
    # Find indices within the slice range
    mask = (indices[dim] >= start) & (indices[dim] < end)
    
    # Filter indices and values
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]
    
    # Adjust indices for the new tensor
    if start != 0:
        filtered_indices[dim] -= start
    
    # Create new sizes
    new_size = list(sparse_tensor.size())
    new_size[dim] = end - start
    
    # Create and return new sparse tensor
    return torch.sparse_coo_tensor(
        filtered_indices, 
        filtered_values,
        torch.Size(new_size)
    )

def efficient_sparse_matmul(a, b):
    """
    Efficient sparse-sparse or sparse-dense matrix multiplication.
    
    Args:
        a: First tensor (sparse or dense)
        b: Second tensor (sparse or dense)
        
    Returns:
        Result of matrix multiplication
    """
    # Case 1: Both sparse - use special handling
    if a.is_sparse and b.is_sparse:
        # If available, use specialized sparse-sparse mm
        if hasattr(torch.sparse, 'mm'):
            return torch.sparse.mm(a, b)
        # Otherwise, convert one to dense
        else:
            return torch.mm(a.to_dense(), b)
    
    # Case 2: a is sparse, b is dense
    elif a.is_sparse and not b.is_sparse:
        if hasattr(a, 'is_sparse_csr') and a.is_sparse_csr:
            # Use specialized CSR operation if available
            if hasattr(torch.sparse, 'mm'):
                return torch.sparse.mm(a, b)
            else:
                return torch.mm(a.to_dense(), b)
        else:
            # For COO format
            return torch.sparse.mm(a, b)
    
    # Case 3: a is dense, b is sparse
    elif not a.is_sparse and b.is_sparse:
        # Transpose to use efficient sparse mm
        return torch.sparse.mm(b.t(), a.t()).t()
    
    # Case 4: Both dense - use standard mm
    else:
        return torch.mm(a, b)

def tensor_to_device(tensor, device=None):
    """
    Move a tensor to the specified device efficiently.
    
    Args:
        tensor: Input tensor
        device: Target device, or None to use GPU if available
        
    Returns:
        Tensor on the specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # No-op if already on the right device
    if tensor.device == device:
        return tensor
    
    # Handle sparse tensors specially
    if tensor.is_sparse:
        # For sparse tensors, it's more efficient to recreate on the target device
        indices = tensor._indices()
        values = tensor._values()
        
        return torch.sparse_coo_tensor(
            indices.to(device),
            values.to(device),
            tensor.size()
        )
    
    # Standard device transfer for dense tensors
    return tensor.to(device)

def batch_sparse_dense_matmul(sparse_matrices, dense_matrices, batch_size=16):
    """
    Perform batched sparse-dense matrix multiplication efficiently.
    
    Args:
        sparse_matrices: List of sparse matrices
        dense_matrices: List of dense matrices (must be same length as sparse_matrices)
        batch_size: Batch size for processing
        
    Returns:
        List of multiplication results
    """
    if len(sparse_matrices) != len(dense_matrices):
        raise ValueError("Input lists must have the same length")
    
    results = []
    for i in range(0, len(sparse_matrices), batch_size):
        # Get batch
        sparse_batch = sparse_matrices[i:i+batch_size]
        dense_batch = dense_matrices[i:i+batch_size]
        
        # Process batch
        batch_results = []
        for sparse, dense in zip(sparse_batch, dense_batch):
            # Perform multiplication
            if sparse.is_sparse:
                result = torch.sparse.mm(sparse, dense)
            else:
                result = torch.mm(sparse, dense)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results

def sparse_to_scipy(sparse_tensor):
    """
    Convert PyTorch sparse tensor to SciPy sparse matrix.
    
    Args:
        sparse_tensor: PyTorch sparse tensor
        
    Returns:
        SciPy sparse matrix
    """
    import scipy.sparse as sp
    
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Make sure it's coalesced
    sparse_tensor = sparse_tensor.coalesce()
    
    # Get indices and values
    indices = sparse_tensor._indices().cpu().numpy()
    values = sparse_tensor._values().cpu().numpy()
    shape = sparse_tensor.size()
    
    # Convert to scipy COO first
    scipy_coo = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    
    # Convert to CSR for better efficiency
    return scipy_coo.tocsr()

def scipy_to_torch_sparse(scipy_matrix, device=None):
    """
    Convert SciPy sparse matrix to PyTorch sparse tensor.
    
    Args:
        scipy_matrix: SciPy sparse matrix
        device: PyTorch device to place the tensor on
        
    Returns:
        PyTorch sparse tensor
    """
    import scipy.sparse as sp
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to COO if not already
    if not isinstance(scipy_matrix, sp.coo_matrix):
        scipy_matrix = scipy_matrix.tocoo()
    
    # Get indices and values
    indices = torch.tensor(np.vstack((scipy_matrix.row, scipy_matrix.col)), 
                           dtype=torch.long, device=device)
    values = torch.tensor(scipy_matrix.data, dtype=torch.float, device=device)
    shape = torch.Size(scipy_matrix.shape)
    
    # Create sparse tensor
    return torch.sparse_coo_tensor(indices, values, shape)

def save_sparse_tensor(sparse_tensor, file_path):
    """
    Save a sparse tensor to disk efficiently.
    
    Args:
        sparse_tensor: Sparse tensor to save
        file_path: Path to save the tensor to
        
    Returns:
        True if successful
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    # Make sure the tensor is coalesced
    sparse_tensor = sparse_tensor.coalesce()
    
    # Get components
    indices = sparse_tensor._indices().cpu()
    values = sparse_tensor._values().cpu()
    size = sparse_tensor.size()
    
    # Save components
    torch.save({
        'indices': indices,
        'values': values,
        'size': size
    }, file_path)
    
    return True

def load_sparse_tensor(file_path, device=None):
    """
    Load a sparse tensor from disk.
    
    Args:
        file_path: Path to load the tensor from
        device: Device to load the tensor to
        
    Returns:
        Loaded sparse tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load components
    data = torch.load(file_path, map_location=device)
    
    # Create sparse tensor
    return torch.sparse_coo_tensor(
        data['indices'],
        data['values'],
        data['size']
    ).to(device)

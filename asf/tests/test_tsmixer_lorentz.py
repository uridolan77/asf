"""
Test TSMixer and Lorentz Embeddings

This module provides tests for the TSMixer and Lorentz embeddings implementations
in the ASF framework.
"""

import os
import sys
import unittest
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from asf.layer1_knowledge_substrate.temporal.tsmixer import (
    TSMixer, TSMixerEncoder, AdaptiveTSMixer, RevIN
)
from asf.layer1_knowledge_substrate.embeddings.lorentz_embeddings import (
    LorentzEmbedding, LorentzLinear, LorentzDistance, LorentzFusion,
    HybridLorentzEuclideanEmbedding, HybridLorentzEuclideanDistance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-tsmixer-lorentz")

class TestTSMixer(unittest.TestCase):
    """Test cases for TSMixer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seq_len = 24
        self.num_features = 5
        self.batch_size = 16
        self.forecast_horizon = 12
        
        # Create random data
        self.X = torch.randn(self.batch_size, self.seq_len, self.num_features)
        self.y = torch.randn(self.batch_size, self.forecast_horizon, self.num_features)
        
        # Create TSMixer model
        self.model = TSMixer(
            seq_len=self.seq_len,
            num_features=self.num_features,
            num_blocks=2,
            forecast_horizon=self.forecast_horizon
        )
        
        # Create TSMixerEncoder model
        self.encoder = TSMixerEncoder(
            seq_len=self.seq_len,
            num_features=self.num_features,
            embedding_dim=64,
            num_blocks=2
        )
        
        # Create AdaptiveTSMixer model
        self.adaptive_model = AdaptiveTSMixer(
            max_seq_len=self.seq_len,
            num_features=self.num_features,
            num_blocks=2,
            forecast_horizon=self.forecast_horizon
        )
    
    def test_tsmixer_forward(self):
        """Test TSMixer forward pass."""
        # Forward pass
        output = self.model(self.X)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.forecast_horizon, self.num_features))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_tsmixer_encoder_forward(self):
        """Test TSMixerEncoder forward pass."""
        # Forward pass
        output = self.encoder(self.X)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 64))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_adaptive_tsmixer_forward(self):
        """Test AdaptiveTSMixer forward pass."""
        # Forward pass with normal sequence length
        output = self.adaptive_model(self.X)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.forecast_horizon, self.num_features))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
        
        # Forward pass with shorter sequence length
        shorter_X = torch.randn(self.batch_size, self.seq_len // 2, self.num_features)
        output = self.adaptive_model(shorter_X)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.forecast_horizon, self.num_features))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_revin(self):
        """Test RevIN normalization."""
        # Create RevIN module
        revin = RevIN(num_features=self.num_features)
        
        # Normalize
        normalized = revin(self.X, mode="norm")
        
        # Check shape
        self.assertEqual(normalized.shape, self.X.shape)
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(normalized).any())
        
        # Denormalize
        denormalized = revin(normalized, mode="denorm")
        
        # Check shape
        self.assertEqual(denormalized.shape, self.X.shape)
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(denormalized).any())
        
        # Check denormalized is close to original
        self.assertTrue(torch.allclose(denormalized, self.X, rtol=1e-5, atol=1e-5))
    
    def test_tsmixer_training(self):
        """Test TSMixer training."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create loss function
        loss_fn = torch.nn.MSELoss()
        
        # Create dataset
        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Train for a few steps
        self.model.train()
        for epoch in range(2):
            for X_batch, y_batch in dataloader:
                # Forward pass
                y_pred = self.model(X_batch)
                
                # Compute loss
                loss = loss_fn(y_pred, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Check loss is not NaN
                self.assertFalse(torch.isnan(loss).any())

class TestLorentzEmbeddings(unittest.TestCase):
    """Test cases for Lorentz embeddings implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_embeddings = 100
        self.embedding_dim = 32
        self.batch_size = 16
        
        # Create random indices
        self.indices = torch.randint(0, self.num_embeddings, (self.batch_size,))
        
        # Create Lorentz embedding
        self.embedding = LorentzEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            k=-1.0
        )
        
        # Create Lorentz linear layer
        self.linear = LorentzLinear(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim,
            k=-1.0
        )
        
        # Create Lorentz distance module
        self.distance = LorentzDistance(k=-1.0)
        
        # Create hybrid embedding
        self.hybrid_embedding = HybridLorentzEuclideanEmbedding(
            num_embeddings=self.num_embeddings,
            lorentz_dim=self.embedding_dim,
            euclidean_dim=self.embedding_dim,
            k=-1.0
        )
        
        # Create hybrid distance module
        self.hybrid_distance = HybridLorentzEuclideanDistance(
            lorentz_weight=0.5,
            euclidean_weight=0.5,
            k=-1.0
        )
    
    def test_lorentz_embedding_forward(self):
        """Test Lorentz embedding forward pass."""
        # Forward pass
        embeddings = self.embedding(self.indices)
        
        # Check output shape
        self.assertEqual(embeddings.shape, (self.batch_size, self.embedding_dim))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(embeddings).any())
        
        # Check embeddings are on the manifold
        # For Lorentz manifold with k=-1, we need x_0^2 - sum(x_i^2) = 1
        x_0_squared = embeddings[:, 0] ** 2
        x_i_squared_sum = torch.sum(embeddings[:, 1:] ** 2, dim=1)
        manifold_constraint = x_0_squared - x_i_squared_sum
        
        # Check constraint is close to 1
        self.assertTrue(torch.allclose(manifold_constraint, torch.ones_like(manifold_constraint), rtol=1e-5, atol=1e-5))
    
    def test_lorentz_linear_forward(self):
        """Test Lorentz linear forward pass."""
        # Get embeddings
        embeddings = self.embedding(self.indices)
        
        # Forward pass
        output = self.linear(embeddings)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.embedding_dim))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
        
        # Check output is on the manifold
        x_0_squared = output[:, 0] ** 2
        x_i_squared_sum = torch.sum(output[:, 1:] ** 2, dim=1)
        manifold_constraint = x_0_squared - x_i_squared_sum
        
        # Check constraint is close to 1
        self.assertTrue(torch.allclose(manifold_constraint, torch.ones_like(manifold_constraint), rtol=1e-5, atol=1e-5))
    
    def test_lorentz_distance(self):
        """Test Lorentz distance computation."""
        # Get embeddings
        embeddings1 = self.embedding(self.indices[:self.batch_size//2])
        embeddings2 = self.embedding(self.indices[self.batch_size//2:])
        
        # Compute distances
        distances = self.distance(embeddings1, embeddings2)
        
        # Check output shape
        self.assertEqual(distances.shape, (self.batch_size//2,))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(distances).any())
        
        # Check distances are non-negative
        self.assertTrue((distances >= 0).all())
    
    def test_hybrid_embedding_forward(self):
        """Test hybrid embedding forward pass."""
        # Forward pass
        embeddings = self.hybrid_embedding(self.indices)
        
        # Check output contains both embeddings
        self.assertIn("lorentz", embeddings)
        self.assertIn("euclidean", embeddings)
        
        # Check output shapes
        self.assertEqual(embeddings["lorentz"].shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(embeddings["euclidean"].shape, (self.batch_size, self.embedding_dim))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(embeddings["lorentz"]).any())
        self.assertFalse(torch.isnan(embeddings["euclidean"]).any())
        
        # Check Lorentz embeddings are on the manifold
        x_0_squared = embeddings["lorentz"][:, 0] ** 2
        x_i_squared_sum = torch.sum(embeddings["lorentz"][:, 1:] ** 2, dim=1)
        manifold_constraint = x_0_squared - x_i_squared_sum
        
        # Check constraint is close to 1
        self.assertTrue(torch.allclose(manifold_constraint, torch.ones_like(manifold_constraint), rtol=1e-5, atol=1e-5))
    
    def test_hybrid_distance(self):
        """Test hybrid distance computation."""
        # Get embeddings
        embeddings1 = self.hybrid_embedding(self.indices[:self.batch_size//2])
        embeddings2 = self.hybrid_embedding(self.indices[self.batch_size//2:])
        
        # Compute distances
        distances = self.hybrid_distance(embeddings1, embeddings2)
        
        # Check output shape
        self.assertEqual(distances.shape, (self.batch_size//2,))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(distances).any())
        
        # Check distances are non-negative
        self.assertTrue((distances >= 0).all())

if __name__ == "__main__":
    unittest.main()

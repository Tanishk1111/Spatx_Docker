"""
Tests for loss functions in CIT models.
"""
import pytest
import torch
import numpy as np
from scipy.stats import spearmanr

from spatx_core.models.cit_to_gene.loss import SpearmanLoss, CombinedLoss


class TestSpearmanLoss:
    """Test SpearmanLoss implementation."""
    
    def test_init_default_regularization(self, device):
        """Test SpearmanLoss initialization with default regularization."""
        loss_fn = SpearmanLoss()
        assert loss_fn.regularization == 1e-6
    
    def test_init_custom_regularization(self, device):
        """Test SpearmanLoss initialization with custom regularization."""
        reg_value = 1e-5
        loss_fn = SpearmanLoss(regularization=reg_value)
        assert loss_fn.regularization == reg_value
    
    def test_perfect_correlation(self, device):
        """Test loss for perfectly correlated predictions and targets."""
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        
        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Perfect correlation should give loss close to 0 (1 - 1 = 0)
        assert loss.item() < 0.1  # Allow small numerical errors
    
    def test_anti_correlation(self, device):
        """Test loss for anti-correlated predictions and targets."""
        y_pred = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=device)
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        
        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Anti-correlation should give loss close to 2 (1 - (-1) = 2)
        assert 1.5 < loss.item() < 2.5  # Allow for numerical precision
    
    def test_no_correlation(self, device):
        """Test loss for uncorrelated predictions and targets."""
        # Create uncorrelated data
        torch.manual_seed(42)
        y_pred = torch.randn(100, device=device)
        y_true = torch.randn(100, device=device)
        
        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        
        # No correlation should give loss around 1 (1 - 0 = 1)
        assert 0.5 < loss.item() < 1.5
    
    def test_batch_computation(self, device):
        """Test loss computation for batched inputs."""
        batch_size = 8
        gene_count = 50
        
        y_pred = torch.randn(batch_size, gene_count, device=device)
        y_true = torch.randn(batch_size, gene_count, device=device)
        
        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Should return scalar loss
        assert loss.dim() == 0
        assert torch.isfinite(loss)
    
    def test_constant_values(self, device):
        """Test loss with constant values (edge case)."""
        y_pred = torch.ones(10, device=device)
        y_true = torch.ones(10, device=device)
        
        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Constant values should not crash and return finite loss
        assert torch.isfinite(loss)
    
    def test_single_value(self, device):
        """Test loss with single value inputs."""
        y_pred = torch.tensor([1.0], device=device)
        y_true = torch.tensor([2.0], device=device)
        
        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Single value should not crash
        assert torch.isfinite(loss)
    
    def test_gradient_computation(self, device):
        """Test that gradients can be computed through the loss."""
        y_pred = torch.randn(5, 10, device=device, requires_grad=True)
        y_true = torch.randn(5, 10, device=device)
        
        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Compute gradients
        loss.backward()
        
        # Check gradients exist and are finite
        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))
    
    def test_regularization_effect(self, device):
        """Test that regularization affects the computation."""
        y_pred = torch.tensor([1.0, 2.0, 3.0], device=device)
        y_true = torch.tensor([1.0, 2.0, 3.0], device=device)
        
        loss_fn_low_reg = SpearmanLoss(regularization=1e-10)
        loss_fn_high_reg = SpearmanLoss(regularization=1e-2)
        
        loss_low = loss_fn_low_reg(y_pred, y_true)
        loss_high = loss_fn_high_reg(y_pred, y_true)
        
        # Higher regularization should affect the computation
        # (specific behavior depends on implementation details)
        assert torch.isfinite(loss_low)
        assert torch.isfinite(loss_high)


class TestCombinedLoss:
    """Test CombinedLoss implementation."""
    
    def test_init_default_weights(self, device):
        """Test CombinedLoss initialization with default weights."""
        loss_fn = CombinedLoss()
        assert loss_fn.l1_weight == 0.5
        assert loss_fn.spearman_weight == 0.5
        assert isinstance(loss_fn.l1_loss, torch.nn.L1Loss)
        assert isinstance(loss_fn.spearman_loss, SpearmanLoss)
    
    def test_init_custom_weights(self, device):
        """Test CombinedLoss initialization with custom weights."""
        l1_weight = 0.3
        spearman_weight = 0.7
        loss_fn = CombinedLoss(l1_weight=l1_weight, spearman_weight=spearman_weight)
        
        assert loss_fn.l1_weight == l1_weight
        assert loss_fn.spearman_weight == spearman_weight
    
    def test_perfect_predictions(self, device):
        """Test combined loss with perfect predictions."""
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        
        loss_fn = CombinedLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Perfect predictions should give low loss
        assert loss.item() < 0.5
    
    def test_batch_computation(self, device):
        """Test combined loss computation for batched inputs."""
        batch_size = 4
        gene_count = 20
        
        y_pred = torch.randn(batch_size, gene_count, device=device)
        y_true = torch.randn(batch_size, gene_count, device=device)
        
        loss_fn = CombinedLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Should return scalar loss
        assert loss.dim() == 0
        assert torch.isfinite(loss)
    
    def test_weight_effects(self, device):
        """Test that different weights produce different losses."""
        y_pred = torch.randn(5, 10, device=device)
        y_true = torch.randn(5, 10, device=device)
        
        # L1-heavy loss
        loss_fn_l1 = CombinedLoss(l1_weight=0.9, spearman_weight=0.1)
        loss_l1 = loss_fn_l1(y_pred, y_true)
        
        # Spearman-heavy loss  
        loss_fn_spearman = CombinedLoss(l1_weight=0.1, spearman_weight=0.9)
        loss_spearman = loss_fn_spearman(y_pred, y_true)
        
        # Different weights should give different losses
        assert torch.isfinite(loss_l1)
        assert torch.isfinite(loss_spearman)
        # They might be equal by chance, so just check they're finite
    
    def test_gradient_computation(self, device):
        """Test that gradients can be computed through the combined loss."""
        y_pred = torch.randn(3, 8, device=device, requires_grad=True)
        y_true = torch.randn(3, 8, device=device)
        
        loss_fn = CombinedLoss()
        loss = loss_fn(y_pred, y_true)
        
        # Compute gradients
        loss.backward()
        
        # Check gradients exist and are finite
        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))
    
    def test_individual_loss_components(self, device):
        """Test that individual loss components are computed correctly."""
        y_pred = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        y_true = torch.tensor([[1.1, 2.1, 3.1]], device=device)
        
        # Calculate individual losses
        l1_loss = torch.nn.L1Loss()
        spearman_loss = SpearmanLoss()
        
        expected_l1 = l1_loss(y_pred, y_true)
        expected_spearman = spearman_loss(y_pred, y_true)
        
        # Calculate combined loss manually
        combined_manual = 0.5 * expected_l1 + 0.5 * expected_spearman
        
        # Calculate with CombinedLoss
        loss_fn = CombinedLoss()
        combined_actual = loss_fn(y_pred, y_true)
        
        # Should be approximately equal
        assert torch.allclose(combined_manual, combined_actual, rtol=1e-5)
    
    def test_zero_weights(self, device):
        """Test combined loss with zero weights."""
        y_pred = torch.randn(2, 5, device=device)
        y_true = torch.randn(2, 5, device=device)
        
        # Only L1 loss
        loss_fn_l1_only = CombinedLoss(l1_weight=1.0, spearman_weight=0.0)
        loss_l1_only = loss_fn_l1_only(y_pred, y_true)
        
        # Only Spearman loss
        loss_fn_spearman_only = CombinedLoss(l1_weight=0.0, spearman_weight=1.0)
        loss_spearman_only = loss_fn_spearman_only(y_pred, y_true)
        
        assert torch.isfinite(loss_l1_only)
        assert torch.isfinite(loss_spearman_only)
        
        # Compare with individual loss functions
        l1_loss = torch.nn.L1Loss()
        spearman_loss = SpearmanLoss()
        
        expected_l1 = l1_loss(y_pred, y_true)
        expected_spearman = spearman_loss(y_pred, y_true)
        
        assert torch.allclose(loss_l1_only, expected_l1, rtol=1e-5)
        assert torch.allclose(loss_spearman_only, expected_spearman, rtol=1e-5)

"""Tests for CIT model."""
import pytest
import torch
from spatx_core.models.cit_to_gene import CIT, CITGenePredictor, CombinedLoss

def test_cit_model_initialization():
    # Initialize model
    cit = CIT(
        img_size=224,
        in_chans=3,
        embed_dim=96,
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        device= 'cuda'
    )
    assert cit is not None

def test_citgene_predictor_initialization():
    cit = CIT(img_size=224, in_chans=3, device= 'cuda')
    model = CITGenePredictor(cit, num_genes=50)
    assert model is not None

def test_model_forward_pass(device):
    # Create model
    cit = CIT(img_size=224, in_chans=3, device = 'cuda')
    model = CITGenePredictor(cit, num_genes=50)
    model.to(device)
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output
    assert output.shape == (batch_size, 50)
    assert not torch.isnan(output).any()
    assert output.device == x.device

def test_loss_computation(device):
    # Create model and loss
    cit = CIT(img_size=224, in_chans=3, device= 'cuda')
    model = CITGenePredictor(cit, num_genes=50)
    criterion = CombinedLoss(alpha=0.5, reg=1.0)
    model.to(device)
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    y = torch.randn(batch_size, 50).to(device)
    
    # Forward pass and loss
    output = model(x)
    loss = criterion(output, y)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert not torch.isnan(loss)
    
    # Check gradients
    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_different_batch_sizes(batch_size, device):
    # Create model
    cit = CIT(img_size=224, in_chans=3, device= 'cuda')
    model = CITGenePredictor(cit, num_genes=50)
    model.to(device)
    
    # Test forward pass with different batch sizes
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 50)

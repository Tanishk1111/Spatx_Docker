"""
Tests for the complete CITGenePredictor model (CIT backbone + GeneTransformerHead).
This tests the full model as used in training and prediction workflows.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np

from spatx_core.models.cit_to_gene.CiT_Net_T import CIT
from spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor, GeneTransformerHead
from spatx_core.models.cit_to_gene.loss import CombinedLoss


class TestCITGenePredictor:
    """Test the complete CITGenePredictor model."""
    
    def test_init_basic(self, device, dummy123_gene_ids):
        """Test CITGenePredictor initialization with basic parameters."""
        num_genes = len(dummy123_gene_ids)
        
        # Create CIT backbone
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        
        # Create gene predictor
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        
        # Check basic attributes
        assert model.cit is cit_backbone
        assert isinstance(model.head, GeneTransformerHead)
        assert isinstance(model.reg_head, nn.Sequential)
        
        # Check head configuration
        expected_fused_ch = cit_backbone.embed_dim * 8 * 2  # 96 * 8 * 2 = 1536
        assert model.head.mem_proj.in_features == expected_fused_ch
    
    def test_forward_basic(self, device, dummy123_gene_ids):
        """Test basic forward pass of CITGenePredictor."""
        num_genes = len(dummy123_gene_ids)
        batch_size = 2
        
        # Create model
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        
        # Create input
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, num_genes)
        assert torch.isfinite(output).all()
    
    def test_forward_different_batch_sizes(self, device, dummy123_gene_ids):
        """Test forward pass with different batch sizes."""
        num_genes = 50  # Use subset for faster testing
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            output = model(x)
            
            assert output.shape == (batch_size, num_genes)
            assert torch.isfinite(output).all()
    
    def test_forward_different_gene_counts(self, device):
        """Test forward pass with different numbers of genes."""
        batch_size = 2
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        
        # Test different gene counts
        for num_genes in [10, 50, 100, 500]:
            model = CITGenePredictor(cit_backbone, num_genes=num_genes)
            model.to(device)
            
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            output = model(x)
            
            assert output.shape == (batch_size, num_genes)
            assert torch.isfinite(output).all()
    
    def test_gradient_computation(self, device, dummy123_gene_ids):
        """Test gradient computation through the complete model."""
        num_genes = 30  # Use subset for faster testing
        batch_size = 2
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        
        x = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check that both backbone and head parameters have gradients
        backbone_params_with_grad = 0
        head_params_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
                
                if name.startswith('cit.'):
                    backbone_params_with_grad += 1
                elif name.startswith('head.'):
                    head_params_with_grad += 1
        
        assert backbone_params_with_grad > 0, "Backbone should have parameters with gradients"
        assert head_params_with_grad > 0, "Head should have parameters with gradients"
    
    def test_training_mode_consistency(self, device, dummy123_gene_ids):
        """Test model behavior in training vs evaluation mode."""
        num_genes = 25
        batch_size = 1
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Test evaluation mode consistency
        model.eval()
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2, rtol=1e-5), "Model should be deterministic in eval mode"
        
        # Test training mode (should work but may have slight variations due to dropout)
        model.train()
        output3 = model(x)
        output4 = model(x)
        
        # Should produce valid outputs in training mode
        assert torch.isfinite(output3).all()
        assert torch.isfinite(output4).all()
        assert output3.shape == output4.shape == (batch_size, num_genes)
    
    def test_feature_extraction_pipeline(self, device, dummy123_gene_ids):
        """Test the internal feature extraction pipeline."""
        num_genes = 20
        batch_size = 1
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        model.eval()
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Test that we can extract intermediate features
        with torch.no_grad():
            # Extract bottleneck features by calling the CIT backbone directly
            bottleneck = model.cit(x)  # Should return concatenated Cnn4 + Swin4
            
            # Check bottleneck shape
            expected_channels = model.cit.embed_dim * 8 * 2  # 96 * 8 * 2 = 1536
            assert bottleneck.shape == (batch_size, expected_channels, 7, 7)
            assert torch.isfinite(bottleneck).all()
            
            # Test gene head separately
            gene_predictions = model.head(bottleneck)
            assert gene_predictions.shape == (batch_size, num_genes)
            assert torch.isfinite(gene_predictions).all()
            
            # Test that manual pipeline matches full forward pass
            full_output = model(x)
            assert torch.allclose(gene_predictions, full_output, rtol=1e-5)
    
    def test_with_combined_loss(self, device, dummy123_gene_ids):
        """Test CITGenePredictor with CombinedLoss as used in training."""
        num_genes = 40
        batch_size = 4
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        
        # Create loss function as used in training
        criterion = CombinedLoss(alpha=0.5, reg=1.0)
        criterion.to(device)
        
        # Create sample data
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        target = torch.randn(batch_size, num_genes, device=device)
        
        # Forward pass and loss computation
        predictions = model(x)
        loss = criterion(predictions, target)
        
        # Check outputs
        assert predictions.shape == (batch_size, num_genes)
        assert torch.isfinite(predictions).all()
        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss)
        
        # Test backpropagation
        loss.backward()
        
        # Check that gradients exist
        param_count = 0
        params_with_grad = 0
        for param in model.parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all()
                    params_with_grad += 1
        
        assert param_count > 0, "Model should have trainable parameters"
        assert params_with_grad > 0, "At least some parameters should have gradients"
    
    def test_parameter_sharing(self, device, dummy123_gene_ids):
        """Test that the same CIT backbone can be shared across different gene predictors."""
        num_genes1, num_genes2 = 20, 30
        
        # Shared backbone
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        
        # Two different gene predictors with same backbone
        model1 = CITGenePredictor(cit_backbone, num_genes=num_genes1)
        model2 = CITGenePredictor(cit_backbone, num_genes=num_genes2)
        
        # Check that they share the same backbone
        assert model1.cit is model2.cit is cit_backbone
        
        # Check that heads are different
        assert model1.head is not model2.head
        assert model1.head.query_embed.shape[0] == num_genes1
        assert model2.head.query_embed.shape[0] == num_genes2
    
    def test_device_consistency(self, device, dummy123_gene_ids):
        """Test that model operations stay on correct device."""
        num_genes = 15
        batch_size = 2
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        output = model(x)
        
        # Check output device
        assert output.device.type == device if isinstance(device, str) else output.device == device
        
        # Check that all parameters are on correct device
        for param in model.parameters():
            if isinstance(device, str):
                assert param.device.type == device or str(param.device) == device
            else:
                assert param.device == device
    
    def test_model_state_dict_save_load(self, device, dummy123_gene_ids):
        """Test saving and loading model state dict."""
        num_genes = 35
        batch_size = 1
        
        # Create original model
        cit_backbone1 = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model1 = CITGenePredictor(cit_backbone1, num_genes=num_genes)
        model1.to(device)
        model1.eval()
        
        # Create input
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Get original output
        with torch.no_grad():
            output1 = model1(x)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model with same architecture
        cit_backbone2 = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model2 = CITGenePredictor(cit_backbone2, num_genes=num_genes)
        model2.to(device)
        
        # Load state dict
        model2.load_state_dict(state_dict)
        model2.eval()
        
        # Get output from loaded model
        with torch.no_grad():
            output2 = model2(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-6)
    
    def test_realistic_training_scenario(self, device, dummy123_gene_ids):
        """Test a realistic training scenario with multiple epochs."""
        num_genes = 25
        batch_size = 4
        num_epochs = 3
        
        # Create model and optimizer
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = CombinedLoss(alpha=0.5, reg=1.0)
        criterion.to(device)
        
        # Track losses
        losses = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            # Simulate multiple batches per epoch
            for batch_idx in range(3):
                # Create random data
                x = torch.randn(batch_size, 3, 224, 224, device=device)
                target = torch.randn(batch_size, num_genes, device=device)
                
                # Forward pass
                predictions = model(x)
                loss = criterion(predictions, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Check that outputs are valid
                assert torch.isfinite(predictions).all()
                assert torch.isfinite(loss)
            
            losses.append(epoch_loss)
        
        # Check that training completed successfully
        assert len(losses) == num_epochs
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    
    def test_inference_scenario(self, device, dummy123_gene_ids):
        """Test a realistic inference scenario as used in prediction."""
        num_genes = len(dummy123_gene_ids[:50])  # Use subset
        batch_size = 8
        
        # Create model
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        model.eval()
        
        predictions_list = []
        
        # Simulate inference on multiple batches
        with torch.no_grad():
            for batch_idx in range(5):  # 5 batches
                x = torch.randn(batch_size, 3, 224, 224, device=device)
                predictions = model(x)
                
                # Check predictions
                assert predictions.shape == (batch_size, num_genes)
                assert torch.isfinite(predictions).all()
                
                predictions_list.append(predictions.cpu().numpy())
        
        # Concatenate all predictions
        all_predictions = np.concatenate(predictions_list, axis=0)
        
        # Check final result
        expected_samples = 5 * batch_size
        assert all_predictions.shape == (expected_samples, num_genes)
        assert np.isfinite(all_predictions).all()
    
    def test_memory_efficiency(self, device, dummy123_gene_ids):
        """Test that model doesn't cause memory leaks during repeated inference."""
        if isinstance(device, str) and device == 'cpu':
            pytest.skip("Memory test only relevant for CUDA")
        
        num_genes = 30
        batch_size = 4
        
        cit_backbone = CIT(
            device=str(device),
            img_size=224, in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        model = CITGenePredictor(cit_backbone, num_genes=num_genes)
        model.to(device)
        model.eval()
        
        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Run multiple inference iterations
        with torch.no_grad():
            for i in range(10):
                x = torch.randn(batch_size, 3, 224, 224, device=device)
                output = model(x)
                
                # Clean up
                del x, output
                if i % 3 == 0:
                    torch.cuda.empty_cache()
        
        # Check final memory
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(device)
        
        # Memory usage should not grow significantly
        memory_growth = final_memory - initial_memory
        max_allowed_growth = 100 * 1024 * 1024  # 100MB
        
        assert memory_growth < max_allowed_growth, f"Memory grew by {memory_growth / (1024*1024):.1f}MB"

"""
Tests for CiTGene model (Gene prediction head with CIT backbone).
"""
import pytest
import torch
import torch.nn as nn

from spatx_core.models.cit_to_gene.CiTGene import CiTGene, GeneTransformerHead
from spatx_core.models.cit_to_gene.CiT_Net_T import CiT_Net_T


class TestGeneTransformerHead:
    """Test GeneTransformerHead implementation."""
    
    def test_init_default(self, device):
        """Test GeneTransformerHead initialization with default parameters."""
        num_genes = 100
        embed_dim = 768
        
        head = GeneTransformerHead(num_genes, embed_dim)
        
        assert head.num_genes == num_genes
        assert head.embed_dim == embed_dim
        assert head.num_heads == 8  # Default
        assert isinstance(head.gene_queries, nn.Parameter)
        assert head.gene_queries.shape == (1, num_genes, embed_dim)
        assert isinstance(head.multihead_attn, nn.MultiheadAttention)
        assert isinstance(head.layer_norm, nn.LayerNorm)
        assert isinstance(head.gene_mlp, nn.Sequential)
    
    def test_init_custom(self, device):
        """Test GeneTransformerHead initialization with custom parameters."""
        num_genes = 50
        embed_dim = 384
        num_heads = 6
        
        head = GeneTransformerHead(num_genes, embed_dim, num_heads=num_heads)
        
        assert head.num_genes == num_genes
        assert head.embed_dim == embed_dim
        assert head.num_heads == num_heads
        assert head.gene_queries.shape == (1, num_genes, embed_dim)
    
    def test_forward(self, device):
        """Test GeneTransformerHead forward pass."""
        batch_size = 2
        num_patches = 196  # 14x14 patches
        embed_dim = 768
        num_genes = 100
        
        head = GeneTransformerHead(num_genes, embed_dim)
        head.to(device)
        
        # Spatial features from transformer backbone
        spatial_features = torch.randn(batch_size, num_patches, embed_dim, device=device)
        output = head(spatial_features)
        
        assert output.shape == (batch_size, num_genes)
        assert torch.isfinite(output).all()
    
    def test_forward_different_sizes(self, device):
        """Test GeneTransformerHead with different input sizes."""
        embed_dim = 384
        num_genes = 200
        head = GeneTransformerHead(num_genes, embed_dim, num_heads=6)
        head.to(device)
        
        # Test different numbers of spatial patches
        for num_patches in [49, 196, 784]:  # 7x7, 14x14, 28x28
            spatial_features = torch.randn(1, num_patches, embed_dim, device=device)
            output = head(spatial_features)
            
            assert output.shape == (1, num_genes)
            assert torch.isfinite(output).all()
    
    def test_cross_attention_mechanism(self, device):
        """Test that cross-attention mechanism works correctly."""
        batch_size = 1
        num_patches = 49
        embed_dim = 384
        num_genes = 10
        
        head = GeneTransformerHead(num_genes, embed_dim, num_heads=6)
        head.to(device)
        
        # Create spatial features with distinct patterns
        spatial_features = torch.randn(batch_size, num_patches, embed_dim, device=device)
        
        # Forward pass
        output = head(spatial_features)
        
        assert output.shape == (batch_size, num_genes)
        
        # Check that output changes with different spatial features
        spatial_features2 = torch.randn(batch_size, num_patches, embed_dim, device=device)
        output2 = head(spatial_features2)
        
        # Outputs should be different for different inputs
        assert not torch.allclose(output, output2, rtol=1e-3)
    
    def test_gradient_computation(self, device):
        """Test gradient computation through GeneTransformerHead."""
        num_genes = 50
        embed_dim = 384
        head = GeneTransformerHead(num_genes, embed_dim)
        head.to(device)
        
        spatial_features = torch.randn(2, 196, embed_dim, device=device, requires_grad=True)
        output = head(spatial_features)
        loss = output.sum()
        
        loss.backward()
        
        # Check input gradients
        assert spatial_features.grad is not None
        assert torch.isfinite(spatial_features.grad).all()
        
        # Check parameter gradients
        for name, param in head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    
    def test_gene_queries_learnable(self, device):
        """Test that gene queries are learnable parameters."""
        num_genes = 20
        embed_dim = 192
        head = GeneTransformerHead(num_genes, embed_dim)
        head.to(device)
        
        # Gene queries should require gradients
        assert head.gene_queries.requires_grad
        
        # Test that they get gradients during training
        spatial_features = torch.randn(1, 49, embed_dim, device=device)
        output = head(spatial_features)
        loss = output.sum()
        loss.backward()
        
        assert head.gene_queries.grad is not None
        assert torch.isfinite(head.gene_queries.grad).all()


class TestCiTGene:
    """Test CiTGene (full model) implementation."""
    
    def test_init_default(self, device, dummy123_gene_ids):
        """Test CiTGene initialization with default parameters."""
        gene_ids = dummy123_gene_ids
        model = CiTGene(gene_ids)
        
        assert model.num_genes == len(gene_ids)
        assert isinstance(model.backbone, CiT_Net_T)
        assert isinstance(model.gene_head, GeneTransformerHead)
        assert model.gene_head.num_genes == len(gene_ids)
    
    def test_init_custom(self, device, dummy123_gene_ids):
        """Test CiTGene initialization with custom parameters."""
        gene_ids = dummy123_gene_ids[:50]  # Use subset
        img_size = 112
        patch_size = 8
        embed_dim = 384
        depth = 6
        num_heads = 6
        
        model = CiTGene(
            gene_ids, img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )
        
        assert model.num_genes == len(gene_ids)
        assert model.backbone.patch_embed.img_size == (img_size, img_size)
        assert model.backbone.patch_embed.patch_size == (patch_size, patch_size)
        assert model.gene_head.embed_dim == embed_dim
        assert model.gene_head.num_heads == num_heads
    
    def test_forward(self, device, dummy123_gene_ids):
        """Test CiTGene forward pass."""
        gene_ids = dummy123_gene_ids
        batch_size = 2
        
        model = CiTGene(gene_ids, img_size=224, embed_dim=768)
        model.to(device)
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        output = model(x)
        
        assert output.shape == (batch_size, len(gene_ids))
        assert torch.isfinite(output).all()
    
    def test_forward_different_sizes(self, device, dummy123_gene_ids):
        """Test CiTGene with different input sizes."""
        gene_ids = dummy123_gene_ids[:100]  # Use subset for faster testing
        
        model = CiTGene(gene_ids, img_size=112, patch_size=8, embed_dim=384, depth=4)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device)
        output = model(x)
        
        assert output.shape == (1, len(gene_ids))
        assert torch.isfinite(output).all()
    
    def test_feature_extraction_pipeline(self, device, dummy123_gene_ids):
        """Test the full feature extraction pipeline."""
        gene_ids = dummy123_gene_ids[:50]
        model = CiTGene(gene_ids, img_size=112, embed_dim=384, depth=3)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device)
        
        # Extract spatial features from backbone
        spatial_features = model.extract_features(x)
        expected_patches = (112 // 16) ** 2  # Default patch size is 16
        assert spatial_features.shape == (1, expected_patches, 384)
        
        # Forward through gene head
        gene_predictions = model.gene_head(spatial_features)
        assert gene_predictions.shape == (1, len(gene_ids))
        
        # Full forward pass should match
        full_output = model(x)
        assert torch.allclose(full_output, gene_predictions, rtol=1e-5)
    
    def test_extract_features(self, device, dummy123_gene_ids):
        """Test extract_features method."""
        gene_ids = dummy123_gene_ids[:20]
        model = CiTGene(gene_ids, img_size=224, embed_dim=768)
        model.to(device)
        
        x = torch.randn(2, 3, 224, 224, device=device)
        features = model.extract_features(x)
        
        expected_patches = (224 // 16) ** 2  # Default patch size
        assert features.shape == (2, expected_patches, 768)
        assert torch.isfinite(features).all()
    
    def test_gradient_computation(self, device, dummy123_gene_ids):
        """Test gradient computation through full CiTGene model."""
        gene_ids = dummy123_gene_ids[:30]  # Use subset for faster testing
        model = CiTGene(gene_ids, img_size=112, embed_dim=192, depth=2)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check that both backbone and gene head have gradients
        backbone_has_grads = any(
            param.grad is not None and param.requires_grad
            for param in model.backbone.parameters()
        )
        gene_head_has_grads = any(
            param.grad is not None and param.requires_grad
            for param in model.gene_head.parameters()
        )
        
        assert backbone_has_grads, "Backbone should have gradients"
        assert gene_head_has_grads, "Gene head should have gradients"
    
    def test_eval_mode(self, device, dummy123_gene_ids):
        """Test CiTGene in evaluation mode."""
        gene_ids = dummy123_gene_ids[:25]
        model = CiTGene(gene_ids, img_size=224, embed_dim=384)
        model.to(device)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224, device=device)
        
        # Multiple forward passes should give same results in eval mode
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2, rtol=1e-5)
    
    def test_gene_prediction_consistency(self, device, dummy123_gene_ids):
        """Test that gene predictions are consistent across forward passes."""
        gene_ids = dummy123_gene_ids[:15]
        model = CiTGene(gene_ids, img_size=112, embed_dim=256, depth=2)
        model.to(device)
        model.eval()
        
        # Same input should give same output
        x = torch.randn(1, 3, 112, 112, device=device)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2, rtol=1e-6)
        
        # Different inputs should give different outputs
        x2 = torch.randn(1, 3, 112, 112, device=device)
        with torch.no_grad():
            output3 = model(x2)
        
        assert not torch.allclose(output1, output3, rtol=1e-3)
    
    def test_parameter_count(self, device, dummy123_gene_ids):
        """Test that model has reasonable number of parameters."""
        gene_ids = dummy123_gene_ids
        model = CiTGene(gene_ids, img_size=224, embed_dim=384, depth=6)
        
        total_params = sum(p.numel() for p in model.parameters())
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        gene_head_params = sum(p.numel() for p in model.gene_head.parameters())
        
        # Basic sanity checks
        assert total_params > 10000  # At least 10K parameters
        assert total_params < 200_000_000  # Less than 200M parameters
        assert backbone_params + gene_head_params == total_params
        assert gene_head_params > 0  # Gene head should have parameters
    
    def test_device_consistency(self, device, dummy123_gene_ids):
        """Test that all model operations stay on correct device."""
        gene_ids = dummy123_gene_ids[:40]
        model = CiTGene(gene_ids, img_size=112, embed_dim=256)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device)
        output = model(x)
        
        assert output.device == device
        
        # Check that all parameters are on correct device
        for param in model.parameters():
            assert param.device == device
    
    def test_gene_ordering_consistency(self, device):
        """Test that gene ordering in output matches input gene_ids."""
        # Use a specific set of gene IDs to test ordering
        gene_ids = ['GENE_A', 'GENE_B', 'GENE_C', 'GENE_D', 'GENE_E']
        model = CiTGene(gene_ids, img_size=112, embed_dim=128, depth=1)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device)
        output = model(x)
        
        # Output should have same number of dimensions as genes
        assert output.shape == (1, len(gene_ids))
        
        # Model should maintain the gene ordering internally
        assert model.num_genes == len(gene_ids)

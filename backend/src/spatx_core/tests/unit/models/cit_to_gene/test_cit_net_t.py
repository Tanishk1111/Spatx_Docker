"""
Tests for CiT_Net_T (Compact Image Transformer) backbone model.
"""
import pytest
import torch
import torch.nn as nn

from spatx_core.models.cit_to_gene.CiT_Net_T import CiT_Net_T, PatchEmbed, DropPath, Mlp, CiTBlock, CiTAttention


class TestPatchEmbed:
    """Test PatchEmbed module implementation."""
    
    def test_init_default(self, device):
        """Test PatchEmbed initialization with default parameters."""
        patch_embed = PatchEmbed()
        
        assert patch_embed.img_size == (224, 224)
        assert patch_embed.patch_size == (16, 16)
        assert patch_embed.num_patches == 196  # (224//16) * (224//16)
        assert patch_embed.in_chans == 3
        assert patch_embed.embed_dim == 768
        assert isinstance(patch_embed.proj, nn.Conv2d)
    
    def test_init_custom(self, device):
        """Test PatchEmbed initialization with custom parameters."""
        img_size = 112
        patch_size = 8
        in_chans = 1
        embed_dim = 384
        
        patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        
        expected_patches = (img_size // patch_size) ** 2
        assert patch_embed.num_patches == expected_patches
        assert patch_embed.in_chans == in_chans
        assert patch_embed.embed_dim == embed_dim
    
    def test_forward(self, device):
        """Test PatchEmbed forward pass."""
        batch_size = 2
        patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
        patch_embed.to(device)
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        output = patch_embed(x)
        
        # Output should be (batch_size, num_patches, embed_dim)
        expected_patches = (224 // 16) ** 2
        assert output.shape == (batch_size, expected_patches, 768)
        assert torch.isfinite(output).all()
    
    def test_forward_different_sizes(self, device):
        """Test PatchEmbed with different input sizes."""
        patch_embed = PatchEmbed(img_size=112, patch_size=8, embed_dim=384)
        patch_embed.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device)
        output = patch_embed(x)
        
        expected_patches = (112 // 8) ** 2
        assert output.shape == (1, expected_patches, 384)
        assert torch.isfinite(output).all()


class TestDropPath:
    """Test DropPath module implementation."""
    
    def test_init(self, device):
        """Test DropPath initialization."""
        drop_prob = 0.1
        drop_path = DropPath(drop_prob)
        assert drop_path.drop_prob == drop_prob
    
    def test_forward_training(self, device):
        """Test DropPath forward pass in training mode."""
        drop_path = DropPath(0.2)
        drop_path.to(device)
        drop_path.train()
        
        x = torch.randn(4, 10, device=device)
        output = drop_path(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_forward_eval(self, device):
        """Test DropPath forward pass in eval mode."""
        drop_path = DropPath(0.5)
        drop_path.to(device)
        drop_path.eval()
        
        x = torch.randn(4, 10, device=device)
        output = drop_path(x)
        
        # In eval mode, should be identity
        assert torch.allclose(output, x)
    
    def test_zero_drop_prob(self, device):
        """Test DropPath with zero drop probability."""
        drop_path = DropPath(0.0)
        drop_path.to(device)
        drop_path.train()
        
        x = torch.randn(2, 5, device=device)
        output = drop_path(x)
        
        # Should be identity with zero drop probability
        assert torch.allclose(output, x)


class TestMlp:
    """Test Mlp (Multi-layer Perceptron) module implementation."""
    
    def test_init_default(self, device):
        """Test Mlp initialization with default parameters."""
        in_features = 768
        mlp = Mlp(in_features)
        
        assert isinstance(mlp.fc1, nn.Linear)
        assert isinstance(mlp.fc2, nn.Linear)
        assert isinstance(mlp.act, nn.GELU)
        assert isinstance(mlp.drop, nn.Dropout)
        
        assert mlp.fc1.in_features == in_features
        assert mlp.fc1.out_features == in_features * 4  # Default ratio
        assert mlp.fc2.out_features == in_features
    
    def test_init_custom(self, device):
        """Test Mlp initialization with custom parameters."""
        in_features = 384
        hidden_features = 512
        out_features = 256
        drop = 0.2
        
        mlp = Mlp(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            drop=drop
        )
        
        assert mlp.fc1.in_features == in_features
        assert mlp.fc1.out_features == hidden_features
        assert mlp.fc2.in_features == hidden_features
        assert mlp.fc2.out_features == out_features
        assert mlp.drop.p == drop
    
    def test_forward(self, device):
        """Test Mlp forward pass."""
        batch_size, seq_len, in_features = 2, 196, 768
        mlp = Mlp(in_features)
        mlp.to(device)
        
        x = torch.randn(batch_size, seq_len, in_features, device=device)
        output = mlp(x)
        
        assert output.shape == (batch_size, seq_len, in_features)
        assert torch.isfinite(output).all()


class TestCiTAttention:
    """Test CiTAttention module implementation."""
    
    def test_init_default(self, device):
        """Test CiTAttention initialization with default parameters."""
        dim = 768
        attention = CiTAttention(dim)
        
        assert attention.num_heads == 8  # Default
        assert attention.head_dim == dim // 8
        assert attention.scale == (dim // 8) ** -0.5
        assert isinstance(attention.qkv, nn.Linear)
        assert isinstance(attention.proj, nn.Linear)
    
    def test_init_custom(self, device):
        """Test CiTAttention initialization with custom parameters."""
        dim = 384
        num_heads = 6
        qkv_bias = True
        qk_scale = 0.1
        attn_drop = 0.1
        proj_drop = 0.2
        
        attention = CiTAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop
        )
        
        assert attention.num_heads == num_heads
        assert attention.head_dim == dim // num_heads
        assert attention.scale == qk_scale
        assert attention.attn_drop.p == attn_drop
        assert attention.proj_drop.p == proj_drop
    
    def test_forward(self, device):
        """Test CiTAttention forward pass."""
        batch_size, seq_len, dim = 2, 196, 768
        attention = CiTAttention(dim)
        attention.to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert torch.isfinite(output).all()
    
    def test_different_sequence_lengths(self, device):
        """Test CiTAttention with different sequence lengths."""
        dim = 384
        attention = CiTAttention(dim, num_heads=6)
        attention.to(device)
        
        for seq_len in [49, 196, 784]:  # 7x7, 14x14, 28x28
            x = torch.randn(1, seq_len, dim, device=device)
            output = attention(x)
            
            assert output.shape == (1, seq_len, dim)
            assert torch.isfinite(output).all()


class TestCiTBlock:
    """Test CiTBlock (Transformer Block) implementation."""
    
    def test_init_default(self, device):
        """Test CiTBlock initialization with default parameters."""
        dim = 768
        block = CiTBlock(dim)
        
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.attn, CiTAttention)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert isinstance(block.mlp, Mlp)
        assert isinstance(block.drop_path, DropPath)
    
    def test_init_custom(self, device):
        """Test CiTBlock initialization with custom parameters."""
        dim = 384
        num_heads = 6
        mlp_ratio = 3.0
        drop = 0.1
        attn_drop = 0.1
        drop_path = 0.2
        
        block = CiTBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )
        
        assert block.attn.num_heads == num_heads
        assert block.mlp.fc1.out_features == int(dim * mlp_ratio)
        assert block.drop_path.drop_prob == drop_path
    
    def test_forward(self, device):
        """Test CiTBlock forward pass."""
        batch_size, seq_len, dim = 2, 196, 768
        block = CiTBlock(dim)
        block.to(device)
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert torch.isfinite(output).all()
    
    def test_residual_connections(self, device):
        """Test that residual connections work properly."""
        dim = 384
        block = CiTBlock(dim, drop_path=0.0)  # No drop path for exact residual
        block.to(device)
        block.eval()  # Eval mode to avoid randomness
        
        x = torch.randn(1, 49, dim, device=device)
        
        # The output should be different from input due to transformations
        output = block(x)
        assert not torch.allclose(output, x, rtol=1e-3)  # Should be transformed
        assert output.shape == x.shape


class TestCiTNetT:
    """Test CiT_Net_T (full model) implementation."""
    
    def test_init_default(self, device):
        """Test CiT_Net_T initialization with default parameters."""
        model = CiT_Net_T()
        
        assert model.num_classes == 1000
        assert len(model.blocks) == 12  # Default depth
        assert isinstance(model.patch_embed, PatchEmbed)
        assert isinstance(model.norm, nn.LayerNorm)
        assert model.head.out_features == 1000
    
    def test_init_custom(self, device):
        """Test CiT_Net_T initialization with custom parameters."""
        img_size = 112
        patch_size = 8
        num_classes = 500
        embed_dim = 384
        depth = 6
        num_heads = 6
        
        model = CiT_Net_T(
            img_size=img_size, patch_size=patch_size,
            num_classes=num_classes, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads
        )
        
        assert model.num_classes == num_classes
        assert len(model.blocks) == depth
        assert model.head.out_features == num_classes
        assert model.patch_embed.embed_dim == embed_dim
    
    def test_forward(self, device):
        """Test CiT_Net_T forward pass."""
        batch_size = 2
        model = CiT_Net_T(img_size=224, num_classes=1000)
        model.to(device)
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        output = model(x)
        
        assert output.shape == (batch_size, 1000)
        assert torch.isfinite(output).all()
    
    def test_forward_different_sizes(self, device):
        """Test CiT_Net_T with different input and output sizes."""
        model = CiT_Net_T(img_size=112, patch_size=8, num_classes=200, embed_dim=384)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device)
        output = model(x)
        
        assert output.shape == (1, 200)
        assert torch.isfinite(output).all()
    
    def test_feature_extraction(self, device):
        """Test CiT_Net_T feature extraction (without final head)."""
        model = CiT_Net_T(embed_dim=384, depth=6)
        model.to(device)
        
        x = torch.randn(2, 3, 224, 224, device=device)
        
        # Forward through patch embedding
        features = model.patch_embed(x)
        assert features.shape == (2, 196, 384)
        
        # Add positional embedding
        features = features + model.pos_embed
        features = model.pos_drop(features)
        
        # Forward through transformer blocks
        for block in model.blocks:
            features = block(features)
        
        # Final norm
        features = model.norm(features)
        
        assert features.shape == (2, 196, 384)
        assert torch.isfinite(features).all()
    
    def test_gradient_computation(self, device):
        """Test gradient computation through CiT_Net_T."""
        model = CiT_Net_T(img_size=112, num_classes=10, embed_dim=192, depth=3)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check model parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    
    def test_eval_mode(self, device):
        """Test CiT_Net_T in evaluation mode."""
        model = CiT_Net_T(img_size=224, num_classes=100)
        model.to(device)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224, device=device)
        
        # Multiple forward passes should give same results in eval mode
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2, rtol=1e-5)
    
    def test_parameter_count(self, device):
        """Test that model has reasonable number of parameters."""
        model = CiT_Net_T(embed_dim=384, depth=6, num_heads=6)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have parameters (not exact, but reasonable bounds)
        assert total_params > 1000  # At least 1K parameters
        assert total_params < 100_000_000  # Less than 100M parameters
        assert trainable_params == total_params  # All should be trainable by default
    
    def test_device_consistency(self, device):
        """Test that all model operations stay on correct device."""
        model = CiT_Net_T(img_size=112, embed_dim=192)
        model.to(device)
        
        x = torch.randn(1, 3, 112, 112, device=device)
        output = model(x)
        
        assert output.device == device
        
        # Check that all parameters are on correct device
        for param in model.parameters():
            assert param.device == device

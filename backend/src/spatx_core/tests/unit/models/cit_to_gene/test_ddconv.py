"""
Tests for DDConv (Dynamic Deformable Convolution) module.
"""
import pytest
import torch
import torch.nn as nn

from spatx_core.models.cit_to_gene.DDConv import DDConv, SConv2D, _routing


class TestDDConv:
    """Test DDConv (Dynamic Deformable Convolution) implementation."""
    
    def test_init_basic(self, device):
        """Test DDConv initialization with basic parameters."""
        inc, outc = 64, 128
        ddconv = DDConv(inc, outc, device=str(device))
        
        assert ddconv.device == str(device)
        assert ddconv.kernel_size == 3
        assert ddconv.padding == 1
        assert ddconv.stride == 1
        assert ddconv.modulation == False
        assert isinstance(ddconv.conv, SConv2D)
        assert isinstance(ddconv.p_conv, SConv2D)
        assert ddconv.m_conv is None  # No modulation by default
    
    def test_init_with_modulation(self, device):
        """Test DDConv initialization with modulation enabled."""
        inc, outc = 32, 64
        ddconv = DDConv(inc, outc, device=str(device), modulation=True)
        
        assert ddconv.modulation == True
        assert ddconv.m_conv is not None
        assert isinstance(ddconv.m_conv, SConv2D)
    
    def test_init_custom_parameters(self, device):
        """Test DDConv initialization with custom parameters."""
        inc, outc = 16, 32
        kernel_size = 5
        padding = 2
        stride = 2
        
        ddconv = DDConv(
            inc, outc, device=str(device),
            kernel_size=kernel_size, padding=padding, stride=stride
        )
        
        assert ddconv.kernel_size == kernel_size
        assert ddconv.padding == padding
        assert ddconv.stride == stride
    
    def test_forward_basic(self, device):
        """Test basic forward pass without modulation."""
        inc, outc = 16, 32
        batch_size, height, width = 2, 32, 32
        
        ddconv = DDConv(inc, outc, device=str(device))
        ddconv.to(device)
        
        x = torch.randn(batch_size, inc, height, width, device=device)
        output = ddconv(x)
        
        # Check output shape - depends on conv implementation but should be valid
        assert output.dim() == 4
        assert output.size(0) == batch_size
        assert output.size(1) == outc
        assert torch.isfinite(output).all()
    
    def test_forward_with_modulation(self, device):
        """Test forward pass with modulation enabled."""
        inc, outc = 16, 32
        batch_size, height, width = 2, 24, 24
        
        ddconv = DDConv(inc, outc, device=str(device), modulation=True)
        ddconv.to(device)
        
        x = torch.randn(batch_size, inc, height, width, device=device)
        output = ddconv(x)
        
        assert output.dim() == 4
        assert output.size(0) == batch_size
        assert output.size(1) == outc
        assert torch.isfinite(output).all()
    
    def test_forward_different_sizes(self, device):
        """Test forward pass with different input sizes."""
        inc, outc = 8, 16
        ddconv = DDConv(inc, outc, device=str(device))
        ddconv.to(device)
        
        # Test different input sizes
        for height, width in [(16, 16), (32, 24), (28, 28)]:
            x = torch.randn(1, inc, height, width, device=device)
            output = ddconv(x)
            
            assert output.dim() == 4
            assert output.size(0) == 1
            assert output.size(1) == outc
            assert torch.isfinite(output).all()
    
    def test_gradient_computation(self, device):
        """Test gradient computation through DDConv."""
        inc, outc = 8, 16
        ddconv = DDConv(inc, outc, device=str(device))
        ddconv.to(device)
        
        x = torch.randn(2, inc, 16, 16, device=device, requires_grad=True)
        output = ddconv(x)
        loss = output.sum()
        
        loss.backward()
        
        # Check gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check model parameters have gradients
        for param in ddconv.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_device_consistency(self, device):
        """Test that DDConv operations stay on correct device."""
        inc, outc = 4, 8
        ddconv = DDConv(inc, outc, device=str(device))
        ddconv.to(device)
        
        x = torch.randn(1, inc, 8, 8, device=device)
        output = ddconv(x)
        
        assert output.device == device
    
    def test_p_conv_initialization(self, device):
        """Test that p_conv weights are initialized to zero."""
        inc, outc = 4, 8
        ddconv = DDConv(inc, outc, device=str(device))
        
        # p_conv weights should be initialized to zero
        assert torch.allclose(ddconv.p_conv.weight, torch.zeros_like(ddconv.p_conv.weight))
    
    def test_m_conv_initialization(self, device):
        """Test that m_conv weights are initialized to zero when modulation is enabled."""
        inc, outc = 4, 8
        ddconv = DDConv(inc, outc, device=str(device), modulation=True)
        
        # m_conv weights should be initialized to zero
        assert torch.allclose(ddconv.m_conv.weight, torch.zeros_like(ddconv.m_conv.weight))


class TestSConv2D:
    """Test SConv2D (Selective Convolution) implementation."""
    
    def test_init_basic(self, device):
        """Test SConv2D initialization with basic parameters."""
        in_channels, out_channels = 16, 32
        kernel_size = 3
        
        sconv = SConv2D(in_channels, out_channels, kernel_size)
        
        assert sconv.in_channels == in_channels
        assert sconv.out_channels == out_channels
        assert sconv.kernel_size == (kernel_size, kernel_size)
        assert hasattr(sconv, '_routing_fn')
        assert isinstance(sconv._routing_fn, _routing)
    
    def test_init_custom_parameters(self, device):
        """Test SConv2D initialization with custom parameters."""
        in_channels, out_channels = 8, 16
        kernel_size = 5
        stride = 2
        padding = 1
        num_experts = 4
        dropout_rate = 0.1
        
        sconv = SConv2D(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            num_experts=num_experts, dropout_rate=dropout_rate
        )
        
        assert sconv.stride == (stride, stride)
        assert sconv.padding == (padding, padding)
        # Check weight shape includes num_experts
        assert sconv.weight.size(0) == num_experts
    
    def test_forward_single_sample(self, device):
        """Test SConv2D forward pass with single sample."""
        in_channels, out_channels = 4, 8
        kernel_size = 3
        
        sconv = SConv2D(in_channels, out_channels, kernel_size)
        sconv.to(device)
        
        # Single sample input
        x = torch.randn(1, in_channels, 16, 16, device=device)
        output = sconv(x)
        
        assert output.dim() == 4
        assert output.size(0) == 1
        assert output.size(1) == out_channels
        assert torch.isfinite(output).all()
    
    def test_forward_batch(self, device):
        """Test SConv2D forward pass with batch input."""
        in_channels, out_channels = 8, 16
        kernel_size = 3
        batch_size = 4
        
        sconv = SConv2D(in_channels, out_channels, kernel_size)
        sconv.to(device)
        
        x = torch.randn(batch_size, in_channels, 16, 16, device=device)
        output = sconv(x)
        
        assert output.dim() == 4
        assert output.size(0) == batch_size
        assert output.size(1) == out_channels
        assert torch.isfinite(output).all()
    
    def test_gradient_computation(self, device):
        """Test gradient computation through SConv2D."""
        in_channels, out_channels = 4, 8
        sconv = SConv2D(in_channels, out_channels, 3)
        sconv.to(device)
        
        x = torch.randn(2, in_channels, 8, 8, device=device, requires_grad=True)
        output = sconv(x)
        loss = output.sum()
        
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check model parameters have gradients
        for param in sconv.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()


class TestRouting:
    """Test _routing module implementation."""
    
    def test_init(self, device):
        """Test _routing initialization."""
        in_channels = 32
        num_experts = 8
        dropout_rate = 0.2
        
        routing = _routing(in_channels, num_experts, dropout_rate)
        
        assert isinstance(routing.dropout, nn.Dropout)
        assert routing.dropout.p == dropout_rate
        assert isinstance(routing.fc, nn.Linear)
        assert routing.fc.in_features == in_channels
        assert routing.fc.out_features == num_experts
    
    def test_forward(self, device):
        """Test _routing forward pass."""
        in_channels = 16
        num_experts = 4
        routing = _routing(in_channels, num_experts, 0.1)
        routing.to(device)
        
        # Input should be flattened, so shape doesn't matter much
        x = torch.randn(1, in_channels, 8, 8, device=device)
        output = routing(x)
        
        assert output.dim() == 1
        assert output.size(0) == num_experts
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
        assert torch.isfinite(output).all()
    
    def test_sigmoid_output(self, device):
        """Test that routing output is properly sigmoid-activated."""
        in_channels = 8
        num_experts = 3
        routing = _routing(in_channels, num_experts, 0.0)  # No dropout for consistency
        routing.to(device)
        
        x = torch.randn(1, in_channels, 4, 4, device=device)
        output = routing(x)
        
        # All outputs should be between 0 and 1
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_gradient_computation(self, device):
        """Test gradient computation through routing."""
        in_channels = 8
        num_experts = 4
        routing = _routing(in_channels, num_experts, 0.1)
        routing.to(device)
        
        x = torch.randn(1, in_channels, 4, 4, device=device, requires_grad=True)
        output = routing(x)
        loss = output.sum()
        
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check model parameters have gradients
        for param in routing.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_different_input_shapes(self, device):
        """Test routing with different input shapes."""
        in_channels = 12
        num_experts = 6
        routing = _routing(in_channels, num_experts, 0.0)
        routing.to(device)
        
        # Test different spatial sizes
        for height, width in [(4, 4), (8, 8), (16, 16)]:
            x = torch.randn(1, in_channels, height, width, device=device)
            output = routing(x)
            
            assert output.dim() == 1
            assert output.size(0) == num_experts
            assert torch.all(output >= 0) and torch.all(output <= 1)
            assert torch.isfinite(output).all()

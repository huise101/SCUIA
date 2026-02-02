"""SCUIA: Semantic Contrast for Domain-Robust Underwater Image Quality Assessment

This module implements the neural network architectures,
including image and semantic encoders with ADDB (Adaptive Dual-Domain Block).
"""

from __future__ import division
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision.models import resnet18, resnet50
from timm.models.layers import DropPath
from ptflops import get_model_complexity_info

def get_sobel(in_chan, out_chan):
    """Create Sobel edge detection filters.
    
    Args:
        in_chan: Number of input channels
        out_chan: Number of output channels
        
    Returns:
        Tuple of (sobel_x, sobel_y) convolution layers
    """
    # Sobel filter kernels
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ], dtype=np.float32)
    
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ], dtype=np.float32)

    # Reshape and repeat for multiple channels
    filter_x = np.repeat(filter_x[None, None, :, :], in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)
    
    filter_y = np.repeat(filter_y[None, None, :, :], in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    # Create convolution layers with fixed weights
    filter_x = nn.Parameter(torch.from_numpy(filter_x), requires_grad=False)
    filter_y = nn.Parameter(torch.from_numpy(filter_y), requires_grad=False)
    
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, x):
    """Apply Sobel edge detection.
    
    Args:
        conv_x: Sobel convolution in x direction
        conv_y: Sobel convolution in y direction
        x: Input tensor
        
    Returns:
        Edge-enhanced input tensor
    """
    g_x = conv_x(x)
    g_y = conv_y(x)
    g = torch.sqrt(g_x.pow(2) + g_y.pow(2))
    return torch.sigmoid(g) * x


class EdgeSpatialAttention(nn.Module):
    """Edge-enhanced Spatial Attention Module.
    
    Uses Sobel filters to extract edge information and enhance spatial features.
    """
    def __init__(self, in_channels):
        super(EdgeSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sobel_x, self.sobel_y = get_sobel(in_channels, in_channels)

    def forward(self, x):
        # Edge detection with Sobel filters
        y = run_sobel(self.sobel_x, self.sobel_y, x)
        y = F.relu(self.bn(y))
        
        # Convolutional processing with residual connection
        y = self.conv1(y)
        y = x + y
        
        y = self.conv2(y)
        y = F.relu(self.bn(y))
        return y

class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network with 1x1 convolutions.
    
    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (defaults to in_features)
        out_features: Number of output features (defaults to in_features)
        act_layer: Activation layer class
        drop: Dropout rate
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def normal_init(module, mean=0, std=1, bias=0):
    """Initialize module with normal distribution."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    """Initialize module with constant values."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DynamicUpsampler(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=1, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        """Initialize position grid for offset sampling."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.stack(torch.meshgrid([h, h], indexing='ij'))
        return grid.transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        """Sample with dynamic offsets using bilinear interpolation."""
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        
        # Create coordinate grid
        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='xy'))
        coords = coords.transpose(1, 2).unsqueeze(1).unsqueeze(0)
        
        # Normalize coordinates to [-1, 1]
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        
        # Upsample and reshape
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
        coords = coords.view(B, 2, -1, self.scale * H, self.scale * W)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        
        # Grid sample
        x_reshaped = x.reshape(B * self.groups, -1, H, W)
        output = F.grid_sample(x_reshaped, coords, mode='bilinear',
                              align_corners=False, padding_mode="border")
        return output.view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

        


class ResNet18Encoder(nn.Module):
    """ResNet18-based feature extractor.
    
    Extracts 512-dimensional features from input images.
    """
    def __init__(self, pretrained=True):
        super(ResNet18Encoder, self).__init__()
        base_model = resnet18(pretrained=pretrained)
        # Remove the final classification layer
        self.resnet18 = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        features = self.resnet18(x)
        return features.squeeze(-1).squeeze(-1)




class ConvBlock(nn.Module):
    """Basic convolution block with optional normalization and activation.
    
    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        bias: Whether to use bias
        norm: Whether to use batch normalization
        relu: Whether to use GELU activation
        transpose: Whether to use transposed convolution
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, 
                 bias=True, norm=False, relu=True, transpose=False):
        super(ConvBlock, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = []
        
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, 
                                            padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, 
                                   padding=padding, stride=stride, bias=bias))
        
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
            
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResidualConvBlock(nn.Module):
    """Residual convolution block with skip connection."""
    def __init__(self, in_channel, out_channel):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = ConvBlock(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = ConvBlock(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.trans_layer = ConvBlock(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans_layer(out)
        out = self.conv2(out)
        return out + x



class SpatialProcessBlock(nn.Module):
    """Spatial processing block using residual convolutions."""
    def __init__(self, nc):
        super(SpatialProcessBlock, self).__init__()
        self.block = ResidualConvBlock(in_channel=nc, out_channel=nc)

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention module.
    
    Args:
        channel: Number of input channels
        reduction: Channel reduction ratio for bottleneck
    """
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FreqSpectrumAggregation(nn.Module):
    """Frequency spectrum dynamic aggregation module.
    
    Processes magnitude and phase components separately in frequency domain.
    """
    def __init__(self, nc):
        super(FreqSpectrumAggregation, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            ChannelAttention(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            ChannelAttention(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        # Extract magnitude and phase
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        
        # Process with residual connections
        mag = ori_mag + self.processmag(ori_mag)
        pha = ori_pha + self.processpha(ori_pha)
        
        # Reconstruct complex tensor
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        return torch.complex(real, imag)


class DualDomainMapping(nn.Module):
    """Dual-domain (spatial + frequency) nonlinear mapping.
    
    Processes input in both spatial and frequency domains, then fuses the results.
    """
    def __init__(self, in_nc):
        super(DualDomainMapping, self).__init__()
        self.spatial_process = SpatialProcessBlock(in_nc)
        self.frequency_process = FreqSpectrumAggregation(in_nc)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        _, _, H, W = x.shape
        
        # Frequency domain processing
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        
        # Spatial domain processing
        x_spatial = self.spatial_process(x)
        
        # Fusion
        x_cat = torch.cat([x_spatial, x_freq_spatial], dim=1)
        return self.cat(x_cat)


class ADDB(nn.Module):
    """Adaptive Dual-Domain Block (ADDB).
    
    Core component from SCUIA paper combining:
    1. DynamicUpsampler: Dynamic offset sampling with learned offsets (std=0.001 init)
    2. EdgeSpatialAttention: Edge-enhanced spatial attention using Sobel filters  
    3. DualDomainMapping: Spatial + frequency domain fusion (FFT → process → iFFT)
    
    
    Args:
        in_channels: Number of input channels"""
    def __init__(self, in_channels):
        super(ADDB, self).__init__()
        self.dy = DynamicUpsampler(in_channels)
        self.am = EdgeSpatialAttention(in_channels)
        self.ff = DualDomainMapping(in_channels)
    
    def forward(self, x):
        """Forward pass through ADDB.
        
        Processing pipeline:
        1. Dynamic upsampling for resolution enhancement
        2. Edge-enhanced spatial feature extraction
        3. Dual-domain (spatial + frequency) fusion
        """
        x = self.dy(x)  # Dynamic upsampling
        x = self.am(x)  # Spatial enhancement
        x = self.ff(x)  # Dual-domain mapping
        return x


class ImageEncoder(nn.Module):
    """Image feature extraction model.
    
    Combines ADDB preprocessing with ResNet encoder and projection head.
    
    Args:
        encoder: Encoder architecture ('resnet18', 'resnet50', 'swin')
        head: Projection head type ('linear' or 'mlp')
        feat_out_dim: Output feature dimension
    """
    def __init__(self, encoder='resnet18', head='mlp', feat_out_dim=128):
        super(ImageEncoder, self).__init__()
        
        # ADDB preprocessing
        self.addb = ADDB(in_channels=3)
        
        # Encoder network
        network_dims = {'resnet18': 512, 'resnet50': 2048, 'swin': 768}
        if encoder == 'resnet18':
            self.encoder = ResNet18Encoder()
        else:
            raise NotImplementedError(f'Encoder not supported: {encoder}')
        
        # Projection head
        encoder_dim = network_dims[encoder]
        if head == 'linear':
            self.head = nn.Linear(encoder_dim, feat_out_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.BatchNorm1d(encoder_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(encoder_dim // 2, encoder_dim // 4),
                nn.BatchNorm1d(encoder_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(encoder_dim // 4, feat_out_dim)
            )
        else:
            raise NotImplementedError(f'Head not supported: {head}')

    def forward(self, x):
        """Extract image features.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Feature vectors [B, feat_out_dim]
        """
        # ADDB processing
        x = self.addb(x)
        # Feature extraction
        feat = self.encoder(x)
        # Projection
        feat = self.head(feat)
        return feat



class SemanticEncoder(nn.Module):
    """Semantic feature extraction model using CLIP.
    
    Args:
        head_count: Number of annotator-specific projection heads
    """
    def __init__(self, head_count=1):
        super(SemanticEncoder, self).__init__()
        self.device = "cuda"
        self.head_count = head_count
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self.device)
        self.image_encoder = self.clip_model.visual
        self.text_encoder = self.clip_model.transformer
        
        # Annotator-specific projection heads
        self.projection_heads = nn.ModuleList(
            [nn.Linear(1024, 128) for _ in range(head_count)]
        )
        self.annotator_specific_projections = {}

    def forward(self, x):
        """Extract semantic features.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Tuple of (clip_features, annotator_projections)
        """
        # Extract CLIP features
        clip_features = self.image_encoder(x).cpu()
        
        # Apply annotator-specific projections
        for i in range(self.head_count):
            self.annotator_specific_projections[i] = self.projection_heads[i](clip_features)
            
        return clip_features, self.annotator_specific_projections


#
# if __name__ == '__main__':
#     model = ImageEncoder(encoder='resnet18', head='mlp')
#     input_tensor = torch.rand(2, 3, 224, 224)
#     output = model(input_tensor)
#     print(f"Input shape: {input_tensor.size()}")
#     print(f"Output shape: {output.size()}")


# ==============================================================================
# Model Performance Measurement Utilities
# ==============================================================================

def count_parameters(model):
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, input_res):
    """Measure FLOPs (floating point operations) of a model.
    
    Args:
        model: PyTorch model
        input_res: Input resolution tuple (C, H, W)
        
    Returns:
        Tuple of (MACs, params)
    """
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, input_res, as_strings=False, print_per_layer_stat=False
        )
    return macs, params


def measure_latency_and_memory(model, input_shape=(3, 224, 224),
                               device='cuda', batch_sizes=(1, 16),
                               n_warmup=20, n_runs=200):
    """Measure inference latency and memory usage.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (C, H, W)
        device: Device to run on ('cuda' or 'cpu')
        batch_sizes: Tuple of batch sizes to test
        n_warmup: Number of warmup iterations
        n_runs: Number of timing iterations
        
    Returns:
        Dictionary with latency and memory statistics per batch size
    """
    results = {}
    model.eval()
    model.to(device)

    for batch_size in batch_sizes:
        x = torch.randn((batch_size,) + input_shape, device=device)
        
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(x)
        if device.startswith('cuda'):
            torch.cuda.synchronize()

        # Reset peak memory counter
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)

        # Timing
        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(x)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / n_runs / batch_size
        peak_mem = None
        if device.startswith('cuda'):
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

        results[batch_size] = {
            'per_image_latency_s': avg_latency,
            'peak_memory_MB': peak_mem
        }
    return results

if __name__ == "__main__":
    print("=" * 70)
    print("SCUIA Model Performance Evaluation")
    print("=" * 70)
    
    # Initialize model
    model = ImageEncoder(encoder='resnet18', head='mlp')
    # model = SemanticEncoder()

    input_shape = (3, 224, 224)
    
    # Parameter count
    params = count_parameters(model)
    print(f"\nParameters: {params:,} ({params/1e6:.2f}M)")
    
    # FLOPs measurement
    macs, _ = measure_flops(model, input_res=input_shape)
    gflops_approx = macs * 2 / 1e9
    print(f"MACs: {macs:,} ({macs/1e9:.3f}G)")
    print(f"Approx GFLOPs: {gflops_approx:.3f}G")

    # GPU performance
    print("\n" + "-" * 70)
    print("GPU Performance:")
    gpu_results = measure_latency_and_memory(
        model, input_shape, device='cuda', batch_sizes=(1, 16)
    )
    for bs, metrics in gpu_results.items():
        print(f"  Batch Size {bs}:")
        print(f"    Per-image latency: {metrics['per_image_latency_s']*1000:.3f} ms")
        print(f"    Peak memory: {metrics['peak_memory_MB']:.2f} MB")

    # CPU performance
    print("\n" + "-" * 70)
    print("CPU Performance:")
    torch.set_num_threads(4)  # Adjust based on your CPU
    cpu_results = measure_latency_and_memory(
        model, input_shape, device='cpu', batch_sizes=(1, 16)
    )
    for bs, metrics in cpu_results.items():
        print(f"  Batch Size {bs}:")
        print(f"    Per-image latency: {metrics['per_image_latency_s']*1000:.3f} ms")
    
    print("\n" + "=" * 70)



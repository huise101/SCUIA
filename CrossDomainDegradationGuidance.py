"""Cross-Domain Degradation Guidance Module.

Implements F_D for extracting and aligning low/high-order statistical features
and underwater-specific metrics to mitigate domain shift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import h5py
import os
import configs
from Evaluation.dataloader import *

class CrossDomainDegradationGuidance(nn.Module):
    """F_D: Cross-Domain Degradation Guidance Module.
    
    Extracts and aligns low/high-order statistical features and underwater-specific
    metrics to mitigate domain shift between training and test domains.
    
    Key components:
    1. Statistical features: RGB channel mean, std, skewness, kurtosis
    2. Underwater metrics: chroma (U_C), saturation (U_S)
    3. Luminance distribution moments: F_μ, F_σ, F_s, F_κ
    4. HSV color bias and saturation deviation
    5. MLP projection g(·) for embedding alignment
    6. Wasserstein distance with high-quality reference set S_ref
    
    Args:
        p: Reference pristine features for comparison
        input_dim: Input feature dimension
        stability_scale: Numerical stability factor
        args: Additional configuration arguments
    """
    def __init__(self, p=None, input_dim=1500, stability_scale=0.001, args=None):
        super(CrossDomainDegradationGuidance, self).__init__()
        
        # Color feature module (chroma U_C)
        self.color_feature_module = ColorFeatureModule(input_dim)
        
        # Contrast feature module (related to saturation U_S)
        self.contrast_feature_module = ContrastFeatureModule(input_dim)

        # Reference statistical moments
        self.mu_r = None  # Mean (F_μ)
        self.sigma_r = None  # Covariance (F_σ)
        self.skewness_r = None  # Skewness (F_s)
        self.kurtosis_r = None  # Kurtosis (F_κ)
        self.eye_stability = None

        self.stability_scale = stability_scale
        self.args = args
        self.input_dim = input_dim

        if p is not None:
            self.compute_pristine_reference(p.unsqueeze(0))
        else:
            raise ValueError("Reference tensor p must be provided for F_D initialization.")

    def forward(self, x):
        # Dimension matching
        if x.shape[-1] != self.input_dim:
            if x.shape[-1] > self.input_dim:
                x = x[..., :self.input_dim]
            else:
                pad_size = self.input_dim - x.shape[-1]
                x = F.pad(x, (0, pad_size))

        # Extract underwater-specific metrics
        color_feature = self.color_feature_module(x)  # Chroma U_C
        contrast_feature = self.contrast_feature_module(x)  # Saturation U_S

        # Compute luminance distribution moments
        mu_t = torch.mean(x, dim=-2, keepdim=True)  # F_μ
        sigma_t = self.batch_covariance(x, mu_t, bias=True)  # F_σ
        skewness_t = self.batch_skewness(x, mu_t)  # F_s
        kurtosis_t = self.batch_kurtosis(x, mu_t)  # F_κ

        # Wasserstein distance approximation with reference
        mean_diff = self.mu_r - mu_t

        # Stable covariance sum
        cov_sum = ((self.sigma_r + sigma_t) / 2) + self.eye_stability
        L = torch.linalg.cholesky(cov_sum)
        cov_sum_inv = torch.cholesky_inverse(L)

        # Mahalanobis-style distance
        wasserstein_approx = torch.matmul(torch.matmul(mean_diff, cov_sum_inv), mean_diff.transpose(-2, -1))

        # Higher-order moment adjustments
        moment_adjustment = self.calculate_moment_adjustments(skewness_t, kurtosis_t)

        # Combined degradation score
        base_score = torch.sqrt(wasserstein_approx) + moment_adjustment

        # Integrate underwater metrics (color bias and saturation deviation)
        final_score = base_score * (1 + 0.95 * color_feature.mean() + 0.05 * contrast_feature.mean())

        return final_score.squeeze()

    def compute_pristine_reference(self, p):
        """Compute reference statistics from high-quality pristine images (S_ref)."""
        # Dimension matching
        if p.shape[-1] != self.input_dim:
            if p.shape[-1] > self.input_dim:
                p = p[..., :self.input_dim]
            else:
                pad_size = self.input_dim - p.shape[-1]
                p = F.pad(p, (0, pad_size))

        # Compute reference moments
        self.mu_r = torch.mean(p, dim=-2, keepdim=True)
        self.sigma_r = self.batch_covariance(p, self.mu_r, bias=True)
        self.skewness_r = self.batch_skewness(p, self.mu_r)
        self.kurtosis_r = self.batch_kurtosis(p, self.mu_r)
        self.eye_stability = self.stability_scale * torch.eye(p.size(-1), device=p.device).unsqueeze(0)

    def batch_covariance(self, tensor, mu, bias=False):
        """Compute batch covariance matrix."""
        tensor = tensor - mu
        factor = 1 / (tensor.shape[-2] - int(not bias))
        return factor * tensor.transpose(-1, -2) @ tensor.conj()

    def batch_skewness(self, tensor, mu):
        """Compute third-order moment (skewness F_s)."""
        diff = tensor - mu
        return torch.mean((diff ** 3), dim=-2)

    def batch_kurtosis(self, tensor, mu):
        """Compute fourth-order moment (kurtosis F_κ)."""
        diff = tensor - mu
        return torch.mean((diff ** 4), dim=-2) - 3

    def calculate_moment_adjustments(self, skewness_t, kurtosis_t, weight_skew=0.1, weight_kurt=0.1):
        """Calculate higher-order moment adjustments for distribution matching."""
        skew_adjustment = torch.mean(torch.abs(skewness_t)) * weight_skew
        kurt_adjustment = torch.mean(torch.relu(kurtosis_t)) * weight_kurt
        return skew_adjustment + kurt_adjustment


# Alias for backward compatibility
UIQE = CrossDomainDegradationGuidance


class ColorFeatureModule(nn.Module):
    """Extract chroma features (U_C) for underwater color cast analysis.
    
    Uses MLP projection g(·) to map statistical vectors s(I) to embeddings u.
    """
    def __init__(self, input_dim):
        super(ColorFeatureModule, self).__init__()
        hidden_dim = input_dim // 4

        # MLP projection for statistical embedding
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 375),
            nn.GELU(),
            nn.BatchNorm1d(375),
            nn.Linear(375, 64),
            nn.Sigmoid()
        )

        # Squeeze-and-Excitation for adaptive weighting
        self.se_module = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_patches, feat_dim = x.shape if x.dim() == 3 else (None, None, x.shape[-1])
        if batch_size is not None:
            x = x.view(-1, feat_dim)

        x = self.feature_extractor(x)
        se_weights = self.se_module(x)
        # Avoid numerical instability with bounded weights [0.5, 1.0]
        x = x * (0.5 + 0.5 * se_weights)

        if batch_size is not None:
            x = x.view(batch_size, num_patches, -1)

        return x.mean(dim=1).mean(dim=1, keepdim=True)


class ContrastFeatureModule(nn.Module):
    """Extract saturation features (U_S) for underwater image quality assessment.
    
    Uses 1D convolution to capture spatial contrast patterns.
    """
    def __init__(self, input_dim):
        super(ContrastFeatureModule, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(64)
        self.activation = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Ensure input shape is (batch, channels, features)
        if x.dim() == 3:
            batch_size, num_patches, feat_dim = x.shape
            x = x.permute(0, 2, 1)

        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.global_avg_pool(x)

        return x.mean(dim=1, keepdim=True)


def compute_degradation_guidance_distance(model, test_dataset, img_dir, data_loc, config):
    """Compute F_D degradation guidance scores for a dataset.
    
    Evaluates cross-domain degradation by comparing test images with high-quality
    reference set S_ref using Wasserstein distance approximation.
    """
    with torch.no_grad():
        ps = config.patch_size
        print("Computing degradation guidance scores using F_D module")
        scores = []
        moss = []
        names = []

        # Extract reference features from pristine images
        first_patches = pristine(config).to(config.device)
        all_ref_feats = model_features(model, first_patches)
        
        # Initialize F_D module
        fd_module = CrossDomainDegradationGuidance(all_ref_feats).to(config.device)

        dataset = TestDataset(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for batch, (x, y, name) in enumerate(tqdm(loader)):
            x = x.to(config.device)
            # Extract patches
            x = x.unfold(-3, x.size(-3), x.size(-3)).unfold(-3, ps, int(ps / 2)).unfold(-3, ps, int(ps / 2)).squeeze(1)
            x = x.contiguous().view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4), x.size(5))
            patches = x.view(-1, x.size(-3), x.size(-2), x.size(-1))

            # Extract features and compute degradation score
            all_rest_feats = model_features(model, patches)
            all_rest_feats = all_rest_feats.view(x.size(0), x.size(1), -1)

            score = fd_module(all_rest_feats)
            # Normalize score to [0, 1] range
            scaled_score = 1.0 - (1 / (1 + torch.exp(-score / 100.0)))
            
            if scaled_score.shape == torch.Size([]):
                scores.append(scaled_score.item())
            else:
                scores.extend(scaled_score.cpu().detach().tolist())
            moss.extend(y.tolist())
            names.extend(list(name))

            torch.cuda.empty_cache()

    return names, scores, moss


# Alias for backward compatibility
compute_uiqe_distance = compute_degradation_guidance_distance


def compute_degradation_guidance_single_image(model, test_image_path, config, tensor_return=False):
    """Compute F_D degradation guidance score for a single image."""
    with torch.no_grad():
        ps = config.patch_size

        # Extract reference features
        first_patches = pristine(config)
        all_ref_feats = model_features(model, first_patches)
        fd_module = CrossDomainDegradationGuidance(all_ref_feats).to(config.device)

        scores = []
        transform = transforms.ToTensor()
        x = Image.open(test_image_path)
        x = transform(x)

        x = x.to(config.device)
        # Extract patches
        x = x.unfold(-3, x.size(-3), x.size(-3)).unfold(-3, ps, int(ps / 2)).unfold(-3, ps, int(ps / 2)).squeeze(1)
        x = x.contiguous().view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4), x.size(5))
        patches = x.view(-1, x.size(-3), x.size(-2), x.size(-1))

        # Extract features and compute degradation score
        all_rest_feats = model_features(model, patches)
        all_rest_feats = all_rest_feats.view(x.size(0), x.size(1), -1)

        score = fd_module(all_rest_feats)
        scaled_score = 1.0 - (1 / (1 + torch.exp(-score / 100.0)))
        
        if scaled_score.shape == torch.Size([]):
            scores.append(scaled_score.item())
        else:
            scores.extend(scaled_score.cpu().detach().tolist())

        torch.cuda.empty_cache()

    if tensor_return:
        return scaled_score
    else:
        return scores


# Alias for backward compatibility
compute_uiqe_distance_single_image = compute_degradation_guidance_single_image

def model_features(model, frames):
    """Extract features from model for degradation analysis."""
    try:
        main_output = model(frames).squeeze()
    except:
        main_output = model(frames)
    return main_output


def cov(tensor, rowvar=False, bias=False):
    """Estimate covariance matrix for statistical feature extraction."""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    """Generate 2D Gaussian kernel for sharpness-based patch selection."""
    coords = torch.arange(kernel_size).to(dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()
    g /= g.sum()
    return g.unsqueeze(0)


def select_patches(all_patches, config):
    """Select high-sharpness patches for reference set S_ref construction."""
    p = config.sharpness_param

    selected_patches = torch.empty(1, all_patches.size(1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.to(config.device)

    kernel_size = 7
    kernel_sigma = float(7 / 6)
    deltas = []

    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        rest = rest.unsqueeze(dim=0)
        rest = transforms.Grayscale()(rest)
        kernel = gaussian_filter(kernel_size=kernel_size, sigma=kernel_sigma).view(
            1, 1, kernel_size, kernel_size).to(rest)
        
        # Compute local sharpness metric
        mu = F.conv2d(rest, kernel, padding=kernel_size // 2)
        mu_sq = mu ** 2
        std = F.conv2d(rest ** 2, kernel, padding=kernel_size // 2)
        std = ((std - mu_sq).abs().sqrt())
        delta = torch.sum(std)
        deltas.append([delta])

    peak_sharpness = max(deltas)[0].item()

    # Select patches above sharpness threshold
    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > p * peak_sharpness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches


def select_colorful_patches(all_patches, config):
    """Select high-colorfulness patches for reference set S_ref construction.
    
    Uses RG-YB opponent color space for underwater color assessment.
    """
    pc = config.colorfulness_param

    selected_patches = torch.empty(1, all_patches.size(1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.to(config.device)
    deltas = []

    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        R = rest[0, :, :]
        G = rest[1, :, :]
        B = rest[2, :, :]
        
        # Opponent color space for underwater metrics
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R + G) - B)
        rbMean = torch.mean(rg)
        rbStd = torch.std(rg)
        ybMean = torch.mean(yb)
        ybStd = torch.std(yb)
        
        # Colorfulness metric
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
        delta = stdRoot + meanRoot
        deltas.append([delta])

    peak_colorfulness = max(deltas)[0].item()

    # Select patches above colorfulness threshold
    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > pc * peak_colorfulness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches


def pristine(config):
    """Extract high-quality reference set S_ref for Wasserstein distance computation.
    
    Constructs reference distribution from pristine underwater images by selecting
    patches with high sharpness and colorfulness.
    """
    pristine_img_dir = config.pristine_img_dir
    ps = config.patch_size

    toten = transforms.ToTensor()
    refs = os.listdir(pristine_img_dir)

    cache_file = 'pristine_patches_%03d_%0.2f_%0.2f.hdf5' % (
        config.patch_size, config.sharpness_param, config.colorfulness_param)

    if not os.path.isfile(cache_file):
        print('Constructing high-quality reference set S_ref (first time initialization)')
        
        # Process first reference image
        temp = np.array(Image.open(pristine_img_dir + refs[0]))
        temp = toten(temp)
        batch = temp.to(config.device).unsqueeze(dim=0)
        patches = batch.unfold(1, 3, 3).unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.contiguous().view(1, -1, 3, ps, ps)

        # Randomize patch order
        for ix in range(patches.size(0)):
            patches[ix, :, :, :, :] = patches[ix, torch.randperm(patches.size()[1]), :, :, :]
        
        first_patches = patches.squeeze()
        # Select high-quality patches for S_ref
        first_patches = select_colorful_patches(select_patches(first_patches, config), config)

        # Process remaining reference images
        refs = refs[1:]
        for irx, rs in enumerate(tqdm(refs)):
            temp = np.array(Image.open(pristine_img_dir + rs))
            temp = toten(temp)
            batch = temp.to(config.device).unsqueeze(dim=0)
            patches = batch.unfold(1, 3, 3).unfold(2, ps, ps).unfold(3, ps, ps)
            patches = patches.contiguous().view(1, -1, 3, ps, ps)

            for ix in range(patches.size(0)):
                patches[ix, :, :, :, :] = patches[ix, torch.randperm(patches.size()[1]), :, :, :]
            
            second_patches = patches.squeeze()
            second_patches = select_colorful_patches(select_patches(second_patches, config), config)
            first_patches = torch.cat((first_patches, second_patches))

        # Cache reference set
        with h5py.File(cache_file, 'w') as f:
            dset = f.create_dataset('data', data=np.array(first_patches.detach().cpu(), dtype=np.float32))
    else:
        # Load cached reference set
        with h5py.File(cache_file, 'r') as f:
            first_patches = torch.tensor(f['data'][:], device=config.device)

    return first_patches

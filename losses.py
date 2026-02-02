"""Loss functions for SCUIA.

Implements:
- LHAC: Hierarchical Adaptive Contrastive Loss
- LTQC: Triplet Quality Contrastive Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
import datetime
import time


class TripletQualityContrastiveLoss(nn.Module):
    """LTQC: Triplet Quality Contrastive Loss.
    
    Explicitly models Bad/Fair/Perfect quality hierarchies to resolve feature entanglement
    across different quality images.
    
    Key components:
    - Embedding space interpolation: z_k = λz_i + (1-λ)z_j with learnable λ
    - Quality stratification (UIEB example):
        * low: interpolation weight < 0.4
        * mid: 0.4 - 0.7
        * high: >= 0.7
    - Positive pairs: same quality tier, weight difference <= 0.3
    - Negative pairs: different quality tiers, tier gap >= 1 (θ=1)
    - InfoNCE-based loss with temperature τ for hard negative control
    
    Args:
        batch_size: Batch size for processing
        temperature: Temperature parameter τ for controlling hard negative strength
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_i, emb_j):
        # Embedding interpolation: z_k = λz_i + (1-λ)z_j (λ=0.5 for simplicity)
        emb_k = (emb_i + emb_j) / 2

        # Concatenate and L2 normalize features
        representations = torch.cat([emb_i, emb_j, emb_k], dim=0)
        representations = F.normalize(representations, dim=1)

        M, N = emb_i.size(0), emb_j.size(0)
        P = emb_k.size(0)
        total = M + N + P

        # Quality tier labels (0: low, 1: mid, 2: high)
        labels = torch.cat([
            torch.zeros(M, dtype=torch.long, device=emb_i.device),
            torch.ones(N, dtype=torch.long, device=emb_i.device),
            2 * torch.ones(P, dtype=torch.long, device=emb_i.device)
        ])

        # Cosine similarity matrix
        sim_matrix = torch.mm(representations, representations.T)

        # Positive mask: same tier, excluding self
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(total, dtype=bool, device=emb_i.device)

        # Negative mask: different tiers
        neg_mask = ~pos_mask & ~torch.eye(total, dtype=bool, device=emb_i.device)

        # Compute positive and negative similarities
        pos_sim = sim_matrix[pos_mask].view(total, -1)
        neg_sim = sim_matrix[neg_mask].view(total, -1)

        # InfoNCE-style contrastive loss
        pos_term = torch.logsumexp(pos_sim / self.temperature, dim=1)
        neg_term = torch.logsumexp(neg_sim / self.temperature, dim=1)

        loss = (neg_term - pos_term).mean()
        return loss


def hierarchical_adaptive_contrastive_loss(features_images, features_augmentations, tau, annotator_matrices, mode_in=True):
    """LHAC: Hierarchical Adaptive Contrastive Loss.
    
    Constructs quality-ordered embedding space with dynamic weighting that emphasizes
    sample pairs with large quality differences or high uncertainty.
    
    Key components:
    1. Feature normalization: L2 norm, similarity via cosine
    2. Order Difference Matrix O_ij = |y_i - y_j| / D
       - y_i, y_j approximated by interpolation weights (unlabeled case)
       - Higher weight = higher quality
    3. Annotator Distribution Matrix A_ij:
       - Uses multi-view/multi-crop similarity variance as self-supervised proxy
       - Represents perception inconsistency/uncertainty
    4. Dynamic modulation weight α_ij: clamped to [0.5, 3.0] for stability
    5. Bidirectional aggregation to avoid directional bias
    
    Args:
        features_images: Image features of shape (B, D, C)
        features_augmentations: Augmented features of shape (B, D, C)
        tau: Temperature parameter for contrastive learning
        annotator_matrices: Annotator uncertainty matrices of shape (B, D, D)
        mode_in: If True, use L_in; otherwise use L_out
        
    Returns:
        Contrastive loss value
    """
    B, D, C = features_images.shape
    device = features_images.device
    eps = 1e-8

    # Step 1: Feature normalization (L2 norm)
    norm_images = torch.linalg.norm(features_images, dim=-1, keepdim=True).clamp(min=eps)
    norm_augs = torch.linalg.norm(features_augmentations, dim=-1, keepdim=True).clamp(min=eps)
    feats_img_norm = features_images / norm_images
    feats_aug_norm = features_augmentations / norm_augs

    # Compute feature similarity for alpha adjustment
    feat_sim = torch.bmm(feats_img_norm, feats_aug_norm.transpose(1, 2)).detach()  # (B, D, D)

    # Step 2: Order Difference Matrix O_ij = |y_i - y_j| / D
    indices = torch.arange(D, dtype=torch.float32, device=device)
    order_diff = (indices[:, None] - indices[None, :]).abs() / D  # (D, D)

    # Step 3 & 4: Dynamic modulation weight α_ij with Annotator Distribution Matrix
    # α_ij = clamp(2.0 - 2.0 / (1 + A_ij^2 + O_ij + feature_sim), [0.5, 3.0])
    annotator_sq = annotator_matrices.pow(2)
    alpha_base = 1.0 + annotator_sq + order_diff + feat_sim
    alpha = torch.clamp(2.0 - 2.0 / alpha_base, min=0.5, max=3.0)  # (B, D, D)

    # Cosine similarity matrix (scaled by temperature τ)
    logits = torch.bmm(feats_img_norm, feats_aug_norm.transpose(1, 2)) / tau  # (B, D, D)

    # Step 5: Bidirectional aggregation (avoid directional bias)
    if mode_in:
        # Numerically stable computation using logsumexp
        log_alpha = torch.log(alpha + eps)

        # Positive samples with alpha weighting in log space
        pos_logits = logits + log_alpha  # alpha * exp(logits)

        # Row-wise contrastive
        row_pos = torch.logsumexp(pos_logits, dim=-1)  # (B, D)
        row_neg = torch.logsumexp(logits, dim=-1)  # (B, D)
        row_loss = row_neg - row_pos  # (B, D)

        # Column-wise contrastive
        col_pos = torch.logsumexp(pos_logits.transpose(1, 2), dim=-1)  # (B, D)
        col_neg = torch.logsumexp(logits.transpose(1, 2), dim=-1)  # (B, D)
        col_loss = col_neg - col_pos  # (B, D)

        loss = (row_loss.mean() + col_loss.mean()) / 2
    else:
        # Negative sample space weighted contrastive
        log_denom_row = torch.logsumexp(logits, dim=-1, keepdim=True)  # (B, D, 1)
        log_denom_col = torch.logsumexp(logits, dim=-2, keepdim=True)  # (B, 1, D)

        # Cross-entropy style computation
        loss_row = alpha * (log_denom_row - logits)  # (B, D, D)
        loss_col = alpha * (log_denom_col - logits)  # (B, D, D)

        loss = (loss_row.mean() + loss_col.mean()) / 2

    return loss

# Testing LHAC
def test_lhac():
    ssim = torch.tril(torch.rand(5, 9, 9), diagonal=-1)
    ssim = ssim + torch.transpose(ssim, dim0=1, dim1=2)  # To get full matrix from lower triangular matrix
    ssim = torch.exp(-ssim)

    feat1 = torch.rand(5, 9, 128)
    feat2 = torch.rand(5, 9, 128)

    losses = hierarchical_adaptive_contrastive_loss(feat1, feat2, 0.2, ssim)
    print(losses)
    return


# Testing LTQC
def test_ltqc():
    pseudo_labels=torch.rand(16,1)
    f_feat=torch.rand(16,256)
    batch_size = 16

    idx = np.argsort(pseudo_labels.cpu(), axis=0)
    f_pos_feat = []
    f_neg_feat = []

    for n in range( batch_size // 4):
        try:
            f_pos_feat.append(f_feat[idx[n]])
            f_neg_feat.append(f_feat[idx[-n - 1]])
        except:
            continue

    f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
    f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)

    loss_fn = TripletQualityContrastiveLoss(f_pos_feat.shape[0], 1).cuda()
    loss = loss_fn(f_neg_feat, f_pos_feat)
    print(loss)

    return


def main():
    test_lhac()
    # test_ltqc()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

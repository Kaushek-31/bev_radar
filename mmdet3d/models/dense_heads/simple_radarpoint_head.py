# Copyright (c) Phigent Robotics. All rights reserved.
"""
🌟 PRODUCTION-READY RADAR POINT PREDICTION HEAD WITH VISUALIZATION
Simple, Fast, and Effective Query-Based Approach

Features:
- Query-based architecture (DETR-style)
- Proper false positive penalties
- Dynamic point count
- End-to-end training (no NMS)
- Built-in training visualization

Usage:
    radar_head = dict(
        type='RadarPointHead',
        in_channels=256,
        embed_dims=256,
        num_queries=500,
        visualize_training=True,  # Enable visualization
    )
"""


# ============================================================================
# IMPROVED CNN-BASED RADAR POINT HEAD
# ============================================================================

from mmdet.core import multi_apply, reduce_mean
from mmcv.runner import force_fp32
from ..builder import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import os
import numpy as np
import matplotlib.pyplot as plt

class SpatialAttentionPooling(nn.Module):
    """Learnable spatial attention pooling."""
    
    def __init__(self, in_channels, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        self.pool = nn.AdaptiveAvgPool2d(grid_size)
        
    def forward(self, x):
        attn_weights = self.attention(x)  # (B, 1, H, W)
        x_weighted = x * attn_weights     # (B, C, H, W)
        x_pooled = self.pool(x_weighted)  # (B, C, grid_size, grid_size)
        return x_pooled, attn_weights

@HEADS.register_module()
class SimpleCNNRadarHead(nn.Module):
    """
    NeuRadar-inspired CNN-based radar point prediction head.
    """
    
    def __init__(self,
                 in_channels=256,
                 num_points=256,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 hidden_channels=[512, 256, 128],
                 use_multi_scale=False,
                 fusion_method='concat',  # 'concat', 'sum', 'attention'
                 use_spatial_attention=True,
                 max_offset=4.5,
                 use_probabilistic=False,
                 loss_weight=0.02,
                 confidence_reg_weight = 0.5,
                 confidence_target = 0.3,
                 diversity_loss_weight=0.0,
                 visualize_training=False,
                 vis_interval=100):
        super().__init__()
        
        self.num_points = num_points
        self.loss_weight = loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.use_spatial_attention = use_spatial_attention
        self.use_multi_scale = use_multi_scale
        self.fusion_method = fusion_method
        self.max_offset = max_offset
        self.use_probabilistic = use_probabilistic
        self.visualize_training = visualize_training
        self.vis_interval = vis_interval
        self.fp16_enabled = False
        self.in_channels = in_channels  # Store for dynamic handling
        self.confidence_reg_weight = confidence_reg_weight
        self.confidence_target = confidence_target
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
        # Multi-scale fusion setup
        # FIXED: Don't assume number of levels, handle dynamically
        if use_multi_scale:
            if fusion_method == 'concat':
                # Will be determined dynamically based on actual number of levels
                # Create a flexible projection layer
                self.multi_scale_proj = None  # Created dynamically in forward
                encoder_in_channels = in_channels  # Will project to this
            elif fusion_method == 'attention':
                # Support up to 5 scale levels
                self.scale_attention = nn.ModuleList([
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(in_channels, in_channels // 4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels // 4, in_channels, 1),
                    ) for _ in range(5)  # Support up to 5 levels
                ])
                encoder_in_channels = in_channels
            else:  # 'sum'
                encoder_in_channels = in_channels
        else:
            encoder_in_channels = in_channels
        
        # Feature encoder - now uses fixed in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_in_channels, hidden_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels[1], hidden_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True),
        )
        
        # Spatial attention pooling
        if use_spatial_attention:
            self.grid_size = 8
            self.spatial_pool = SpatialAttentionPooling(hidden_channels[2], self.grid_size)
            fc_input_dim = hidden_channels[2] * self.grid_size * self.grid_size
        else:
            fc_input_dim = hidden_channels[2]
        
        # Reference point prediction + confidence
        self.reference_head = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(512, num_points * 4),  # (x, y, z, confidence)
        )
        
        # Offset prediction
        self.offset_head = nn.Sequential(
            nn.Linear(fc_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, num_points * 3),  # (δx, δy, δz)
        )
        
        # Probabilistic mode: Laplace scale parameters
        if use_probabilistic:
            self.laplace_head = nn.Sequential(
                nn.Linear(fc_input_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_points * 3),  # (b_x, b_y, b_z)
            )
        
        self._init_weights()
        
        # Visualization
        self.vis_iter = 0
        self.vis_dir = "work_dirs/neuradar_radar_viz"
        if self.visualize_training:
            os.makedirs(self.vis_dir, exist_ok=True)
            print(f"[NeuRadar RadarHead] Initialized:")
            print(f"  - In channels: {in_channels}")
            print(f"  - Multi-scale: {use_multi_scale}")
            print(f"  - Fusion: {fusion_method if use_multi_scale else 'N/A'}")
            print(f"  - Spatial attention: {use_spatial_attention}")
            print(f"  - Max offset: {max_offset}m")
            print(f"  - Probabilistic: {use_probabilistic}")
            print(f"  - Num points: {num_points}")
    
    def _init_weights(self):
        """Initialize weights with FORWARD-BIASED distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # CRITICAL FIX: Forward-biased initialization
        last_layer = self.reference_head[-1]
        
        import numpy as np
        np.random.seed(42)
        
        # Forward-biased: beta distribution pushes points forward
        # X: [-35, 50]m (front and back)
        # Y: [-50, 50]m (full lateral)
        # Z: [-5, 3]m (full height)
        x_init = np.random.uniform(self.pc_range[0], self.pc_range[3], self.num_points)
        y_init = np.random.uniform(self.pc_range[1], self.pc_range[4], self.num_points)
        z_init = np.random.uniform(self.pc_range[2], self.pc_range[5], self.num_points)
        
        init_points_world = np.stack([x_init, y_init, z_init], axis=1)  # (N, 3)
        
        # Convert to normalized coordinates for tanh
        pc_min = self.pc_range[:3].cpu().numpy()
        pc_max = self.pc_range[3:6].cpu().numpy()
        pc_center = (pc_min + pc_max) / 2
        pc_scale = (pc_max - pc_min) / 2
        
        init_points_norm = (init_points_world - pc_center) / pc_scale
        init_points_norm = np.clip(init_points_norm, -0.95, 0.95)  # Avoid tanh saturation
        
        # Inverse tanh
        init_bias = 0.5 * np.log((1 + init_points_norm) / (1 - init_points_norm))
        
        # Initialize confidence to target
        init_conf = np.full((self.num_points, 1), -0.847)  # logit(0.3)
        
        init_full = np.concatenate([init_bias, init_conf], axis=1).flatten()
        init_full = torch.from_numpy(init_full).float()
        
        with torch.no_grad():
            last_layer.bias.copy_(init_full)
        
        # Initialize offsets to VERY SMALL values
        with torch.no_grad():
            self.offset_head[-1].weight.normal_(0, 0.0001)  # Very small noise
            self.offset_head[-1].bias.zero_()
        
        if self.use_probabilistic:
            with torch.no_grad():
                self.laplace_head[-1].bias.fill_(-1.0)
        
        print(f"[NeuRadar RadarHead] Initialized with FORWARD-BIASED distribution")
    
    def _create_projection_layer(self, concat_channels, target_channels, device):
        """Dynamically create projection layer for concatenated features."""
        if concat_channels == target_channels:
            return nn.Identity().to(device)
        
        proj = nn.Sequential(
            nn.Conv2d(concat_channels, target_channels, 1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        ).to(device)
        
        # Initialize
        nn.init.kaiming_normal_(proj[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(proj[1].weight, 1)
        nn.init.constant_(proj[1].bias, 0)
        
        return proj
    
    def forward(self, feats):
        """Forward pass.
        
        Args:
            feats (list[torch.Tensor]): Multi-level BEV features
        
        Returns:
            tuple(list[dict]): Predictions per level
        """
        if self.use_multi_scale and len(feats) > 1:
            return self._forward_multi_scale(feats)
        else:
            return multi_apply(self.forward_single, feats)
    
    def _forward_multi_scale(self, feats):
        """Forward with multi-scale fusion."""
        B, C, H, W = feats[0].shape
        num_levels = len(feats)
        
        # Upsample to same size
        upsampled_feats = []
        for feat in feats:
            if feat.shape[-2:] != (H, W):
                feat_up = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            else:
                feat_up = feat
            upsampled_feats.append(feat_up)
        
        # Fuse features
        if self.fusion_method == 'concat':
            # Concatenate all available levels
            fused_feat = torch.cat(upsampled_feats, dim=1)  # (B, C*num_levels, H, W)
            
            # FIXED: Dynamically create projection layer if needed
            concat_channels = C * num_levels
            if concat_channels != self.in_channels:
                # Create projection layer on-the-fly
                if not hasattr(self, '_dynamic_proj') or \
                   self._dynamic_proj is None or \
                   self._last_concat_channels != concat_channels:
                    self._dynamic_proj = self._create_projection_layer(
                        concat_channels, self.in_channels, fused_feat.device
                    )
                    self._last_concat_channels = concat_channels
                    print(f"[NeuRadar] Created dynamic projection: {concat_channels} -> {self.in_channels}")
                
                fused_feat = self._dynamic_proj(fused_feat)
        
        elif self.fusion_method == 'sum':
            fused_feat = torch.stack(upsampled_feats, dim=0).sum(dim=0)
        
        elif self.fusion_method == 'attention':
            attention_weights = []
            for i, feat in enumerate(upsampled_feats):
                if i < len(self.scale_attention):
                    attn = self.scale_attention[i](feat)
                    attn = torch.sigmoid(attn)
                    attention_weights.append(attn)
                else:
                    # If more levels than attention modules, use uniform weight
                    attention_weights.append(torch.ones_like(feat[:, :1, :, :]))
            
            # Normalize attention
            attn_stack = torch.stack(attention_weights, dim=0)
            attn_stack = F.softmax(attn_stack, dim=0)
            
            weighted_feats = []
            for feat, attn in zip(upsampled_feats, attention_weights):
                weighted_feats.append(feat * attn)
            
            fused_feat = torch.stack(weighted_feats, dim=0).sum(dim=0)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        pred_dict = self.forward_single(fused_feat)
        
        return tuple([pred_dict for _ in range(len(feats))])
    
    def forward_single(self, bev_feat):
        """Forward with improved confidence calibration."""
        B = bev_feat.shape[0]
        
        # Encode features
        feat = self.encoder(bev_feat)
        
        # Spatial attention pooling
        if self.use_spatial_attention:
            feat_pooled, attn_weights = self.spatial_pool(feat)
            feat_pooled = feat_pooled.flatten(1)
        else:
            feat_pooled = F.adaptive_avg_pool2d(feat, 1)
            feat_pooled = feat_pooled.flatten(1)
            attn_weights = None
        
        # Predict reference points + confidence
        ref_conf_flat = self.reference_head(feat_pooled)
        ref_conf = ref_conf_flat.view(B, self.num_points, 4)
        
        ref_points_norm = torch.tanh(ref_conf[:, :, :3])
        confidence_logits = ref_conf[:, :, 3]
        
        # CHANGED: Apply temperature scaling to prevent saturation
        temperature = 2.0  # Higher = more conservative confidences
        confidence = torch.sigmoid(confidence_logits / temperature)
        
        # Predict offsets
        offset_flat = self.offset_head(feat_pooled)
        offsets = offset_flat.view(B, self.num_points, 3)
        offsets = torch.tanh(offsets) * self.max_offset  # Now uses larger max_offset
        
        # Scale to world coordinates
        pc_min = self.pc_range[:3]
        pc_max = self.pc_range[3:6]
        pc_center = (pc_min + pc_max) / 2
        pc_scale = (pc_max - pc_min) / 2
        
        ref_points_world = ref_points_norm * pc_scale + pc_center
        final_points = ref_points_world + offsets
        
        result = {
            'radar_points': final_points,
            'confidence': confidence,
            'ref_points': ref_points_world,
            'offsets': offsets,
            'attn_weights': attn_weights,
            'bev_feat': bev_feat,
        }
        
        if self.use_probabilistic:
            laplace_flat = self.laplace_head(feat_pooled)
            laplace_scales = laplace_flat.view(B, self.num_points, 3)
            laplace_scales = F.softplus(laplace_scales) + 1e-6
            result['laplace_scales'] = laplace_scales
        
        return [result]
    
    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, preds_dicts, gt_radar_points, img_metas=None, **kwargs):
        """Enhanced loss with all regularizations."""
        
        # Aggregate predictions
        all_pred_points = []
        all_confidence = []
        all_laplace_scales = []
        
        for level_preds in preds_dicts:
            all_pred_points.append(level_preds[0]['radar_points'])
            all_confidence.append(level_preds[0]['confidence'])
            if self.use_probabilistic:
                all_laplace_scales.append(level_preds[0]['laplace_scales'])
        
        if len(all_pred_points) > 1:
            pred_points = torch.stack(all_pred_points).mean(dim=0)
            confidence = torch.stack(all_confidence).mean(dim=0)
            if self.use_probabilistic:
                laplace_scales = torch.stack(all_laplace_scales).mean(dim=0)
        else:
            pred_points = all_pred_points[0]
            confidence = all_confidence[0]
            if self.use_probabilistic:
                laplace_scales = all_laplace_scales[0]
        
        # Main loss (now with normalization!)
        if self.use_probabilistic:
            main_loss = self.compute_probabilistic_loss(
                pred_points, confidence, laplace_scales,
                gt_radar_points, self.pc_range.tolist()
            )
        else:
            main_loss = self.compute_deterministic_loss(
                pred_points, confidence,
                gt_radar_points, self.pc_range.tolist()
            )
        
        # Confidence regularization (enhanced!)
        confidence_reg = self._confidence_regularization(confidence)
        
        # Diversity loss
        diversity_loss = self._diversity_loss(pred_points)
        
        # Coverage loss (enhanced!)
        coverage_loss = self._coverage_loss(pred_points, gt_radar_points)
        
        # Total loss
        total_loss = (
            main_loss + 
            self.confidence_reg_weight * confidence_reg +
            self.diversity_loss_weight * diversity_loss +
            0.1 * coverage_loss
        )
        
        # Visualization (unchanged)
        if self.visualize_training:
            self.vis_iter += 1
            if self.vis_iter % self.vis_interval == 0:
                if isinstance(gt_radar_points, list):
                    max_pts = max([len(p) for p in gt_radar_points])
                    padded_gt = []
                    for p in gt_radar_points:
                        p = torch.as_tensor(p, device=pred_points.device)
                        if len(p) < max_pts:
                            padding = torch.zeros((max_pts - len(p), 3), device=p.device)
                            p = torch.cat([p, padding])
                        padded_gt.append(p)
                    gt_radar_points_vis = torch.stack(padded_gt)
                else:
                    gt_radar_points_vis = gt_radar_points
                
                self._visualize_training(
                    pred_points, confidence, gt_radar_points_vis,
                    main_loss.item(), diversity_loss.item()
                )
        
        return {
            'loss_radar': self.loss_weight * total_loss,
            'loss_radar_main': main_loss,
            'loss_confidence_reg': confidence_reg,
            'loss_diversity': diversity_loss,
            'loss_coverage': coverage_loss,
            'avg_confidence': confidence.mean(),
            'max_confidence': confidence.max(),
            'min_confidence': confidence.min(),
            'high_conf_count': (confidence > 0.5).sum().float() / confidence.numel(),
        }
    
    def compute_deterministic_loss(self, pred_points, confidence, gt_points, point_cloud_range):
        """
        Enhanced NeuRadar deterministic loss with normalization and debugging.
        
        Key improvements:
        1. Normalize by GT count to handle variable GT across batches
        2. Enhanced debugging every 50 iterations
        3. Better handling of edge cases
        """
        # Handle list input
        if isinstance(gt_points, list):
            batch_size = len(gt_points)
            device = pred_points.device
            
            gt_points_tensors = []
            for gt in gt_points:
                if not isinstance(gt, torch.Tensor):
                    gt = torch.as_tensor(gt, device=device, dtype=torch.float32)
                else:
                    gt = gt.to(device)
                gt_points_tensors.append(gt)
        else:
            batch_size = pred_points.shape[0]
            gt_points_tensors = [gt_points[b] for b in range(batch_size)]
        
        xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range
        
        total_loss = 0.0
        total_matched_loss = 0.0
        total_unmatched_loss = 0.0
        total_gt_points = 0
        total_matched_pairs = 0
        
        # Track iteration for debug logging
        if not hasattr(self, 'iter_count'):
            self.iter_count = 0
        
        for b in range(batch_size):
            gt_b = gt_points_tensors[b]  # (M, 3)
            pred_b = pred_points[b]       # (N, 3)
            conf_b = confidence[b]         # (N,)
            
            # Filter GT points within range
            if len(gt_b) > 0:
                mask = (gt_b[:, 0] >= xmin) & (gt_b[:, 0] <= xmax) & \
                    (gt_b[:, 1] >= ymin) & (gt_b[:, 1] <= ymax) & \
                    (gt_b[:, 2] >= zmin) & (gt_b[:, 2] <= zmax)
                gt_b_filtered = gt_b[mask]
            else:
                gt_b_filtered = gt_b
            
            N_gt = gt_b_filtered.shape[0]
            N_pred = pred_b.shape[0]
            total_gt_points += N_gt
            
            if N_gt == 0:
                # No valid GT: penalize all high-confidence predictions
                unmatched_loss = -torch.log((1 - conf_b).clamp(min=1e-7)).sum()
                # Normalize by prediction count
                batch_loss = unmatched_loss / N_pred
                total_loss += batch_loss
                total_unmatched_loss += unmatched_loss.item()
                continue
            
            # Compute cost matrix (Equation 12c)
            distances = torch.cdist(pred_b, gt_b_filtered, p=1)  # L1 distance
            cost = distances - torch.log(conf_b.unsqueeze(1).clamp(min=1e-7))
            
            # Hungarian matching
            cost_matrix = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            total_matched_pairs += len(row_ind)
            
            # Matched vs unmatched
            matched_pred_idx = set(row_ind.tolist())
            unmatched_pred_idx = set(range(N_pred)) - matched_pred_idx
            
            # Loss on matched pairs (Equation 13, term 1)
            matched_loss = 0.0
            if len(row_ind) > 0:
                for i, j in zip(row_ind, col_ind):
                    matched_loss += torch.abs(pred_b[i] - gt_b_filtered[j]).sum()
                    matched_loss -= torch.log(conf_b[i].clamp(min=1e-7))
            
            # Loss on unmatched predictions (Equation 13, term 2)
            unmatched_loss = 0.0
            if len(unmatched_pred_idx) > 0:
                for i in unmatched_pred_idx:
                    unmatched_loss -= torch.log((1 - conf_b[i]).clamp(min=1e-7))
            
            # ===== KEY CHANGE: NORMALIZE LOSSES =====
            # Normalize matched loss by GT count (how many points we should fit)
            # Normalize unmatched loss by unmatched prediction count
            normalized_matched_loss = matched_loss / N_gt if N_gt > 0 else 0.0
            normalized_unmatched_loss = unmatched_loss / max(len(unmatched_pred_idx), 1)
            
            batch_loss = normalized_matched_loss + normalized_unmatched_loss
            
            total_loss += batch_loss
            total_matched_loss += matched_loss if isinstance(matched_loss, float) else matched_loss.item()
            total_unmatched_loss += unmatched_loss if isinstance(unmatched_loss, float) else unmatched_loss.item()
            
            # ===== DEBUG LOGGING EVERY 50 ITERATIONS =====
            if self.iter_count % 50 == 0 and b == 0:  # Only log first batch
                print(f"\n{'='*60}")
                print(f"[Iter {self.iter_count}] LOSS DEBUG - Batch {b}")
                print(f"{'='*60}")
                
                # GT statistics
                print(f"\n📊 GT Statistics:")
                print(f"   GT points (filtered): {N_gt}")
                if N_gt > 0:
                    gt_range_x = (gt_b_filtered[:, 0].min().item(), gt_b_filtered[:, 0].max().item())
                    gt_range_y = (gt_b_filtered[:, 1].min().item(), gt_b_filtered[:, 1].max().item())
                    gt_range_z = (gt_b_filtered[:, 2].min().item(), gt_b_filtered[:, 2].max().item())
                    print(f"   X range: [{gt_range_x[0]:.2f}, {gt_range_x[1]:.2f}]")
                    print(f"   Y range: [{gt_range_y[0]:.2f}, {gt_range_y[1]:.2f}]")
                    print(f"   Z range: [{gt_range_z[0]:.2f}, {gt_range_z[1]:.2f}]")
                
                # Prediction statistics
                print(f"\n🎯 Prediction Statistics:")
                print(f"   Predictions: {N_pred}")
                pred_range_x = (pred_b[:, 0].min().item(), pred_b[:, 0].max().item())
                pred_range_y = (pred_b[:, 1].min().item(), pred_b[:, 1].max().item())
                pred_range_z = (pred_b[:, 2].min().item(), pred_b[:, 2].max().item())
                print(f"   X range: [{pred_range_x[0]:.2f}, {pred_range_x[1]:.2f}]")
                print(f"   Y range: [{pred_range_y[0]:.2f}, {pred_range_y[1]:.2f}]")
                print(f"   Z range: [{pred_range_z[0]:.2f}, {pred_range_z[1]:.2f}]")
                
                # Confidence statistics
                print(f"\n🔍 Confidence Statistics:")
                print(f"   Mean: {conf_b.mean().item():.4f}")
                print(f"   Std: {conf_b.std().item():.4f}")
                print(f"   Min: {conf_b.min().item():.4f}")
                print(f"   Max: {conf_b.max().item():.4f}")
                print(f"   High conf (>0.5): {(conf_b > 0.5).sum().item()}/{N_pred}")
                print(f"   Low conf (<0.2): {(conf_b < 0.2).sum().item()}/{N_pred}")
                
                # Matching statistics
                print(f"\n🔗 Matching Statistics:")
                print(f"   Matched pairs: {len(row_ind)}")
                print(f"   Unmatched predictions: {len(unmatched_pred_idx)}")
                print(f"   Unmatched GT: {N_gt - len(row_ind)}")
                
                if len(row_ind) > 0:
                    matched_distances = [distances[i, j].item() for i, j in zip(row_ind, col_ind)]
                    matched_confs = [conf_b[i].item() for i in row_ind]
                    print(f"   Avg matched distance: {np.mean(matched_distances):.4f}m")
                    print(f"   Avg matched confidence: {np.mean(matched_confs):.4f}")
                    print(f"   Min matched distance: {np.min(matched_distances):.4f}m")
                    print(f"   Max matched distance: {np.max(matched_distances):.4f}m")
                
                # Compute min distance from each GT to any prediction
                if N_gt > 0:
                    min_dists_to_gt = distances.min(dim=0)[0]  # Min distance for each GT
                    print(f"\n📏 Coverage of GT points:")
                    print(f"   Avg min distance to GT: {min_dists_to_gt.mean().item():.4f}m")
                    print(f"   Max min distance to GT: {min_dists_to_gt.max().item():.4f}m")
                    far_gt = (min_dists_to_gt > 5.0).sum().item()
                    if far_gt > 0:
                        print(f"   ⚠️  GT points >5m from any pred: {far_gt}/{N_gt}")
                
                # Loss breakdown
                print(f"\n💰 Loss Breakdown:")
                print(f"   Matched loss (raw): {matched_loss if isinstance(matched_loss, float) else matched_loss.item():.6f}")
                print(f"   Unmatched loss (raw): {unmatched_loss if isinstance(unmatched_loss, float) else unmatched_loss.item():.6f}")
                print(f"   Matched loss (normalized): {normalized_matched_loss if isinstance(normalized_matched_loss, float) else normalized_matched_loss.item():.6f}")
                print(f"   Unmatched loss (normalized): {normalized_unmatched_loss if isinstance(normalized_unmatched_loss, float) else normalized_unmatched_loss.item():.6f}")
                print(f"   Batch loss: {batch_loss.item():.6f}")
                print(f"{'='*60}\n")
        
        self.iter_count += 1
        
        # Average over batch
        if batch_size == 0:
            return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
        
        avg_loss = total_loss / batch_size
        
        # Summary logging every 50 iterations
        if (self.iter_count - 1) % 50 == 0:
            print(f"\n📈 BATCH SUMMARY (Iter {self.iter_count - 1}):")
            print(f"   Total GT points: {total_gt_points}")
            print(f"   Avg GT per sample: {total_gt_points/batch_size:.1f}")
            print(f"   Total matched pairs: {total_matched_pairs}")
            print(f"   Avg matched per sample: {total_matched_pairs/batch_size:.1f}")
            print(f"   Average batch loss: {avg_loss.item():.6f}")
            print(f"   Total matched loss: {total_matched_loss/batch_size:.6f}")
            print(f"   Total unmatched loss: {total_unmatched_loss/batch_size:.6f}\n")
        
        return avg_loss


    def _confidence_regularization(self, confidence):
        """
        Enhanced confidence regularization to prevent saturation and encourage diversity.
        
        Goals:
        1. Keep average confidence near target (e.g., 0.3)
        2. Encourage variance in confidence (avoid all 0 or all 1)
        3. Penalize extreme values (very close to 0 or 1)
        
        Args:
            confidence: (B, N) confidence scores [0, 1]
        
        Returns:
            loss: Scalar regularization loss
        """
        B, N = confidence.shape
        
        # === Component 1: Target average confidence ===
        # Push average confidence towards target
        avg_conf = confidence.mean()
        target_loss = (avg_conf - self.confidence_target) ** 2
        
        # === Component 2: Encourage variance (diversity) ===
        # We want some predictions confident, others not
        # Penalize if all confidences are the same
        conf_var = confidence.var()
        
        # Target variance depends on confidence target
        # If target=0.3, ideal variance is 0.3*(1-0.3)=0.21 (bernoulli)
        target_var = self.confidence_target * (1 - self.confidence_target)
        variance_loss = (conf_var - target_var) ** 2
        
        # === Component 3: Penalize extreme saturation ===
        # Softly penalize confidences very close to 0 or 1
        # Use a smooth penalty instead of hard clipping
        lower_threshold = 0.05
        upper_threshold = 0.95
        
        # Penalize confidences < 0.05
        lower_penalty = torch.clamp(lower_threshold - confidence, min=0.0).mean()
        
        # Penalize confidences > 0.95
        upper_penalty = torch.clamp(confidence - upper_threshold, min=0.0).mean()
        
        saturation_loss = lower_penalty + upper_penalty
        
        # === Component 4: Entropy regularization (optional) ===
        # Encourage high entropy (uncertainty) in confidence distribution
        # entropy = -p*log(p) - (1-p)*log(1-p)
        eps = 1e-7
        entropy = -(confidence * torch.log(confidence + eps) + 
                    (1 - confidence) * torch.log(1 - confidence + eps))
        
        # We want high entropy on average (more uncertain = more diverse)
        avg_entropy = entropy.mean()
        max_entropy = np.log(2)  # Maximum entropy for binary
        
        # Penalize low entropy (over-confident predictions)
        entropy_loss = (max_entropy - avg_entropy) ** 2
        
        # === Combine all components ===
        total_reg_loss = (
            1.0 * target_loss +           # Keep near target
            0.5 * variance_loss +          # Maintain reasonable variance
            2.0 * saturation_loss +        # Strongly penalize saturation
            0.2 * entropy_loss             # Encourage diversity
        )
        
        # Debug logging
        if hasattr(self, 'iter_count') and self.iter_count % 50 == 0:
            print(f"\n🎯 CONFIDENCE REGULARIZATION (Iter {self.iter_count}):")
            print(f"   Current avg: {avg_conf.item():.4f} (target: {self.confidence_target})")
            print(f"   Current var: {conf_var.item():.4f} (target: {target_var:.4f})")
            print(f"   Avg entropy: {avg_entropy.item():.4f} (max: {max_entropy:.4f})")
            print(f"   Saturated low (<0.05): {(confidence < 0.05).sum().item()}/{B*N}")
            print(f"   Saturated high (>0.95): {(confidence > 0.95).sum().item()}/{B*N}")
            print(f"   Target loss: {target_loss.item():.6f}")
            print(f"   Variance loss: {variance_loss.item():.6f}")
            print(f"   Saturation loss: {saturation_loss.item():.6f}")
            print(f"   Entropy loss: {entropy_loss.item():.6f}")
            print(f"   Total reg loss: {total_reg_loss.item():.6f}")
        
        return total_reg_loss


    def _coverage_loss(self, pred_points, gt_points):
        """
        Enhanced coverage loss to handle variable GT counts better.
        
        Encourages predictions to:
        1. Cover the spatial extent of GT points
        2. Reach distant GT points
        3. Match the density distribution
        """
        if isinstance(gt_points, list):
            batch_size = len(gt_points)
            device = pred_points.device
            gt_points_tensors = []
            for gt in gt_points:
                if not isinstance(gt, torch.Tensor):
                    gt = torch.as_tensor(gt, device=device, dtype=torch.float32)
                else:
                    gt = gt.to(device)
                gt_points_tensors.append(gt)
        else:
            batch_size = pred_points.shape[0]
            gt_points_tensors = [gt_points[b] for b in range(batch_size)]
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            gt_b = gt_points_tensors[b]
            pred_b = pred_points[b]
            
            if len(gt_b) == 0:
                continue
            
            # === Component 1: Spatial extent coverage ===
            gt_min = gt_b.min(dim=0)[0]  # (3,)
            gt_max = gt_b.max(dim=0)[0]  # (3,)
            gt_center = (gt_min + gt_max) / 2
            gt_range = (gt_max - gt_min) / 2
            
            pred_min = pred_b.min(dim=0)[0]
            pred_max = pred_b.max(dim=0)[0]
            pred_center = (pred_min + pred_max) / 2
            pred_range = (pred_max - pred_min) / 2
            
            # Center alignment
            center_loss = torch.abs(pred_center - gt_center).mean()
            
            # Range coverage (predictions should span at least as much as GT)
            range_loss = torch.clamp(gt_range - pred_range, min=0.0).mean()
            
            # === Component 2: Nearest neighbor coverage ===
            # Every GT point should be reasonably close to some prediction
            distances = torch.cdist(gt_b, pred_b)  # (N_gt, N_pred)
            min_dists_to_pred = distances.min(dim=1)[0]  # (N_gt,)
            
            # Penalize if any GT is far from all predictions
            far_threshold = 5.0  # meters
            coverage_penalty = torch.clamp(min_dists_to_pred - far_threshold, min=0.0).mean()
            
            # === Component 3: Distribution matching ===
            # Predictions should roughly match GT density distribution
            # Use standard deviation as a proxy for spread
            gt_std = gt_b.std(dim=0)  # (3,)
            pred_std = pred_b.std(dim=0)  # (3,)
            
            spread_loss = torch.abs(pred_std - gt_std).mean()
            
            # Combine
            batch_coverage_loss = (
                1.0 * center_loss +
                1.0 * range_loss +
                0.5 * coverage_penalty +
                0.5 * spread_loss
            )
            
            total_loss += batch_coverage_loss
            valid_batches += 1
            
            # Debug logging
            if hasattr(self, 'iter_count') and self.iter_count % 50 == 0 and b == 0:
                print(f"\n📦 COVERAGE LOSS (Iter {self.iter_count}, Batch {b}):")
                print(f"   Center loss: {center_loss.item():.4f}")
                print(f"   Range loss: {range_loss.item():.4f}")
                print(f"   Coverage penalty: {coverage_penalty.item():.4f}")
                print(f"   Spread loss: {spread_loss.item():.4f}")
                print(f"   Total: {batch_coverage_loss.item():.4f}")
                print(f"   Max dist to any pred: {min_dists_to_pred.max().item():.2f}m")
        
        if valid_batches == 0:
            return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
        
        return total_loss / valid_batches
    
    def compute_probabilistic_loss(self, pred_points, confidence, laplace_scales, 
                                   gt_points, point_cloud_range):
        """
        NeuRadar probabilistic loss (Equation 14).
        
        L_prob = - Σ_(i,j)∉γ log(1 - r_i) 
                 - Σ_(i,j)∈γ log(r_i * p_i,x(y_j,x) * p_i,y(y_j,y) * p_i,z(y_j,z))
        
        where p_i,d is Laplace distribution with location μ and scale b.
        
        Args:
            pred_points: (B, N, 3) predicted means μ
            confidence: (B, N) existence probabilities r_i
            laplace_scales: (B, N, 3) scale parameters (b_x, b_y, b_z)
            gt_points: (B, M, 3) or list of ground truth points
            point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        
        Returns:
            loss: Scalar loss value
        """
        # Handle list input
        if isinstance(gt_points, list):
            batch_size = len(gt_points)
            device = pred_points.device
            
            gt_points_tensors = []
            for gt in gt_points:
                if not isinstance(gt, torch.Tensor):
                    gt = torch.as_tensor(gt, device=device, dtype=torch.float32)
                else:
                    gt = gt.to(device)
                gt_points_tensors.append(gt)
        else:
            batch_size = pred_points.shape[0]
            gt_points_tensors = [gt_points[b] for b in range(batch_size)]
        
        xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            gt_b = gt_points_tensors[b]  # (M, 3)
            pred_b = pred_points[b]       # (N, 3)
            conf_b = confidence[b]         # (N,)
            scales_b = laplace_scales[b]   # (N, 3)
            
            # Filter GT points
            if len(gt_b) > 0:
                mask = (gt_b[:, 0] >= xmin) & (gt_b[:, 0] <= xmax) & \
                       (gt_b[:, 1] >= ymin) & (gt_b[:, 1] <= ymax) & \
                       (gt_b[:, 2] >= zmin) & (gt_b[:, 2] <= zmax)
                gt_b_filtered = gt_b[mask]
            else:
                gt_b_filtered = gt_b
            
            if gt_b_filtered.shape[0] == 0:
                batch_loss = -torch.log((1 - conf_b).clamp(min=1e-6)).mean()
                total_loss += batch_loss
                valid_batches += 1
                continue
            
            n_gt = gt_b_filtered.shape[0]
            n_pred = pred_b.shape[0]
            
            # Compute cost matrix (same as deterministic)
            distances = torch.cdist(pred_b, gt_b_filtered, p=1)
            cost = distances - torch.log(conf_b.unsqueeze(1).clamp(min=1e-6))
            
            # Hungarian matching
            cost_matrix = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_pred_idx = set(row_ind.tolist())
            unmatched_pred_idx = set(range(n_pred)) - matched_pred_idx
            
            # Loss on matched pairs (Equation 14, term 2)
            matched_loss = 0.0
            for i, j in zip(row_ind, col_ind):
                # Laplace log probability: log p(y|μ,b) = -|y-μ|/b - log(2b)
                diff = torch.abs(pred_b[i] - gt_b_filtered[j])  # (3,)
                b_scales = scales_b[i]  # (3,)
                
                # Log probability for each dimension
                log_prob_x = -diff[0] / b_scales[0] - torch.log(2 * b_scales[0])
                log_prob_y = -diff[1] / b_scales[1] - torch.log(2 * b_scales[1])
                log_prob_z = -diff[2] / b_scales[2] - torch.log(2 * b_scales[2])
                
                # Combined log probability
                log_prob = log_prob_x + log_prob_y + log_prob_z
                
                # Add confidence term
                matched_loss -= torch.log(conf_b[i].clamp(min=1e-6)) + log_prob
            
            # Loss on unmatched predictions (Equation 14, term 1)
            unmatched_loss = 0.0
            for i in unmatched_pred_idx:
                unmatched_loss -= torch.log((1 - conf_b[i]).clamp(min=1e-6))
            
            batch_loss = (matched_loss + unmatched_loss) / n_pred
            
            total_loss += batch_loss
            valid_batches += 1
        
        if valid_batches == 0:
            return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
        
        return total_loss / valid_batches
    
    def _diversity_loss(self, pred_points):
        """Encourage diversity in predictions."""
        B, N, _ = pred_points.shape
        
        pred_expand1 = pred_points.unsqueeze(2)  # (B, N, 1, 3)
        pred_expand2 = pred_points.unsqueeze(1)  # (B, 1, N, 3)
        dists = torch.sqrt(torch.sum((pred_expand1 - pred_expand2) ** 2, dim=-1) + 1e-6)
        
        min_distance = 0.5  # meters
        violation = torch.clamp(min_distance - dists, min=0.0)
        
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_points.device).unsqueeze(0)
        violation = violation * mask.float()
        
        return violation.sum() / (B * N * (N - 1) + 1e-6)
    
    @torch.no_grad()
    def _visualize_training(self, pred_points, confidence, gt_points,
                          main_loss, diversity_loss):
        """Visualize training progress."""
        try:
            B = pred_points.shape[0]
            s_dir = os.path.join(self.vis_dir, f"iter_{self.vis_iter:06d}")
            os.makedirs(s_dir, exist_ok=True)
            
            b = 0
            vis_path = os.path.join(s_dir, f"batch{b}_visualization.png")
            
            # Visualize with confidence
            visualize_predictions_with_confidence(
                pred_points[b], confidence[b], gt_points[b], vis_path
            )
            
            # Save statistics
            stats_path = os.path.join(s_dir, "stats.txt")
            with open(stats_path, 'w') as f:
                f.write(f"=== Iteration {self.vis_iter} ===\n\n")
                f.write(f"Main loss: {main_loss:.6f}\n")
                f.write(f"Diversity loss: {diversity_loss:.6f}\n")
                f.write(f"Avg confidence: {confidence[b].mean():.4f}\n")
                f.write(f"Max confidence: {confidence[b].max():.4f}\n")
                f.write(f"Min confidence: {confidence[b].min():.4f}\n")
                f.write(f"High conf points (>0.5): {(confidence[b] > 0.5).sum().item()}\n")
            
            print(f"[NeuRadar RadarHead] Visualization saved: {s_dir}")
        except Exception as e:
            print(f"[NeuRadar RadarHead] Visualization failed: {e}")


def visualize_predictions_with_confidence(pred_points, confidence, gt_points, save_path):
    """Visualization with confidence scores."""
    try:
        pred_np = pred_points.detach().cpu().numpy().reshape(-1, 3)
        conf_np = confidence.detach().cpu().numpy().reshape(-1)
        gt_np = gt_points.detach().cpu().numpy().reshape(-1, 3)
        
        # Remove NaN/Inf
        valid_mask = np.isfinite(pred_np).all(axis=1) & np.isfinite(conf_np)
        pred_np = pred_np[valid_mask]
        conf_np = conf_np[valid_mask]
        gt_np = gt_np[np.isfinite(gt_np).all(axis=1)]
        
        fig = plt.figure(figsize=(24, 10))
        
        # BEV View with confidence
        ax1 = fig.add_subplot(2, 3, 1)
        if pred_np.size > 0:
            scatter = ax1.scatter(pred_np[:, 0], pred_np[:, 1], 
                                 c=conf_np, cmap='viridis', 
                                 s=30, alpha=0.7, marker='o',
                                 vmin=0, vmax=1)
            plt.colorbar(scatter, ax=ax1, label='Confidence')
        if gt_np.size > 0:
            ax1.scatter(gt_np[:, 0], gt_np[:, 1], 
                       c='red', s=30, alpha=0.6, label='GT', marker='x')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'BEV with Confidence | Pred: {len(pred_np)} | GT: {len(gt_np)}')
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(-50, 50)
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # High confidence predictions only
        ax2 = fig.add_subplot(2, 3, 2)
        high_conf_mask = conf_np > 0.5
        high_conf_pred = pred_np[high_conf_mask]
        if high_conf_pred.size > 0:
            ax2.scatter(high_conf_pred[:, 0], high_conf_pred[:, 1],
                       c='blue', s=30, alpha=0.7, label=f'High Conf (>{0.5})', marker='o')
        if gt_np.size > 0:
            ax2.scatter(gt_np[:, 0], gt_np[:, 1],
                       c='red', s=30, alpha=0.6, label='GT', marker='x')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'High Confidence Only | {len(high_conf_pred)} points')
        ax2.set_xlim(-50, 50)
        ax2.set_ylim(-50, 50)
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.hist(conf_np, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0.5, color='r', linestyle='--', label='Threshold=0.5')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Count')
        ax3.set_title('Confidence Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 3D view
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        if pred_np.size > 0:
            ax4.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2],
                       c=conf_np, cmap='viridis', s=20, alpha=0.6,
                       vmin=0, vmax=1)
        if gt_np.size > 0:
            ax4.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2],
                       c='red', s=20, alpha=0.6, label='GT')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('3D View')
        ax4.set_xlim(-50, 50)
        ax4.set_ylim(-50, 50)
        ax4.set_zlim(-5, 5)
        ax4.view_init(elev=20, azim=45)
        ax4.legend()
        
        # Close-up BEV (±20m)
        ax5 = fig.add_subplot(2, 3, 5)
        if pred_np.size > 0:
            scatter = ax5.scatter(pred_np[:, 0], pred_np[:, 1],
                                 c=conf_np, cmap='viridis',
                                 s=50, alpha=0.7, marker='o',
                                 vmin=0, vmax=1)
        if gt_np.size > 0:
            ax5.scatter(gt_np[:, 0], gt_np[:, 1],
                       c='red', s=50, alpha=0.6, label='GT', marker='x')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_title('Close-up BEV (±20m)')
        ax5.set_xlim(-20, 20)
        ax5.set_ylim(-20, 20)
        ax5.set_aspect('equal')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Confidence vs distance
        ax6 = fig.add_subplot(2, 3, 6)
        if pred_np.size > 0:
            distances = np.sqrt(pred_np[:, 0]**2 + pred_np[:, 1]**2)
            ax6.scatter(distances, conf_np, alpha=0.5, s=10)
            ax6.set_xlabel('Distance from sensor (m)')
            ax6.set_ylabel('Confidence')
            ax6.set_title('Confidence vs Distance')
            ax6.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        plt.close('all')


# ============================================================================
# IMPROVED CNN-BASED RADAR POINT HEAD
# ============================================================================

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.runner import force_fp32
# from mmdet.models import HEADS
# import numpy as np
# import os
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mmdet.core import multi_apply

# # ============================================================================
# # VISUALIZATION UTILITIES
# # ============================================================================

# def visualize_bev(bev_feat, save_path):
#     """Visualize BEV features as heatmap."""
#     try:
#         bev_np = bev_feat.detach().cpu().numpy()
#         bev_heatmap = bev_np.mean(axis=0)
        
#         plt.figure(figsize=(8, 8))
#         plt.imshow(bev_heatmap, cmap='viridis', origin='lower')
#         plt.colorbar(label='Feature Magnitude')
#         plt.title('BEV Feature Heatmap')
#         plt.axis('off')
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=100, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f"Warning: BEV visualization failed: {e}")
#         plt.close('all')

# def visualize_point_cloud_3d(pred_points, gt_points, save_path, pred_scores=None, title=""):
#     """Visualize predicted vs ground truth points in 3D."""
#     try:
#         pred_np = pred_points.detach().cpu().numpy()
#         gt_np = gt_points.detach().cpu().numpy()
        
#         fig = plt.figure(figsize=(20, 6))
        
#         # 1. Predicted points
#         ax1 = fig.add_subplot(1, 4, 1, projection='3d')
#         if len(pred_np) > 0:
#             if pred_scores is not None:
#                 scores_np = pred_scores.detach().cpu().numpy()
#                 scatter = ax1.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2],
#                                     c=scores_np, cmap='viridis', s=30, alpha=0.7,
#                                     vmin=0, vmax=1)
#                 plt.colorbar(scatter, ax=ax1, label='Confidence', shrink=0.6)
#             else:
#                 ax1.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2],
#                           c='blue', s=30, alpha=0.7)
#         ax1.set_xlabel('X (m)', fontsize=8)
#         ax1.set_ylabel('Y (m)', fontsize=8)
#         ax1.set_zlabel('Z (m)', fontsize=8)
#         ax1.set_title(f'Predicted ({len(pred_np)} pts)', fontsize=10)
#         ax1.set_xlim(-50, 50)
#         ax1.set_ylim(-50, 50)
#         ax1.set_zlim(-5, 5)
#         ax1.view_init(elev=20, azim=45)
#         ax1.grid(True, alpha=0.3)
        
#         # 2. Ground truth
#         ax2 = fig.add_subplot(1, 4, 2, projection='3d')
#         if len(gt_np) > 0:
#             ax2.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2],
#                        c='red', s=30, alpha=0.7)
#         ax2.set_xlabel('X (m)', fontsize=8)
#         ax2.set_ylabel('Y (m)', fontsize=8)
#         ax2.set_zlabel('Z (m)', fontsize=8)
#         ax2.set_title(f'Ground Truth ({len(gt_np)} pts)', fontsize=10)
#         ax2.set_xlim(-50, 50)
#         ax2.set_ylim(-50, 50)
#         ax2.set_zlim(-5, 5)
#         ax2.view_init(elev=20, azim=45)
#         ax2.grid(True, alpha=0.3)
        
#         # 3. Overlay
#         ax3 = fig.add_subplot(1, 4, 3, projection='3d')
#         if len(pred_np) > 0:
#             ax3.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2],
#                        c='blue', s=20, alpha=0.5, label='Predicted')
#         if len(gt_np) > 0:
#             ax3.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2],
#                        c='red', s=20, alpha=0.5, label='GT')
#         ax3.set_xlabel('X (m)', fontsize=8)
#         ax3.set_ylabel('Y (m)', fontsize=8)
#         ax3.set_zlabel('Z (m)', fontsize=8)
#         ax3.set_title('Overlay', fontsize=10)
#         ax3.set_xlim(-50, 50)
#         ax3.set_ylim(-50, 50)
#         ax3.set_zlim(-5, 5)
#         ax3.view_init(elev=20, azim=45)
#         ax3.legend(fontsize=8)
#         ax3.grid(True, alpha=0.3)
        
#         # 4. BEV (top view)
#         ax4 = fig.add_subplot(1, 4, 4)
#         if len(pred_np) > 0:
#             ax4.scatter(pred_np[:, 0], pred_np[:, 1], c='blue', s=20, alpha=0.5, label='Pred')
#         if len(gt_np) > 0:
#             ax4.scatter(gt_np[:, 0], gt_np[:, 1], c='red', s=20, alpha=0.5, label='GT')
#         ax4.set_xlabel('X (m)', fontsize=8)
#         ax4.set_ylabel('Y (m)', fontsize=8)
#         ax4.set_title('BEV (Top View)', fontsize=10)
#         ax4.set_xlim(-50, 50)
#         ax4.set_ylim(-50, 50)
#         ax4.set_aspect('equal')
#         ax4.grid(True, alpha=0.3)
#         ax4.legend(fontsize=8)
        
#         if title:
#             fig.suptitle(title, fontsize=12, fontweight='bold')
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=100, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f"Warning: 3D visualization failed: {e}")
#         plt.close('all')

# def visualize_query_distribution(query_pos, cls_scores, save_path, title=""):
#     """Visualize query position distribution and confidence."""
#     try:
#         query_np = query_pos.detach().cpu().numpy()
#         scores_np = cls_scores.detach().cpu().numpy()
        
#         fig = plt.figure(figsize=(15, 5))
        
#         # 1. Query positions colored by confidence
#         ax1 = fig.add_subplot(1, 3, 1)
#         scatter = ax1.scatter(query_np[:, 0], query_np[:, 1], 
#                             c=scores_np, s=15, alpha=0.6, cmap='viridis', vmin=0, vmax=1)
#         plt.colorbar(scatter, ax=ax1, label='Confidence')
#         ax1.set_xlabel('X (m)')
#         ax1.set_ylabel('Y (m)')
#         ax1.set_title(f'Query Distribution ({len(query_np)} queries)')
#         ax1.set_xlim(-50, 50)
#         ax1.set_ylim(-50, 50)
#         ax1.set_aspect('equal')
#         ax1.grid(True, alpha=0.3)
        
#         # 2. High-confidence queries
#         ax2 = fig.add_subplot(1, 3, 2)
#         high_conf_mask = scores_np > 0.5
#         if high_conf_mask.sum() > 0:
#             ax2.scatter(query_np[high_conf_mask, 0], query_np[high_conf_mask, 1],
#                        c='green', s=25, alpha=0.7, label=f'High conf ({high_conf_mask.sum()})')
#         if (~high_conf_mask).sum() > 0:
#             ax2.scatter(query_np[~high_conf_mask, 0], query_np[~high_conf_mask, 1],
#                        c='gray', s=10, alpha=0.3, label=f'Low conf ({(~high_conf_mask).sum()})')
#         ax2.set_xlabel('X (m)')
#         ax2.set_ylabel('Y (m)')
#         ax2.set_title('High vs Low Confidence')
#         ax2.set_xlim(-50, 50)
#         ax2.set_ylim(-50, 50)
#         ax2.set_aspect('equal')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         # 3. Confidence histogram
#         ax3 = fig.add_subplot(1, 3, 3)
#         ax3.hist(scores_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
#         ax3.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
#         ax3.set_xlabel('Confidence Score')
#         ax3.set_ylabel('Count')
#         ax3.set_title('Confidence Distribution')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)
        
#         if title:
#             fig.suptitle(title, fontsize=12, fontweight='bold')
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=100, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f"Warning: Query visualization failed: {e}")
#         plt.close('all')

# def visualize_loss_components(losses_dict, save_path, title=""):
#     """Visualize loss components as bar chart."""
#     try:
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         loss_names = list(losses_dict.keys())
#         loss_values = [losses_dict[k].item() if isinstance(losses_dict[k], torch.Tensor) 
#                       else losses_dict[k] for k in loss_names]
        
#         colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
#         bars = ax.bar(range(len(loss_names)), loss_values, color=colors[:len(loss_names)])
        
#         ax.set_xlabel('Loss Component', fontsize=12)
#         ax.set_ylabel('Loss Value', fontsize=12)
#         ax.set_title(title if title else 'Loss Components', fontsize=14, fontweight='bold')
#         ax.set_xticks(range(len(loss_names)))
#         ax.set_xticklabels([k.replace('loss_radar_', '') for k in loss_names], rotation=45, ha='right')
#         ax.grid(True, alpha=0.3, axis='y')
        
#         # Add value labels on bars
#         for bar, val in zip(bars, loss_values):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height,
#                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=100, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f"Warning: Loss visualization failed: {e}")
#         plt.close('all')

# # ============================================================================
# # MAIN HEAD - Simple & Effective with Visualization
# # ============================================================================

# @HEADS.register_module()
# class SimpleQueryRadarPointHead(nn.Module):
#     """
#     Simple Query-Based Radar Point Prediction Head with Training Visualization.
    
#     Architecture:
#         BEV Features → Query Generation → Transformer → Point Predictions
        
#     Key Features:
#         Dynamic point count (adapts to scene)
#         No NMS required (end-to-end)
#         Proper false positive penalties
#         Fast and memory efficient
#         Production ready
#         Built-in training visualization
#     """
    
#     def __init__(self,
#                  in_channels=256,
#                  embed_dims=256,
#                  num_queries=500,
#                  num_decoder_layers=3,
#                  num_heads=8,
#                  point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
#                  confidence_threshold=0.3,
#                  loss_cls_weight=2.0,
#                  loss_reg_weight=5.0,
#                  loss_fp_weight=1.0,
#                  visualize_training=False,
#                  vis_interval=100):
#         """
#         Args:
#             in_channels (int): Input BEV feature channels
#             embed_dims (int): Embedding dimension for transformer
#             num_queries (int): Maximum number of query points
#             num_decoder_layers (int): Number of transformer decoder layers
#             num_heads (int): Number of attention heads
#             point_cloud_range (list): [x_min, y_min, z_min, x_max, y_max, z_max]
#             confidence_threshold (float): Threshold for inference
#             loss_cls_weight (float): Classification loss weight
#             loss_reg_weight (float): Regression loss weight
#             loss_fp_weight (float): False positive penalty weight
#             visualize_training (bool): Enable training visualization
#             vis_interval (int): Visualize every N iterations
#         """
#         super().__init__()
        
#         self.embed_dims = embed_dims
#         self.num_queries = num_queries
#         self.conf_threshold = confidence_threshold
#         self.loss_cls_weight = loss_cls_weight
#         self.loss_reg_weight = loss_reg_weight
#         self.loss_fp_weight = loss_fp_weight
#         self.visualize_training = visualize_training
#         self.vis_interval = vis_interval
        
#         self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
#         # Input projection
#         self.input_proj = nn.Sequential(
#             nn.Conv2d(in_channels, embed_dims, 3, padding=1),
#             nn.BatchNorm2d(embed_dims),
#             nn.ReLU(inplace=True),
#         )
        
#         # Learnable query embeddings
#         self.query_embed = nn.Embedding(num_queries, embed_dims)
#         self.query_pos_embed = nn.Embedding(num_queries, 3)  # (x, y, z) positions
        
#         # Transformer decoder
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=embed_dims,
#             nhead=num_heads,
#             dim_feedforward=embed_dims * 4,
#             dropout=0.1,
#             activation='relu',
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
#         # Prediction heads
#         self.cls_head = nn.Sequential(
#             nn.Linear(embed_dims, embed_dims // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dims // 2, 1),
#         )
        
#         self.reg_head = nn.Sequential(
#             nn.Linear(embed_dims, embed_dims // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dims // 2, 3),
#         )
        
#         self._init_weights()
        
#         # Visualization tracking
#         self.vis_iter = 0
#         self.vis_dir = "work_dirs/simple_radar_viz"
#         if self.visualize_training:
#             os.makedirs(self.vis_dir, exist_ok=True)
#             print(f"📊 Radar visualization enabled: {self.vis_dir}")
    
#     def _init_weights(self):
#         """Initialize weights."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
#         # Initialize query positions in a circular pattern
#         angles = torch.linspace(0, 2 * np.pi, self.num_queries)
#         radii = torch.linspace(5, 40, self.num_queries)
#         x = radii * torch.cos(angles)
#         y = radii * torch.sin(angles)
#         z = torch.zeros_like(x)
#         init_pos = torch.stack([x, y, z], dim=1)
#         self.query_pos_embed.weight.data = init_pos
    
#     def forward(self, bev_feat):
#         """
#         Forward pass.
        
#         Args:
#             bev_feat (Tensor): BEV features (B, C, H, W)
            
#         Returns:
#             list[dict]: List containing prediction dict
#         """
#         B, C, H, W = bev_feat.shape
        
#         # Project input
#         bev_feat_proj = self.input_proj(bev_feat)  # (B, embed_dims, H, W)
        
#         # Flatten for transformer
#         bev_flat = bev_feat_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dims)
        
#         # Get learnable queries
#         queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, embed_dims)
#         query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, 3)
        
#         # Transformer decoder
#         queries = self.decoder(queries, bev_flat)  # (B, num_queries, embed_dims)
        
#         # Predictions
#         cls_logits = self.cls_head(queries).squeeze(-1)  # (B, num_queries)
#         cls_scores = torch.sigmoid(cls_logits)
        
#         pred_offsets = self.reg_head(queries)  # (B, num_queries, 3)
#         pred_positions = query_pos + torch.tanh(pred_offsets) * 5.0  # Scale offsets
        
#         # During inference, filter by confidence
#         if not self.training:
#             radar_points = []
#             point_scores = []
            
#             for b in range(B):
#                 mask = cls_scores[b] > self.conf_threshold
#                 if mask.sum() > 0:
#                     radar_points.append(pred_positions[b, mask])
#                     point_scores.append(cls_scores[b, mask])
#                 else:
#                     radar_points.append(torch.zeros((0, 3), device=bev_feat.device))
#                     point_scores.append(torch.zeros((0,), device=bev_feat.device))
#         else:
#             radar_points = pred_positions
#             point_scores = cls_scores
        
#         return [{
#             'radar_points': radar_points,
#             'point_scores': point_scores,
#             'cls_logits': cls_logits,
#             'query_pos': query_pos,
#             'pred_offsets': pred_offsets,
#             'pred_positions': pred_positions,
#             'bev_feat': bev_feat,  # For visualization
#         }]
    
#     @force_fp32(apply_to=('preds_dicts',))
#     def loss(self, preds_dicts, gt_radar_points, img_metas=None, **kwargs):
#         """
#         Compute losses with Hungarian matching and visualization.
        
#         Args:
#             preds_dicts (list[dict]): Predictions
#             gt_radar_points: Ground truth radar points
#             img_metas: Image metadata
            
#         Returns:
#             dict: Loss dictionary
#         """
#         pred = preds_dicts[0]
        
#         cls_logits = pred['cls_logits']
#         query_pos = pred['query_pos']
#         pred_offsets = pred['pred_offsets']
#         pred_positions = pred['pred_positions']
#         cls_scores = torch.sigmoid(cls_logits)
        
#         B = cls_logits.shape[0]
        
#         # Prepare GT
#         if isinstance(gt_radar_points, torch.Tensor):
#             gt_radar_points = [gt_radar_points[i] for i in range(B)]
#         else:
#             gt_radar_points = [torch.as_tensor(pts, device=cls_logits.device)
#                              for pts in gt_radar_points]
        
#         # Hungarian matching
#         indices = self._hungarian_matching(pred_positions, gt_radar_points, cls_logits)
        
#         # Compute losses
#         cls_loss = self._classification_loss(cls_logits, indices, gt_radar_points)
#         reg_loss = self._regression_loss(pred_positions, indices, gt_radar_points)
#         fp_loss = self._false_positive_loss(pred_positions, cls_logits, indices, gt_radar_points)
        
#         losses = {
#             'loss_radar_cls': self.loss_cls_weight * cls_loss,
#             'loss_radar_reg': self.loss_reg_weight * reg_loss,
#             'loss_radar_fp': self.loss_fp_weight * fp_loss,
#         }
        
#         # Visualization
#         if self.visualize_training:
#             self.vis_iter += 1
#             if self.vis_iter % self.vis_interval == 0:
#                 self._visualize_training(pred, gt_radar_points, indices, losses)
        
#         return losses
    
#     def _hungarian_matching(self, pred_positions, gt_points_list, cls_logits):
#         """Hungarian matching between predictions and ground truth."""
#         from scipy.optimize import linear_sum_assignment
        
#         B = len(gt_points_list)
#         indices = []
        
#         for b in range(B):
#             pred_pts = pred_positions[b]  # (N_queries, 3)
#             gt_pts = gt_points_list[b]    # (N_gt, 3)
            
#             if len(gt_pts) == 0:
#                 indices.append((np.array([]), np.array([])))
#                 continue
            
#             # Cost matrix: spatial distance + classification cost
#             cost_spatial = torch.cdist(pred_pts, gt_pts)  # (N_queries, N_gt)
#             cost_cls = -cls_logits[b].sigmoid().unsqueeze(1)  # (N_queries, 1)
#             cost = cost_spatial + cost_cls
            
#             # Hungarian algorithm
#             pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
#             indices.append((pred_idx, gt_idx))
        
#         return indices
    
#     def _classification_loss(self, cls_logits, indices, gt_points_list):
#         """Focal loss for classification."""
#         B = cls_logits.shape[0]
#         losses = []
        
#         for b in range(B):
#             pred_idx, gt_idx = indices[b]
            
#             # Create target
#             target = torch.zeros_like(cls_logits[b])
#             if len(pred_idx) > 0:
#                 target[pred_idx] = 1.0
            
#             # Focal loss
#             cls_probs = torch.sigmoid(cls_logits[b])
            
#             # Positive loss
#             pos_loss = -target * torch.pow(1 - cls_probs, 2) * torch.log(cls_probs + 1e-8)
            
#             # Negative loss (penalize high-confidence false positives)
#             neg_loss = -(1 - target) * torch.pow(cls_probs, 2) * torch.log(1 - cls_probs + 1e-8)
            
#             loss = (pos_loss + neg_loss).mean()
#             losses.append(loss)
        
#         return torch.stack(losses).mean()
    
#     def _regression_loss(self, pred_positions, indices, gt_points_list):
#         """Smooth L1 loss for matched predictions."""
#         losses = []
        
#         for b in range(len(gt_points_list)):
#             pred_idx, gt_idx = indices[b]
            
#             if len(pred_idx) == 0:
#                 continue
            
#             pred_matched = pred_positions[b, pred_idx]
#             gt_matched = gt_points_list[b][gt_idx]
            
#             loss = F.smooth_l1_loss(pred_matched, gt_matched)
#             losses.append(loss)
        
#         if len(losses) == 0:
#             return pred_positions.sum() * 0
        
#         return torch.stack(losses).mean()
    
#     def _false_positive_loss(self, pred_positions, cls_logits, indices, gt_points_list):
#         """
#         Explicit false positive penalty.
#         Penalizes high-confidence predictions far from any GT.
#         """
#         B = cls_logits.shape[0]
#         losses = []
        
#         for b in range(B):
#             pred_idx, gt_idx = indices[b]
#             gt_pts = gt_points_list[b]
            
#             if len(gt_pts) == 0:
#                 # No GT: all predictions are false positives
#                 cls_probs = torch.sigmoid(cls_logits[b])
#                 fp_loss = (cls_probs ** 2).mean()
#                 losses.append(fp_loss)
#                 continue
            
#             # Get unmatched predictions
#             N_queries = cls_logits.shape[1]
#             matched_mask = torch.zeros(N_queries, dtype=torch.bool, device=cls_logits.device)
#             if len(pred_idx) > 0:
#                 matched_mask[pred_idx] = True
            
#             unmatched_mask = ~matched_mask
#             unmatched_positions = pred_positions[b, unmatched_mask]
#             unmatched_scores = torch.sigmoid(cls_logits[b, unmatched_mask])
            
#             if len(unmatched_positions) == 0:
#                 losses.append(torch.tensor(0.0, device=cls_logits.device))
#                 continue
            
#             # Distance to nearest GT
#             dists = torch.cdist(unmatched_positions, gt_pts)
#             min_dists = dists.min(dim=1)[0]
            
#             # Penalty for far predictions with high confidence
#             distance_threshold = 2.0  # meters
#             is_far = (min_dists > distance_threshold).float()
#             normalized_dist = torch.clamp(min_dists / 10.0, 0, 1)
            
#             # FP penalty = confidence² × is_far × distance
#             fp_penalty = (unmatched_scores ** 2) * is_far * normalized_dist
#             losses.append(fp_penalty.mean())
        
#         return torch.stack(losses).mean()
    
#     def _visualize_training(self, pred, gt_points_list, indices, losses):
#         """Visualize training progress."""
#         try:
#             B = len(gt_points_list)
#             s_dir = os.path.join(self.vis_dir, f"iter_{self.vis_iter:06d}")
#             os.makedirs(s_dir, exist_ok=True)
            
#             pred_positions = pred['pred_positions']
#             cls_scores = torch.sigmoid(pred['cls_logits'])
#             bev_feat = pred.get('bev_feat', None)
            
#             # Visualize first sample in batch
#             b = 0
            
#             # 1. BEV features
#             if bev_feat is not None:
#                 bev_path = os.path.join(s_dir, f"batch{b}_bev_features.png")
#                 visualize_bev(bev_feat[b], bev_path)
            
#             # 2. Get matched predictions
#             pred_idx, gt_idx = indices[b]
#             if len(pred_idx) > 0:
#                 matched_pred = pred_positions[b, pred_idx]
#                 matched_scores = cls_scores[b, pred_idx]
#             else:
#                 matched_pred = torch.zeros((0, 3), device=pred_positions.device)
#                 matched_scores = torch.zeros((0,), device=pred_positions.device)
            
#             # 3. Point cloud visualization
#             pts_path = os.path.join(s_dir, f"batch{b}_point_cloud.png")
#             title = f"Iter {self.vis_iter} | Pred: {len(matched_pred)} | GT: {len(gt_points_list[b])}"
#             visualize_point_cloud_3d(matched_pred, gt_points_list[b], pts_path, 
#                                     matched_scores, title)
            
#             # 4. Query distribution
#             query_path = os.path.join(s_dir, f"batch{b}_queries.png")
#             visualize_query_distribution(pred_positions[b], cls_scores[b], query_path,
#                                         title=f"Iter {self.vis_iter} - Query Distribution")
            
#             # 5. Loss components
#             loss_path = os.path.join(s_dir, f"losses.png")
#             visualize_loss_components(losses, loss_path,
#                                      title=f"Iter {self.vis_iter} - Loss Components")
            
#             # 6. Save statistics
#             stats_path = os.path.join(s_dir, f"stats.txt")
#             with open(stats_path, 'w') as f:
#                 f.write(f"=== Iteration {self.vis_iter} Statistics ===\n\n")
#                 f.write(f"Predictions:\n")
#                 f.write(f"  Total queries: {self.num_queries}\n")
#                 f.write(f"  High confidence (>0.5): {(cls_scores[b] > 0.5).sum().item()}\n")
#                 f.write(f"  Matched predictions: {len(pred_idx)}\n")
#                 f.write(f"  Mean confidence (matched): {matched_scores.mean().item():.4f}\n")
#                 f.write(f"  Mean confidence (all): {cls_scores[b].mean().item():.4f}\n")
#                 f.write(f"\nGround Truth:\n")
#                 f.write(f"  Total GT points: {len(gt_points_list[b])}\n")
#                 f.write(f"\nLosses:\n")
#                 for k, v in losses.items():
#                     val = v.item() if isinstance(v, torch.Tensor) else v
#                     f.write(f"  {k}: {val:.6f}\n")
            
#             print(f"Visualization saved: {s_dir}")
            
#         except Exception as e:
#             print(f"Warning: Visualization failed: {e}")
#             import traceback
#             traceback.print_exc()


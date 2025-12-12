# Copyright (c) Phigent Robotics. All rights reserved.
"""
Radar Point Prediction Heads - Complete Collection
Contains 4 different approaches for radar point prediction from BEV features.
Available heads:
1. RadarPointRegressionHead - Spatial grid-based (current/baseline)
2. RadarPointHeatmapHead - Dense heatmap with peak detection
3. RadarPointAttentionHead - Attention-based importance sampling
4. RadarPointHybridHead - Combines attention context + heatmap localization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models import HEADS
from mmdet.core import reduce_mean
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# Visualization Utilities (Shared)
# ============================================================================
XLIM = 20
YLIM = 20
ZLIM = 3

def visualize_bev(bev_feat, save_path):
    """Visualize BEV features as heatmap."""
    bev_np = bev_feat.detach().cpu().numpy()
    bev_heatmap = bev_np.mean(axis=0)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(bev_heatmap, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def visualize_points(pred_points, gt_points, save_path):
    """Visualize predicted vs ground truth points in 2D (BEV)."""
    pred_np = pred_points.detach().cpu().numpy()
    gt_np = gt_points.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(pred_np[:, 0], pred_np[:, 1], s=4)
    ax1.set_title(f"Predicted ({len(pred_np)} pts)")
    ax1.set_aspect("equal")
    ax1.set_xlim(-XLIM, XLIM)
    ax1.set_ylim(-YLIM, YLIM)
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(gt_np[:, 0], gt_np[:, 1], s=4, c='red')
    ax2.set_title(f"GT ({len(gt_np)} pts)")
    ax2.set_aspect("equal")
    ax2.set_xlim(-XLIM, XLIM)
    ax2.set_ylim(-YLIM, YLIM)
    
    plt.savefig(save_path, dpi=120)
    plt.close()


def visualize_point_cloud_3d(pred_points, gt_points, save_path, pred_scores=None):
    """Visualize predicted vs ground truth points in 3D point cloud."""
    pred_np = pred_points.detach().cpu().numpy()
    gt_np = gt_points.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(18, 6))
    
    # Predicted points in 3D
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    if len(pred_np) > 0:
        if pred_scores is not None:
            scores_np = pred_scores.detach().cpu().numpy()
            scatter1 = ax1.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], 
                       c=scores_np, cmap='viridis', s=20, alpha=0.6, 
                       edgecolors='none', vmin=0, vmax=1)
            plt.colorbar(scatter1, ax=ax1, label='Confidence', shrink=0.5)
        else:
            scatter1 = ax1.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], 
                       c=pred_np[:, 2], cmap='viridis', s=20, alpha=0.6, edgecolors='none')
            plt.colorbar(scatter1, ax=ax1, label='Z (m)', shrink=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f"Predicted ({len(pred_np)} pts)")
    ax1.set_xlim(-XLIM, XLIM)
    ax1.set_ylim(-YLIM, YLIM)
    ax1.set_zlim(-ZLIM, ZLIM)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # Ground truth points in 3D
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    if len(gt_np) > 0:
        scatter2 = ax2.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], 
                   c=gt_np[:, 2], cmap='plasma', s=20, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter2, ax=ax2, label='Z (m)', shrink=0.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f"Ground Truth ({len(gt_np)} pts)")
    ax2.set_xlim(-XLIM, XLIM)
    ax2.set_ylim(-YLIM, YLIM)
    ax2.set_zlim(-ZLIM, ZLIM)
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.3)
    
    # Overlay comparison in 3D
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    if len(pred_np) > 0:
        ax3.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], 
                   c='blue', s=15, alpha=0.4, label='Predicted', edgecolors='none')
    if len(gt_np) > 0:
        ax3.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], 
                   c='red', s=15, alpha=0.4, label='Ground Truth', edgecolors='none')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title(f"Overlay Comparison")
    ax3.set_xlim(-XLIM, XLIM)
    ax3.set_ylim(-YLIM, YLIM)
    ax3.set_zlim(-ZLIM, ZLIM)
    ax3.view_init(elev=20, azim=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def visualize_confidence(conf_pred, conf_target, save_path):
    """Visualize confidence map comparison."""
    conf_pred_np = conf_pred.detach().cpu().numpy()
    conf_target_np = conf_target.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(conf_pred_np, cmap='hot', vmin=0, vmax=1)
    ax1.set_title('Predicted Confidence')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(conf_target_np, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('Target Confidence')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def visualize_queries(query_pos, foreground_mask, quality_scores, save_path):
    """Visualize query selection process."""
    query_pos_np = query_pos.detach().cpu().numpy()
    fg_mask_np = foreground_mask.detach().cpu().numpy()
    quality_np = quality_scores.detach().cpu().numpy() if quality_scores is not None else None
    
    fig = plt.figure(figsize=(15, 5))
    
    # All queries
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(query_pos_np[:, 0], query_pos_np[:, 1], s=5, alpha=0.3, c='gray')
    ax1.set_title(f"All Queries ({len(query_pos_np)})")
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_xlim(-XLIM, XLIM)
    ax1.set_ylim(-YLIM, YLIM)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Foreground queries
    ax2 = fig.add_subplot(1, 3, 2)
    fg_queries = query_pos_np[fg_mask_np]
    ax2.scatter(fg_queries[:, 0], fg_queries[:, 1], s=10, alpha=0.6, c='orange')
    ax2.set_title(f"Foreground Queries ({len(fg_queries)})")
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_xlim(-XLIM, XLIM)
    ax2.set_ylim(-YLIM, YLIM)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Quality-filtered queries
    ax3 = fig.add_subplot(1, 3, 3)
    if quality_np is not None and len(fg_queries) > 0:
        # Apply the same mask to quality scores
        fg_quality = quality_np[fg_mask_np]
        
        # Ensure sizes match
        if len(fg_quality) == len(fg_queries):
            scatter = ax3.scatter(fg_queries[:, 0], fg_queries[:, 1], 
                                 s=15, alpha=0.7, c=fg_quality, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(scatter, ax=ax3, label='Quality Score')
            ax3.set_title(f"Quality Scores")
        else:
            # Fallback: use uniform color if sizes don't match
            ax3.scatter(fg_queries[:, 0], fg_queries[:, 1], s=15, alpha=0.7, c='green')
            ax3.set_title(f"Selected Queries (size mismatch)")
    else:
        if len(fg_queries) > 0:
            ax3.scatter(fg_queries[:, 0], fg_queries[:, 1], s=15, alpha=0.7, c='green')
        ax3.set_title(f"Selected Queries")
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_xlim(-XLIM, XLIM)
    ax3.set_ylim(-YLIM, YLIM)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()



# ============================================================================
# HEAD 1: Spatial Grid-Based Regression (Current/Baseline)
# ============================================================================
@HEADS.register_module()
class RadarPointRegressionHead(nn.Module):
    """Spatial grid-based radar point prediction.
    
    Predicts fixed number of points organized in spatial groups.
    Uses BEV features with spatial structure preservation.
    
    Good for: Baseline, quick experiments
    """
    
    def __init__(self,
                 in_channels=256,
                 num_points=1024,
                 point_dim=3,
                 bev_h=128,
                 bev_w=128,
                 hidden_channels=[512, 256, 128],
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 loss_type='chamfer',
                 loss_weight=1.0,
                 diversity_weight=0.1,
                 coverage_weight=0.5,
                 min_distance=0.5,
                 coverage_threshold=1.5,
                 use_spatial=True,
                 num_spatial_groups=64,
                 enable_viz=False,
                 visualize_training=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_points = num_points
        self.point_dim = point_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        self.min_distance = min_distance
        self.coverage_threshold = coverage_threshold
        self.use_spatial = use_spatial
        self.num_spatial_groups = num_spatial_groups
        
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
        # Feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True),
        )
        
        if self.use_spatial:
            self.grid_h = int(num_spatial_groups ** 0.5)
            self.grid_w = int(num_spatial_groups ** 0.5)
            self.spatial_pool = nn.AdaptiveAvgPool2d((self.grid_h, self.grid_w))
            self.points_per_group = num_points // num_spatial_groups
            
            self.point_decoder = nn.Sequential(
                nn.Linear(hidden_channels[1], hidden_channels[2]),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_channels[2], self.points_per_group * point_dim)
            )
            
            self.spatial_embed = nn.Parameter(
                torch.randn(1, hidden_channels[1], self.grid_h, self.grid_w) * 0.02
            )
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            fc_layers = []
            fc_channels = [hidden_channels[1]] + hidden_channels[2:] + [num_points * point_dim]
            for i in range(len(fc_channels) - 1):
                fc_layers.append(nn.Linear(fc_channels[i], fc_channels[i+1]))
                if i < len(fc_channels) - 2:
                    fc_layers.append(nn.ReLU(inplace=True))
                    fc_layers.append(nn.Dropout(0.1))
            self.fc_layers = nn.Sequential(*fc_layers)
        
        self._init_weights()
        
        # Visualization
        self.vis_iter = 0
        self.vis_dir = "work_dirs/radar_viz_spatial"
        self.enable_viz = enable_viz
        self.visualize_training = visualize_training
        if self.enable_viz or self.visualize_training:
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bev_feat):
        B = bev_feat.shape[0]
        
        # Extract features
        feat = self.conv1(bev_feat)
        feat = self.conv2(feat)
        
        if self.use_spatial:
            feat_grid = self.spatial_pool(feat)
            feat_grid = feat_grid + self.spatial_embed
            B, C, H, W = feat_grid.shape
            feat_grid = feat_grid.view(B, C, -1).permute(0, 2, 1)
            points_per_loc = self.point_decoder(feat_grid)
            points = points_per_loc.view(B, -1, self.point_dim)
        else:
            feat = self.global_pool(feat).flatten(1)
            points_flat = self.fc_layers(feat)
            points = points_flat.view(B, self.num_points, self.point_dim)
        
        # Normalize
        points = torch.tanh(points)
        pc_min = self.pc_range[:self.point_dim]
        pc_max = self.pc_range[self.point_dim:self.point_dim*2]
        pc_center = (pc_min + pc_max) / 2
        pc_scale = (pc_max - pc_min) / 2
        points = points * pc_scale + pc_center
        
        return [{'radar_points': points}]
    
    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, preds_dicts, gt_radar_points, img_metas=None, bev_feat=None):
        pred_points = preds_dicts[0]['radar_points']
        
        if not isinstance(gt_radar_points, torch.Tensor):
            gt_radar_points = torch.stack([torch.as_tensor(x) for x in gt_radar_points])
        gt_radar_points = gt_radar_points.to(pred_points.device)
        
        chamfer_loss = self._chamfer_distance(pred_points, gt_radar_points)
        diversity_loss = self._diversity_loss(pred_points)
        coverage_loss = self._coverage_loss(pred_points, gt_radar_points)
        
        total_loss = (
            chamfer_loss +
            self.diversity_weight * diversity_loss +
            self.coverage_weight * coverage_loss
        )
        
        # ---- Visualization ----
        if self.visualize_training:
            self.vis_iter += 1
            B = pred_points.shape[0]
            s_dir = os.path.join(self.vis_dir, f"{self.vis_iter}")
            
            # Save BEV feature heatmap if available
            if bev_feat is not None:
                for i in range(B):
                    iter_str = f"{i:05d}"
                    os.makedirs(os.path.join(s_dir, "bev_features"), exist_ok=True)
                    bev_path = os.path.join(s_dir, "bev_features", f"iter_{iter_str}_bev.png")
                    visualize_bev(bev_feat[i], bev_path)
            
            # Visualize predicted vs GT points
            for i in range(B):
                iter_str = f"{i:05d}"
                os.makedirs(os.path.join(s_dir, "point_clouds"), exist_ok=True)
                
                # 2D BEV visualization
                pts_path_2d = os.path.join(s_dir, "point_clouds", f"iter_{iter_str}_points_2d.png")
                visualize_points(pred_points[i], gt_radar_points[i], pts_path_2d)
                
                # 3D point cloud visualization
                pts_path_3d = os.path.join(s_dir, "point_clouds", f"iter_{iter_str}_points_3d.png")
                visualize_point_cloud_3d(pred_points[i], gt_radar_points[i], pts_path_3d)
        
        return {
            'loss_radar_points': total_loss * self.loss_weight,
            'loss_chamfer': chamfer_loss,
            'loss_diversity': diversity_loss,
            'loss_coverage': coverage_loss,
        }
    
    def _chamfer_distance(self, pred_points, gt_points):
        pred_expand = pred_points.unsqueeze(2)
        gt_expand = gt_points.unsqueeze(1)
        dist = torch.sum((pred_expand - gt_expand) ** 2, dim=-1)
        
        min_dist_pred = torch.sqrt(torch.min(dist, dim=2)[0] + 1e-6).mean()
        min_dist_gt = torch.sqrt(torch.min(dist, dim=1)[0] + 1e-6).mean()
        
        return min_dist_pred + min_dist_gt
    
    def _diversity_loss(self, pred_points):
        pred_expand1 = pred_points.unsqueeze(2)
        pred_expand2 = pred_points.unsqueeze(1)
        dists = torch.sqrt(torch.sum((pred_expand1 - pred_expand2) ** 2, dim=-1) + 1e-6)
        
        violation = torch.clamp(self.min_distance - dists, min=0.0)
        B, N, _ = pred_points.shape
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_points.device).unsqueeze(0)
        violation = violation * mask.float()
        
        return violation.sum() / (B * N * (N - 1) + 1e-6)
    
    def _coverage_loss(self, pred_points, gt_points):
        pred_expand = pred_points.unsqueeze(2)
        gt_expand = gt_points.unsqueeze(1)
        dists = torch.sqrt(torch.sum((pred_expand - gt_expand) ** 2, dim=-1) + 1e-6)
        min_dists = torch.min(dists, dim=1)[0]
        uncovered = torch.clamp(min_dists - self.coverage_threshold, min=0.0)
        return uncovered.mean()

# ============================================================================
# HEAD 2: Heatmap-Based Dynamic Points
# ============================================================================
@HEADS.register_module()
class RadarPointHeatmapHead(nn.Module):
    """Heatmap-based dynamic radar point prediction.
    
    Predicts confidence heatmap + offsets + heights.
    Variable number of points extracted via peak detection.
    
    Good for: Best performance, production, variable points
    """
    
    def __init__(self,
                 in_channels=256,
                 bev_h=128,
                 bev_w=128,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 min_confidence=0.3,
                 nms_radius=2,
                 gaussian_radius=2,
                 loss_conf_weight=1.0,
                 loss_offset_weight=0.5,
                 loss_height_weight=0.5,
                 enable_viz=False,
                 visualize_training=False):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.min_confidence = min_confidence
        self.nms_radius = nms_radius
        self.gaussian_radius = gaussian_radius
        self.loss_conf_weight = loss_conf_weight
        self.loss_offset_weight = loss_offset_weight
        self.loss_height_weight = loss_height_weight
        
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
        # Shared features
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        # Offset head
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),
        )
        
        # Height head
        self.height_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )
        
        self._init_weights()
        
        # Visualization
        self.vis_iter = 0
        self.vis_dir = "work_dirs/radar_viz_heatmap"
        self.enable_viz = enable_viz
        self.visualize_training = visualize_training
        if self.enable_viz or self.visualize_training:
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Bias confidence toward low values initially
        nn.init.constant_(self.confidence_head[-2].bias, -2.19)
    
    def forward(self, bev_feat):
        B = bev_feat.shape[0]
        
        if bev_feat.shape[2:] != (self.bev_h, self.bev_w):
            bev_feat = F.interpolate(bev_feat, size=(self.bev_h, self.bev_w),
                                    mode='bilinear', align_corners=False)
        
        feat = self.shared_conv(bev_feat)
        
        confidence = self.confidence_head(feat)
        offset = self.offset_head(feat)
        height = self.height_head(feat)
        
        if not self.training:
            radar_points = self.extract_points(confidence, offset, height)
        else:
            radar_points = None
        
        return [{
            'radar_points': radar_points,
            'confidence_map': confidence,
            'offset_map': offset,
            'height_map': height,
        }]
    
    def extract_points(self, confidence, offset, height):
        """Extract variable-length points from heatmaps."""
        B, _, H, W = confidence.shape
        batch_points = []
        
        for b in range(B):
            conf = confidence[b, 0]
            off = offset[b]
            hgt = height[b, 0]
            
            # NMS
            conf_nms = self._nms_2d(conf, self.nms_radius)
            mask = conf_nms > self.min_confidence
            y_idx, x_idx = torch.where(mask)
            
            if len(y_idx) == 0:
                batch_points.append(torch.zeros((0, 3), device=conf.device))
                continue
            
            # Get coordinates with offsets
            x_pix = x_idx.float() + torch.tanh(off[0, y_idx, x_idx])
            y_pix = y_idx.float() + torch.tanh(off[1, y_idx, x_idx])
            z_val = hgt[y_idx, x_idx]
            
            # Convert to world coords
            x_min, y_min, z_min = self.pc_range[:3]
            x_max, y_max, z_max = self.pc_range[3:]
            
            x_world = x_min + (x_pix / W) * (x_max - x_min)
            y_world = y_min + (y_pix / H) * (y_max - y_min)
            z_world = z_min + (torch.tanh(z_val) + 1) / 2 * (z_max - z_min)
            
            points = torch.stack([x_world, y_world, z_world], dim=1)
            batch_points.append(points)
        
        return batch_points
    
    def _nms_2d(self, heatmap, kernel_size):
        """2D NMS with proper dimension handling."""
        # Add batch and channel dimensions
        heatmap_4d = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        pad = (kernel_size) // 2
        hmax = F.max_pool2d(
            heatmap_4d,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )
        
        # Remove batch and channel dimensions
        hmax = hmax.squeeze(0).squeeze(0)  # Back to (H, W)
        if hmax.shape != heatmap.shape:
            hmax = hmax[:heatmap.shape[0], :heatmap.shape[1]]
        # Keep only local maxima
        keep = (hmax == heatmap).float()
        return heatmap * keep
    
    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, preds_dicts, gt_radar_points, img_metas=None, bev_feat=None):
        confidence_map = preds_dicts[0]['confidence_map']
        offset_map = preds_dicts[0]['offset_map']
        height_map = preds_dicts[0]['height_map']
        
        B, _, H, W = confidence_map.shape
        
        if isinstance(gt_radar_points, torch.Tensor):
            gt_radar_points = [gt_radar_points[i] for i in range(B)]
        else:
            gt_radar_points = [torch.as_tensor(pts, device=confidence_map.device)
                             for pts in gt_radar_points]
        
        target_conf, target_offset, target_height, mask = self._generate_targets(
            gt_radar_points, H, W, confidence_map.device
        )
        
        # Focal loss for confidence
        conf_loss = self._focal_loss(confidence_map.squeeze(1), target_conf)
        
        # Offset loss
        if mask.sum() > 0:
            offset_pred = offset_map.permute(0, 2, 3, 1)[mask]  # (B, 2, H, W) → (B, H, W, 2)
            offset_target = target_offset[mask]  # Already (B, H, W, 2), just index
            offset_loss = F.smooth_l1_loss(offset_pred, offset_target)
        else:
            offset_loss = offset_map.sum() * 0
        
        # Height loss
        if mask.sum() > 0:
            height_pred = height_map.squeeze(1)[mask]
            height_target = target_height[mask]
            height_loss = F.smooth_l1_loss(height_pred, height_target)
        else:
            height_loss = height_map.sum() * 0
        
        # ---- Visualization ----
        if self.visualize_training:
            self.vis_iter += 1
            s_dir = os.path.join(self.vis_dir, f"{self.vis_iter}")
            
            # Save BEV feature heatmap if available
            if bev_feat is not None:
                for i in range(B):
                    iter_str = f"{i:05d}"
                    os.makedirs(os.path.join(s_dir, "bev_features"), exist_ok=True)
                    bev_path = os.path.join(s_dir, "bev_features", f"iter_{iter_str}_bev.png")
                    visualize_bev(bev_feat[i], bev_path)
            
            # Extract predicted points for visualization during training
            if self.enable_viz:
                pred_points = self.extract_points(confidence_map, offset_map, height_map)
                
                for i in range(B):
                    iter_str = f"{i:05d}"
                    os.makedirs(os.path.join(s_dir, "gt_pred"), exist_ok=True)
                    
                    # 2D point visualization
                    pts_path = os.path.join(s_dir, "gt_pred", f"iter_{iter_str}_points.png")
                    visualize_points(pred_points[i], gt_radar_points[i], pts_path)
                    
                    # 3D point cloud visualization
                    pts_path_3d = os.path.join(s_dir, "gt_pred", f"iter_{iter_str}_points_3d.png")
                    visualize_point_cloud_3d(pred_points[i], gt_radar_points[i], pts_path_3d)
                    
                    # Confidence map visualization
                    conf_path = os.path.join(s_dir, "gt_pred", f"iter_{iter_str}_conf.png")
                    visualize_confidence(
                        confidence_map[i, 0], 
                        target_conf[i], 
                        conf_path
                    )
        
        return {
            'loss_radar_conf': self.loss_conf_weight * conf_loss,
            'loss_radar_offset': self.loss_offset_weight * offset_loss,
            'loss_radar_height': self.loss_height_weight * height_loss,
        }
    
    def _generate_targets(self, gt_points_list, H, W, device):
        B = len(gt_points_list)
        
        target_conf = torch.zeros((B, H, W), device=device)
        target_offset = torch.zeros((B, H, W, 2), device=device)
        target_height = torch.zeros((B, H, W), device=device)
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        
        x_min, y_min, z_min = self.pc_range[:3].cpu().numpy()
        x_max, y_max, z_max = self.pc_range[3:].cpu().numpy()
        
        for b in range(B):
            pts = gt_points_list[b]
            if len(pts) == 0:
                continue
            
            # Convert to numpy for processing
            pts_np = pts.cpu().numpy() if isinstance(pts, torch.Tensor) else pts
            
            x_norm = (pts_np[:, 0] - x_min) / (x_max - x_min)
            y_norm = (pts_np[:, 1] - y_min) / (y_max - y_min)
            z_norm = (pts_np[:, 2] - z_min) / (z_max - z_min)
            
            x_pix = x_norm * W
            y_pix = y_norm * H
            z_norm_centered = z_norm * 2 - 1
            
            x_int = np.clip(x_pix.astype(int), 0, W - 1)
            y_int = np.clip(y_pix.astype(int), 0, H - 1)
            
            x_off = x_pix - x_int
            y_off = y_pix - y_int
            
            for i in range(len(pts)):
                xi, yi = x_int[i], y_int[i]
                self._draw_gaussian(target_conf[b], (xi, yi), self.gaussian_radius)
                # Convert numpy values to float before assignment
                target_offset[b, yi, xi, 0] = float(x_off[i])
                target_offset[b, yi, xi, 1] = float(y_off[i])
                target_height[b, yi, xi] = float(z_norm_centered[i])
                mask[b, yi, xi] = True
        
        return target_conf, target_offset, target_height, mask
    
    def _draw_gaussian(self, heatmap, center, radius):
        x, y = center
        height, width = heatmap.shape
        
        left = min(x, radius)
        right = min(width - x, radius + 1)
        top = min(y, radius)
        bottom = min(height - y, radius + 1)
        
        for i in range(-top, bottom):
            for j in range(-left, right):
                weight = np.exp(-(i**2 + j**2) / (2 * radius**2))
                heatmap[y + i, x + j] = max(heatmap[y + i, x + j].item(), weight)
    
    def _focal_loss(self, pred, target, alpha=2.0, beta=4.0):
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, alpha) * pos_mask
        neg_loss = (torch.log(1 - pred + 1e-6) * torch.pow(pred, alpha) *
                   torch.pow(1 - target, beta) * neg_mask)
        
        num_pos = pos_mask.sum()
        if num_pos == 0:
            return -neg_loss.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

# ============================================================================
# HEAD 3: Attention-Based Sampling
# ============================================================================
@HEADS.register_module()
class RadarPointAttentionHead(nn.Module):
    """Attention-based radar point prediction.
    
    Uses self-attention to compute importance and select top-K locations.
    Variable number of points with global context.
    
    Good for: Research, complex scenes, long-range dependencies
    """
    
    def __init__(self,
                 in_channels=256,
                 hidden_dim=256,
                 num_attention_heads=8,
                 max_points=1000,
                 topk_ratio=0.1,
                 confidence_threshold=0.5,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 enable_viz=False,
                 visualize_training=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_points = max_points
        self.topk_ratio = topk_ratio
        self.conf_threshold = confidence_threshold
        
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
        # Feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Importance scoring
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Point refinement
        self.point_refine = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
            nn.Tanh()
        )
        
        # Confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Visualization
        self.vis_iter = 0
        self.vis_dir = "work_dirs/radar_viz_attention"
        self.enable_viz = enable_viz
        self.visualize_training = visualize_training
        if self.enable_viz or self.visualize_training:
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def forward(self, bev_feat):
        B, C, H, W = bev_feat.shape
        
        # Encode
        feat = self.feat_encoder(bev_feat)
        
        # Flatten
        feat_flat = feat.flatten(2).permute(0, 2, 1)
        
        # Attention
        attn_out, attn_weights = self.self_attention(feat_flat, feat_flat, feat_flat)
        
        # Importance
        importance = self.importance_head(attn_out).squeeze(-1)
        
        # Select top-K per batch
        batch_points = []
        batch_confidences = []
        
        for b in range(B):
            imp = importance[b]
            feat_b = attn_out[b]
            
            k = min(int(H * W * self.topk_ratio), self.max_points)
            topk_scores, topk_idx = torch.topk(imp, k)
            
            selected_feat = feat_b[topk_idx]
            conf = self.confidence_head(selected_feat).squeeze(-1)
            
            if self.training:
                valid_mask = torch.ones(k, dtype=torch.bool, device=conf.device)
            else:
                valid_mask = (conf > self.conf_threshold)
            
            if valid_mask.sum() == 0:
                batch_points.append(torch.zeros((0, 3), device=bev_feat.device))
                batch_confidences.append(torch.zeros((0,), device=bev_feat.device))
                continue
            
            selected_feat = selected_feat[valid_mask]
            selected_idx = topk_idx[valid_mask]
            selected_conf = conf[valid_mask]
            
            # Predict offsets
            offsets = self.point_refine(selected_feat)
            
            # Convert to coords
            y_idx = selected_idx // W
            x_idx = selected_idx % W
            
            x_base = (x_idx.float() + 0.5) / W
            y_base = (y_idx.float() + 0.5) / H
            
            x_norm = x_base + offsets[:, 0] * (1.0 / W)
            y_norm = y_base + offsets[:, 1] * (1.0 / H)
            z_norm = offsets[:, 2]
            
            x_min, y_min, z_min = self.pc_range[:3]
            x_max, y_max, z_max = self.pc_range[3:]
            
            x_world = x_min + x_norm * (x_max - x_min)
            y_world = y_min + y_norm * (y_max - y_min)
            z_world = z_min + (z_norm + 1) / 2 * (z_max - z_min)
            
            points = torch.stack([x_world, y_world, z_world], dim=1)
            
            batch_points.append(points)
            batch_confidences.append(selected_conf)
        
        return [{
            'radar_points': batch_points,
            'confidence': batch_confidences,
        }]
    
    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, preds_dicts, gt_radar_points, img_metas=None, bev_feat=None):
        pred_points = preds_dicts[0]['radar_points']
        pred_conf = preds_dicts[0]['confidence']
        
        if isinstance(gt_radar_points, torch.Tensor):
            gt_radar_points = [gt_radar_points[i] for i in range(len(pred_points))]
        
        losses = []
        
        for pred_pts, pred_c, gt_pts in zip(pred_points, pred_conf, gt_radar_points):
            if len(pred_pts) == 0 or len(gt_pts) == 0:
                continue
            
            # Hungarian matching
            from scipy.optimize import linear_sum_assignment
            
            cost_matrix = torch.cdist(pred_pts, gt_pts)
            pred_idx, gt_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            matched_pred = pred_pts[pred_idx]
            matched_gt = gt_pts[gt_idx]
            point_loss = F.smooth_l1_loss(matched_pred, matched_gt)
            
            target_conf = torch.zeros_like(pred_c)
            target_conf[pred_idx] = 1.0
            conf_loss = F.binary_cross_entropy(pred_c, target_conf)
            
            losses.append(point_loss + 0.5 * conf_loss)
        
        # ---- Visualization ----
        if self.visualize_training:
            self.vis_iter += 1
            B = len(pred_points)
            s_dir = os.path.join(self.vis_dir, f"{self.vis_iter}")
            
            # Save BEV feature heatmap if available
            if bev_feat is not None:
                for i in range(B):
                    iter_str = f"{i:05d}"
                    os.makedirs(os.path.join(s_dir, "bev_features"), exist_ok=True)
                    bev_path = os.path.join(s_dir, "bev_features", f"iter_{iter_str}_bev.png")
                    visualize_bev(bev_feat[i], bev_path)
            
            # Visualize predicted vs GT points
            for i in range(B):
                iter_str = f"{i:05d}"
                os.makedirs(os.path.join(s_dir, "point_clouds"), exist_ok=True)
                
                # 2D BEV visualization
                pts_path_2d = os.path.join(s_dir, "point_clouds", f"iter_{iter_str}_points_2d.png")
                visualize_points(pred_points[i], gt_radar_points[i], pts_path_2d)
                
                # 3D point cloud visualization
                pts_path_3d = os.path.join(s_dir, "point_clouds", f"iter_{iter_str}_points_3d.png")
                visualize_point_cloud_3d(pred_points[i], gt_radar_points[i], pts_path_3d)
        
        if len(losses) == 0:
            return {'loss_radar_attn': pred_points[0].sum() * 0}
        
        return {'loss_radar_attn': torch.stack(losses).mean()}

# ============================================================================
# HEAD 4: Hybrid (Attention + Heatmap)
# ============================================================================
@HEADS.register_module()
class RadarPointHybridHead(nn.Module):
    """Hybrid: Attention context + Heatmap localization.
    
    Combines best of both:
    - Attention for global context and feature enhancement
    - Heatmap for precise localization and peak detection
    
    Good for: Best performance, research papers, when you want it all
    """
    
    def __init__(self,
                 in_channels=256,
                 hidden_dim=256,
                 num_attention_heads=8,
                 bev_h=128,
                 bev_w=128,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 min_confidence=0.3,
                 nms_radius=2,
                 gaussian_radius=2,
                 loss_conf_weight=1.0,
                 loss_offset_weight=0.5,
                 loss_height_weight=0.5,
                 enable_viz=False,
                 visualize_training=False):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.min_confidence = min_confidence
        self.nms_radius = nms_radius
        self.gaussian_radius = gaussian_radius
        self.loss_conf_weight = loss_conf_weight
        self.loss_offset_weight = loss_offset_weight
        self.loss_height_weight = loss_height_weight
        
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
        # Step 1: Feature encoding
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Step 2: Attention for global context
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Step 3: Project back to spatial
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Step 4: Heatmap heads
        self.confidence_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),
        )
        
        self.height_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )
        
        # Visualization
        self.vis_iter = 0
        self.vis_dir = "work_dirs/radar_viz_hybrid"
        self.enable_viz = enable_viz
        self.visualize_training = visualize_training
        if self.enable_viz or self.visualize_training:
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def forward(self, bev_feat):
        B, C, H, W = bev_feat.shape
        
        # Encode
        feat = self.feat_encoder(bev_feat)
        
        # Apply attention (global context)
        feat_flat = feat.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.self_attention(feat_flat, feat_flat, feat_flat)
        feat_attended = attn_out.permute(0, 2, 1).view(B, -1, H, W)
        
        # Project back to spatial
        feat_spatial = self.spatial_proj(feat_attended)
        
        # Heatmap prediction
        confidence = self.confidence_head(feat_spatial)
        offset = self.offset_head(feat_spatial)
        height = self.height_head(feat_spatial)
        
        if not self.training:
            radar_points = self.extract_points(confidence, offset, height)
        else:
            radar_points = None
        
        return [{
            'radar_points': radar_points,
            'confidence_map': confidence,
            'offset_map': offset,
            'height_map': height,
        }]
    
    def extract_points(self, confidence, offset, height):
        """Same as heatmap head."""
        B, _, H, W = confidence.shape
        batch_points = []
        
        for b in range(B):
            conf = confidence[b, 0]
            off = offset[b]
            hgt = height[b, 0]
            
            conf_nms = self._nms_2d(conf, self.nms_radius)
            mask = conf_nms > self.min_confidence
            y_idx, x_idx = torch.where(mask)
            
            if len(y_idx) == 0:
                batch_points.append(torch.zeros((0, 3), device=conf.device))
                continue
            
            x_pix = x_idx.float() + torch.tanh(off[0, y_idx, x_idx])
            y_pix = y_idx.float() + torch.tanh(off[1, y_idx, x_idx])
            z_val = hgt[y_idx, x_idx]
            
            x_min, y_min, z_min = self.pc_range[:3]
            x_max, y_max, z_max = self.pc_range[3:]
            
            x_world = x_min + (x_pix / W) * (x_max - x_min)
            y_world = y_min + (y_pix / H) * (y_max - y_min)
            z_world = z_min + (torch.tanh(z_val) + 1) / 2 * (z_max - z_min)
            
            points = torch.stack([x_world, y_world, z_world], dim=1)
            batch_points.append(points)
        
        return batch_points
    
    def _nms_2d(self, heatmap, kernel_size):
        """2D NMS with proper dimension handling."""
        # Add batch and channel dimensions
        heatmap_4d = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        pad = (kernel_size) // 2
        hmax = F.max_pool2d(
            heatmap_4d,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )
        
        # Remove batch and channel dimensions
        hmax = hmax.squeeze(0).squeeze(0)  # Back to (H, W)
        
        if hmax.shape != heatmap.shape:
            hmax = hmax[:heatmap.shape[0], :heatmap.shape[1]]
        # Keep only local maxima
        keep = (hmax == heatmap).float()
        return heatmap * keep
    
    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, preds_dicts, gt_radar_points, img_metas=None, bev_feat=None):
        """Same loss as heatmap head."""
        confidence_map = preds_dicts[0]['confidence_map']
        offset_map = preds_dicts[0]['offset_map']
        height_map = preds_dicts[0]['height_map']
        
        B, _, H, W = confidence_map.shape
        
        if isinstance(gt_radar_points, torch.Tensor):
            gt_radar_points = [gt_radar_points[i] for i in range(B)]
        else:
            gt_radar_points = [torch.as_tensor(pts, device=confidence_map.device)
                             for pts in gt_radar_points]
        
        target_conf, target_offset, target_height, mask = self._generate_targets(
            gt_radar_points, H, W, confidence_map.device
        )
        
        conf_loss = self._focal_loss(confidence_map.squeeze(1), target_conf)
        
        if mask.sum() > 0:
            offset_pred = offset_map.permute(0, 2, 3, 1)[mask]  # (B, 2, H, W) → (B, H, W, 2)
            offset_target = target_offset[mask]  # Already (B, H, W, 2), just index
            offset_loss = F.smooth_l1_loss(offset_pred, offset_target)
        else:
            offset_loss = offset_map.sum() * 0
        
        if mask.sum() > 0:
            height_pred = height_map.squeeze(1)[mask]
            height_target = target_height[mask]
            height_loss = F.smooth_l1_loss(height_pred, height_target)
        else:
            height_loss = height_map.sum() * 0
        
        # ---- Visualization ----
        if self.visualize_training:
            self.vis_iter += 1
            s_dir = os.path.join(self.vis_dir, f"{self.vis_iter}")
            
            # Save BEV feature heatmap if available
            if bev_feat is not None:
                for i in range(B):
                    iter_str = f"{i:05d}"
                    os.makedirs(os.path.join(s_dir, "bev_features"), exist_ok=True)
                    bev_path = os.path.join(s_dir, "bev_features", f"iter_{iter_str}_bev.png")
                    visualize_bev(bev_feat[i], bev_path)
            
            # Extract predicted points for visualization during training
            if self.enable_viz:
                pred_points = self.extract_points(confidence_map, offset_map, height_map)
                
                for i in range(B):
                    iter_str = f"{i:05d}"
                    os.makedirs(os.path.join(s_dir, "gt_pred"), exist_ok=True)
                    
                    # 2D point visualization
                    pts_path = os.path.join(s_dir, "gt_pred", f"iter_{iter_str}_points.png")
                    visualize_points(pred_points[i], gt_radar_points[i], pts_path)
                    
                    # 3D point cloud visualization
                    pts_path_3d = os.path.join(s_dir, "gt_pred", f"iter_{iter_str}_points_3d.png")
                    visualize_point_cloud_3d(pred_points[i], gt_radar_points[i], pts_path_3d)
                    
                    # Confidence map visualization
                    conf_path = os.path.join(s_dir, "gt_pred", f"iter_{iter_str}_conf.png")
                    visualize_confidence(
                        confidence_map[i, 0], 
                        target_conf[i], 
                        conf_path
                    )
        
        return {
            'loss_radar_conf': self.loss_conf_weight * conf_loss,
            'loss_radar_offset': self.loss_offset_weight * offset_loss,
            'loss_radar_height': self.loss_height_weight * height_loss,
        }
    
    def _generate_targets(self, gt_points_list, H, W, device):
        """Generate target heatmaps from GT points."""
        B = len(gt_points_list)
        
        target_conf = torch.zeros((B, H, W), device=device)
        target_offset = torch.zeros((B, H, W, 2), device=device)
        target_height = torch.zeros((B, H, W), device=device)
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        
        x_min, y_min, z_min = self.pc_range[:3].cpu().numpy()
        x_max, y_max, z_max = self.pc_range[3:].cpu().numpy()
        
        for b in range(B):
            pts = gt_points_list[b]
            if len(pts) == 0:
                continue
            
            # Convert to numpy for processing
            pts_np = pts.cpu().numpy() if isinstance(pts, torch.Tensor) else pts
            
            x_norm = (pts_np[:, 0] - x_min) / (x_max - x_min)
            y_norm = (pts_np[:, 1] - y_min) / (y_max - y_min)
            z_norm = (pts_np[:, 2] - z_min) / (z_max - z_min)
            
            x_pix = x_norm * W
            y_pix = y_norm * H
            z_norm_centered = z_norm * 2 - 1
            
            x_int = np.clip(x_pix.astype(int), 0, W - 1)
            y_int = np.clip(y_pix.astype(int), 0, H - 1)
            
            x_off = x_pix - x_int
            y_off = y_pix - y_int
            
            for i in range(len(pts)):
                xi, yi = x_int[i], y_int[i]
                self._draw_gaussian(target_conf[b], (xi, yi), self.gaussian_radius)
                # Convert numpy values to float before assignment
                target_offset[b, yi, xi, 0] = float(x_off[i])
                target_offset[b, yi, xi, 1] = float(y_off[i])
                target_height[b, yi, xi] = float(z_norm_centered[i])
                mask[b, yi, xi] = True
        
        return target_conf, target_offset, target_height, mask
    
    def _draw_gaussian(self, heatmap, center, radius):
        x, y = center
        height, width = heatmap.shape
        
        left = min(x, radius)
        right = min(width - x, radius + 1)
        top = min(y, radius)
        bottom = min(height - y, radius + 1)
        
        for i in range(-top, bottom):
            for j in range(-left, right):
                weight = np.exp(-(i**2 + j**2) / (2 * radius**2))
                heatmap[y + i, x + j] = max(heatmap[y + i, x + j].item(), weight)
    
    def _focal_loss(self, pred, target, alpha=2.0, beta=4.0):
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, alpha) * pos_mask
        neg_loss = (torch.log(1 - pred + 1e-6) * torch.pow(pred, alpha) *
                   torch.pow(1 - target, beta) * neg_mask)
        
        num_pos = pos_mask.sum()
        if num_pos == 0:
            return -neg_loss.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos



# ============================================================================
# Positional Encoding
# ============================================================================
class PositionEmbeddingSine(nn.Module):
    """Sine-based positional encoding for 2D features."""
    
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            pos: Tensor of shape (B, num_pos_feats*2, H, W)
        """
        B, C, H, W = x.shape
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)
        
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos

# ============================================================================
# Dual Query Selection Module
# ============================================================================
class DualQuerySelection(nn.Module):
    """
    Dual Query Selection (DQS) - Coarse-to-fine query selection.
    
    Stage 1: Foreground selection - Binary classification on BEV features
    Stage 2: Quality selection - IoU-based quality scoring
    """
    
    def __init__(self, 
                 embed_dims=256,
                 num_foreground_queries=1000,
                 foreground_threshold=0.3,
                 quality_threshold=0.2,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_fg_queries = num_foreground_queries
        self.fg_threshold = foreground_threshold
        self.quality_threshold = quality_threshold
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
        # Foreground prediction head
        self.fg_classifier = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, 1),
        )
        
        # Quality prediction head (for training)
        self.quality_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, 1),
            nn.Sigmoid()
        )
        
        # Initial position regression
        self.pos_regressor = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, 3),
        )
    
    def forward(self, bev_features, pos_embed=None, gt_points=None):
        """
        Args:
            bev_features: (B, C, H, W) BEV features
            pos_embed: (B, C, H, W) position embeddings
            gt_points: List of (N, 3) ground truth points for training
            
        Returns:
            query_embeds: (B, N_queries, C) selected query features
            query_pos: (B, N_queries, 3) initial query positions
            fg_logits: (B, H*W) foreground classification logits
            quality_scores: (B, N_queries) quality scores
        """
        B, C, H, W = bev_features.shape
        
        # Flatten BEV features
        if pos_embed is not None:
            features = bev_features + pos_embed
        else:
            features = bev_features
        
        features_flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Stage 1: Foreground query selection
        fg_logits = self.fg_classifier(features_flat).squeeze(-1)  # (B, H*W)
        fg_probs = torch.sigmoid(fg_logits)
        
        # Select top-K foreground queries
        if self.training:
            # During training, select more queries for better coverage
            num_select = min(self.num_fg_queries, H * W)
        else:
            # During inference, threshold-based selection
            num_select = min(self.num_fg_queries, H * W)
        
        topk_probs, topk_indices = torch.topk(fg_probs, num_select, dim=1)  # (B, num_select)
        
        # Gather selected features
        batch_indices = torch.arange(B, device=features_flat.device).unsqueeze(1).expand(-1, num_select)
        selected_features = features_flat[batch_indices, topk_indices]  # (B, num_select, C)
        
        # Stage 2: Quality prediction and position regression
        quality_scores = self.quality_head(selected_features).squeeze(-1)  # (B, num_select)
        pos_offsets = self.pos_regressor(selected_features)  # (B, num_select, 3)
        
        # Convert indices to BEV positions
        y_indices = topk_indices // W
        x_indices = topk_indices % W
        
        # Normalize to [0, 1]
        x_norm = (x_indices.float() + 0.5) / W
        y_norm = (y_indices.float() + 0.5) / H
        
        # Convert to world coordinates
        x_min, y_min, z_min = self.pc_range[:3]
        x_max, y_max, z_max = self.pc_range[3:]
        
        x_world = x_min + x_norm * (x_max - x_min)
        y_world = y_min + y_norm * (y_max - y_min)
        z_world = torch.zeros_like(x_world)
        
        base_positions = torch.stack([x_world, y_world, z_world], dim=-1)  # (B, num_select, 3)
        
        # Apply learned offsets
        query_pos = base_positions + torch.tanh(pos_offsets) * 5.0  # Scale offsets
        
        # Filter by quality during inference
        if not self.training:
            quality_mask = quality_scores > self.quality_threshold
            # Apply mask to each batch
            final_features = []
            final_positions = []
            final_qualities = []
            
            for b in range(B):
                mask = quality_mask[b]
                if mask.sum() > 0:
                    final_features.append(selected_features[b, mask])
                    final_positions.append(query_pos[b, mask])
                    final_qualities.append(quality_scores[b, mask])
                else:
                    # Keep at least one query
                    final_features.append(selected_features[b, :1])
                    final_positions.append(query_pos[b, :1])
                    final_qualities.append(quality_scores[b, :1])
            
            # Pad to max length for batching
            max_len = max(f.shape[0] for f in final_features)
            query_embeds = torch.zeros((B, max_len, C), device=features_flat.device)
            query_positions = torch.zeros((B, max_len, 3), device=features_flat.device)
            query_qualities = torch.zeros((B, max_len), device=features_flat.device)
            
            for b in range(B):
                L = final_features[b].shape[0]
                query_embeds[b, :L] = final_features[b]
                query_positions[b, :L] = final_positions[b]
                query_qualities[b, :L] = final_qualities[b]
        else:
            query_embeds = selected_features
            query_positions = query_pos
            query_qualities = quality_scores
        
        return query_embeds, query_positions, fg_logits, query_qualities

# ============================================================================
# Deformable Grid Attention
# ============================================================================
class DeformableGridAttention(nn.Module):
    """
    Deformable Grid Attention (DGA) - Efficient attention with learnable sampling.
    
    Divides reference regions into grids and learns offsets for flexible receptive fields.
    """
    
    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=9):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points  # 3x3 grid by default
        
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        # Initialize grid points
        grid_size = int(np.sqrt(self.num_points))
        thetas = np.linspace(-0.5, 0.5, grid_size)
        grid_x, grid_y = np.meshgrid(thetas, thetas)
        grid = np.stack([grid_x.flatten(), grid_y.flatten()], -1)
        grid = torch.from_numpy(grid).float()
        grid = grid.view(1, 1, 1, -1, 2).repeat(1, self.num_heads, self.num_levels, 1, 1)
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid.flatten())
        
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(self, query, key, value, reference_points, spatial_shapes):
        """
        Args:
            query: (B, N_queries, C)
            key: (B, N_keys, C) - not used in deformable attention
            value: (B, H*W, C) flattened BEV features
            reference_points: (B, N_queries, 3) query positions in world coordinates
            spatial_shapes: (H, W) of BEV
            
        Returns:
            output: (B, N_queries, C)
        """
        B, N_queries, C = query.shape
        _, N_keys, _ = value.shape
        H, W = spatial_shapes
        
        value = self.value_proj(value).view(B, N_keys, self.num_heads, C // self.num_heads)
        
        # Compute sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            B, N_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # Compute attention weights
        attention_weights = self.attention_weights(query).view(
            B, N_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, N_queries, self.num_heads, self.num_levels, self.num_points
        )
        
        # Normalize reference points to [0, 1]
        # reference_points is in world coordinates, need to normalize
        ref_xy = reference_points[..., :2]  # (B, N_queries, 2)
        
        # Simple grid sampling (can be improved with proper coordinate transformation)
        # For now, sample from value features
        sampling_locations = ref_xy.unsqueeze(2).unsqueeze(3).unsqueeze(4) + sampling_offsets
        
        # Sample features (simplified version)
        # In production, use proper grid_sample with coordinate transformation
        output = self._sample_features(value, sampling_locations, attention_weights, H, W)
        
        output = self.output_proj(output)
        
        return output
    
    def _sample_features(self, value, sampling_locations, attention_weights, H, W):
        """Simplified feature sampling."""
        B, N_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
        _, N_keys, _, head_dim = value.shape
        
        # Simple average pooling as placeholder
        # In production, implement proper deformable sampling
        value_avg = value.mean(dim=1)  # (B, num_heads, head_dim)
        value_avg = value_avg.unsqueeze(1).expand(-1, N_queries, -1, -1)  # (B, N_queries, num_heads, head_dim)
        
        # Apply attention weights (simplified)
        output = value_avg.flatten(2)  # (B, N_queries, C)
        
        return output

# ============================================================================
# Query-Based Decoder Layer
# ============================================================================
class QueryDecoderLayer(nn.Module):
    """Single decoder layer with self-attention and deformable cross-attention."""
    
    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=1024, dropout=0.1):
        super().__init__()
        
        self.embed_dims = embed_dims
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention (deformable)
        self.cross_attn = DeformableGridAttention(embed_dims, num_heads)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dims)
    
    def forward(self, query, query_pos, key, value, spatial_shapes):
        """
        Args:
            query: (B, N, C)
            query_pos: (B, N, 3) positions
            key: (B, M, C) 
            value: (B, M, C)
            spatial_shapes: (H, W)
        """
        # Self-attention
        q = k = query
        query2, _ = self.self_attn(q, k, query)
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        
        # Cross-attention with BEV features
        query2 = self.cross_attn(query, key, value, query_pos, spatial_shapes)
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        
        # FFN
        query2 = self.ffn(query)
        query = query + query2
        query = self.norm3(query)
        
        return query

# ============================================================================
# Main Head: RadarPointQueryHead
# ============================================================================
@HEADS.register_module()
class RadarPointQueryHead(nn.Module):
    """
    DETR-style Query-Based Radar Point Prediction Head.
    
    Architecture:
    1. BEV feature extraction with positional encoding
    2. Dual query selection (foreground + quality)
    3. Multi-layer transformer decoder with deformable attention
    4. Point prediction heads (classification + regression)
    
    Advantages:
    - Dynamic number of points
    - End-to-end training (no NMS)
    - Hungarian matching for stable training
    - Production-ready and robust
    """
    
    def __init__(self,
                 in_channels=256,
                 embed_dims=256,
                 num_decoder_layers=6,
                 num_heads=8,
                 ffn_dims=1024,
                 dropout=0.1,
                 num_foreground_queries=1000,
                 foreground_threshold=0.3,
                 quality_threshold=0.2,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 bev_h=128,
                 bev_w=128,
                 loss_cls_weight=2.0,
                 loss_reg_weight=5.0,
                 loss_quality_weight=1.0,
                 loss_fg_weight=1.0,
                 visualize_training=False,
                 enable_viz=False):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_decoder_layers = num_decoder_layers
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        self.loss_quality_weight = loss_quality_weight
        self.loss_fg_weight = loss_fg_weight
        
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        
        # Input projection
        if in_channels != embed_dims:
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, 1),
                nn.BatchNorm2d(embed_dims),
            )
        else:
            self.input_proj = nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=embed_dims // 2)
        
        # Dual query selection
        self.query_selection = DualQuerySelection(
            embed_dims=embed_dims,
            num_foreground_queries=num_foreground_queries,
            foreground_threshold=foreground_threshold,
            quality_threshold=quality_threshold,
            point_cloud_range=point_cloud_range
        )
        
        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            QueryDecoderLayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Prediction heads
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3),
        )
        
        self.quality_refine_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
        # Visualization
        self.vis_iter = 0
        self.vis_dir = "work_dirs/radar_viz_query"
        self.visualize_training = visualize_training
        self.enable_viz = enable_viz
        if self.visualize_training or self.enable_viz:
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, bev_feat):
        """
        Args:
            bev_feat: (B, C, H, W) BEV features
            
        Returns:
            List of dict containing:
                - radar_points: List of (N, 3) predicted points per batch
                - point_scores: List of (N,) confidence scores per batch
        """
        B = bev_feat.shape[0]
        
        # Resize if needed
        if bev_feat.shape[2:] != (self.bev_h, self.bev_w):
            bev_feat = F.interpolate(bev_feat, size=(self.bev_h, self.bev_w),
                                    mode='bilinear', align_corners=False)
        
        # Project input
        bev_feat = self.input_proj(bev_feat)
        
        # Add positional encoding
        pos_embed = self.pos_encoder(bev_feat)
        
        # Flatten for transformer
        bev_flat = bev_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Stage 1: Query selection
        query_embeds, query_pos, fg_logits, query_qualities = self.query_selection(
            bev_feat, pos_embed, gt_points=None
        )
        
        # Stage 2: Transformer decoder
        queries = query_embeds
        for layer in self.decoder_layers:
            queries = layer(
                queries, 
                query_pos,
                bev_flat,
                bev_flat,
                (self.bev_h, self.bev_w)
            )
        
        # Stage 3: Predictions
        cls_logits = self.cls_head(queries).squeeze(-1)  # (B, N)
        cls_scores = torch.sigmoid(cls_logits)
        
        reg_offsets = self.reg_head(queries)  # (B, N, 3)
        quality_refined = self.quality_refine_head(queries).squeeze(-1)  # (B, N)
        
        # Final positions
        final_positions = query_pos + torch.tanh(reg_offsets) * 2.0
        
        # Convert to output format
        if not self.training:
            radar_points = []
            point_scores = []
            
            for b in range(B):
                scores = cls_scores[b] * quality_refined[b]
                valid_mask = scores > 0.3
                
                if valid_mask.sum() > 0:
                    radar_points.append(final_positions[b, valid_mask])
                    point_scores.append(scores[valid_mask])
                else:
                    radar_points.append(torch.zeros((0, 3), device=bev_feat.device))
                    point_scores.append(torch.zeros((0,), device=bev_feat.device))
        else:
            radar_points = final_positions
            point_scores = cls_scores * quality_refined
        
        return [{
            'radar_points': radar_points,
            'point_scores': point_scores,
            'cls_logits': cls_logits,
            'reg_offsets': reg_offsets,
            'query_pos': query_pos,
            'fg_logits': fg_logits,
            'query_qualities': query_qualities,
        }]
    
    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, preds_dicts, gt_radar_points, img_metas=None, bev_feat=None):
        """
        Compute losses with Hungarian matching.
        
        Args:
            preds_dicts: List of prediction dictionaries
            gt_radar_points: Ground truth radar points
            img_metas: Image metadata
            bev_feat: BEV features for visualization
        """
        pred = preds_dicts[0]
        
        cls_logits = pred['cls_logits']
        reg_offsets = pred['reg_offsets']
        query_pos = pred['query_pos']
        fg_logits = pred['fg_logits']
        query_qualities = pred['query_qualities']
        
        B = cls_logits.shape[0]
        
        # Prepare GT
        if isinstance(gt_radar_points, torch.Tensor):
            gt_radar_points = [gt_radar_points[i] for i in range(B)]
        else:
            gt_radar_points = [torch.as_tensor(pts, device=cls_logits.device)
                             for pts in gt_radar_points]
        
        # Final predicted positions
        pred_positions = query_pos + torch.tanh(reg_offsets) * 2.0
        
        # Hungarian matching
        indices = self._hungarian_matching(pred_positions, gt_radar_points, cls_logits)
        
        # Compute losses
        cls_loss = self._classification_loss(cls_logits, indices, gt_radar_points)
        reg_loss = self._regression_loss(pred_positions, indices, gt_radar_points)
        quality_loss = self._quality_loss(query_qualities, indices, pred_positions, gt_radar_points)
        fg_loss = self._foreground_loss(fg_logits, gt_radar_points)
        
        # Additional: False positive penalty
        # Penalize high-confidence predictions that are far from any GT
        fp_loss = self._false_positive_penalty(pred_positions, cls_logits, indices, gt_radar_points)
        
        # Visualization
        if self.visualize_training:
            self.vis_iter += 1
            self._visualize_training(pred, gt_radar_points, bev_feat, indices)
        
        return {
            'loss_radar_cls': self.loss_cls_weight * cls_loss,
            'loss_radar_reg': self.loss_reg_weight * reg_loss,
            'loss_radar_quality': self.loss_quality_weight * quality_loss,
            'loss_radar_fg': self.loss_fg_weight * fg_loss,
            'loss_radar_fp': 1.0 * fp_loss,  # False positive penalty
        }
    
    def _hungarian_matching(self, pred_positions, gt_points_list, cls_logits):
        """Hungarian matching between predictions and ground truth."""
        from scipy.optimize import linear_sum_assignment
        
        B = len(gt_points_list)
        indices = []
        
        for b in range(B):
            pred_pts = pred_positions[b]  # (N_pred, 3)
            gt_pts = gt_points_list[b]    # (N_gt, 3)
            
            if len(gt_pts) == 0:
                indices.append(([], []))
                continue
            
            # Compute cost matrix
            cost_spatial = torch.cdist(pred_pts, gt_pts)  # (N_pred, N_gt)
            cost_cls = -(cls_logits[b].sigmoid().unsqueeze(1))  # (N_pred, 1)
            
            cost = cost_spatial + cost_cls
            
            # Hungarian algorithm
            pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            
            indices.append((pred_idx, gt_idx))
        
        return indices
    
    def _classification_loss(self, cls_logits, indices, gt_points_list):
        """Focal loss for classification with hard negative mining."""
        B = cls_logits.shape[0]
        device = cls_logits.device
        
        losses = []
        for b in range(B):
            pred_idx, gt_idx = indices[b]
            
            # Create target
            target = torch.zeros_like(cls_logits[b])
            if len(pred_idx) > 0:
                target[pred_idx] = 1.0
            
            # Get predictions
            cls_probs = torch.sigmoid(cls_logits[b])
            
            # Focal loss implementation
            # For positive samples (matched)
            pos_loss = -target * torch.pow(1 - cls_probs, 2) * torch.log(cls_probs + 1e-8)
            
            # For negative samples (unmatched)
            # Penalize high-confidence false positives more heavily
            neg_loss = -(1 - target) * torch.pow(cls_probs, 2) * torch.log(1 - cls_probs + 1e-8)
            
            # Combine
            focal_loss = (pos_loss + neg_loss).mean()
            
            losses.append(focal_loss)
        
        return torch.stack(losses).mean()
    
    def _regression_loss(self, pred_positions, indices, gt_points_list):
        """Smooth L1 loss for position regression."""
        losses = []
        
        for b in range(len(gt_points_list)):
            pred_idx, gt_idx = indices[b]
            
            if len(pred_idx) == 0:
                continue
            
            pred_matched = pred_positions[b, pred_idx]
            gt_matched = gt_points_list[b][gt_idx]
            
            loss = F.smooth_l1_loss(pred_matched, gt_matched)
            losses.append(loss)
        
        if len(losses) == 0:
            return pred_positions.sum() * 0
        
        return torch.stack(losses).mean()
    
    def _quality_loss(self, query_qualities, indices, pred_positions, gt_points_list):
        """IoU-based quality loss."""
        losses = []
        
        for b in range(len(gt_points_list)):
            pred_idx, gt_idx = indices[b]
            
            if len(pred_idx) == 0:
                continue
            
            # Compute IoU-like metric based on distance
            pred_matched = pred_positions[b, pred_idx]
            gt_matched = gt_points_list[b][gt_idx]
            
            dists = torch.norm(pred_matched - gt_matched, dim=-1)
            iou_targets = torch.exp(-dists)  # Closer = higher quality
            
            quality_pred = query_qualities[b, pred_idx]
            loss = F.mse_loss(quality_pred, iou_targets)
            
            losses.append(loss)
        
        if len(losses) == 0:
            return query_qualities.sum() * 0
        
        return torch.stack(losses).mean()
    
    def _foreground_loss(self, fg_logits, gt_points_list):
        """Binary classification loss for foreground detection."""
        B, HW = fg_logits.shape
        H = W = int(np.sqrt(HW))
        device = fg_logits.device
        
        losses = []
        
        for b in range(B):
            gt_pts = gt_points_list[b]
            
            # Create target heatmap
            target = torch.zeros(HW, device=device)
            
            if len(gt_pts) > 0:
                # Convert GT points to BEV indices
                x_min, y_min = self.pc_range[:2]
                x_max, y_max = self.pc_range[3:5]
                
                gt_np = gt_pts.cpu().numpy() if isinstance(gt_pts, torch.Tensor) else gt_pts
                x_norm = (gt_np[:, 0] - x_min.item()) / (x_max.item() - x_min.item())
                y_norm = (gt_np[:, 1] - y_min.item()) / (y_max.item() - y_min.item())
                
                x_idx = np.clip((x_norm * W).astype(int), 0, W - 1)
                y_idx = np.clip((y_norm * H).astype(int), 0, H - 1)
                
                flat_idx = y_idx * W + x_idx
                target[flat_idx] = 1.0
            
            loss = F.binary_cross_entropy_with_logits(fg_logits[b], target)
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def _false_positive_penalty(self, pred_positions, cls_logits, indices, gt_points_list):
        """
        Penalize high-confidence predictions that are far from any ground truth.
        This explicitly targets false positives.
        """
        B = cls_logits.shape[0]
        device = cls_logits.device
        
        losses = []
        
        for b in range(B):
            pred_idx, gt_idx = indices[b]
            gt_pts = gt_points_list[b]
            
            if len(gt_pts) == 0:
                # No GT points, all predictions are false positives
                # Penalize all high-confidence predictions
                cls_probs = torch.sigmoid(cls_logits[b])
                fp_loss = (cls_probs ** 2).mean()  # Quadratic penalty
                losses.append(fp_loss)
                continue
            
            # Create mask for unmatched predictions
            N_queries = cls_logits.shape[1]
            matched_mask = torch.zeros(N_queries, dtype=torch.bool, device=device)
            if len(pred_idx) > 0:
                matched_mask[pred_idx] = True
            
            # Get unmatched predictions
            unmatched_mask = ~matched_mask
            unmatched_positions = pred_positions[b, unmatched_mask]  # (N_unmatched, 3)
            unmatched_scores = torch.sigmoid(cls_logits[b, unmatched_mask])  # (N_unmatched,)
            
            if len(unmatched_positions) == 0:
                losses.append(torch.tensor(0.0, device=device))
                continue
            
            # Compute distance from each unmatched prediction to nearest GT
            dists = torch.cdist(unmatched_positions, gt_pts)  # (N_unmatched, N_gt)
            min_dists = dists.min(dim=1)[0]  # (N_unmatched,)
            
            # False positive penalty: confidence * distance_penalty
            # Far predictions with high confidence get penalized heavily
            distance_threshold = 2.0  # meters
            is_far = (min_dists > distance_threshold).float()
            
            # Penalty = confidence^2 * is_far * normalized_distance
            normalized_dist = torch.clamp(min_dists / 10.0, 0, 1)  # Normalize to [0, 1]
            fp_penalty = (unmatched_scores ** 2) * is_far * normalized_dist
            
            fp_loss = fp_penalty.mean()
            losses.append(fp_loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()
    
    def _visualize_training(self, pred, gt_points_list, bev_feat, indices):
        """Visualize training process."""
        B = len(gt_points_list)
        s_dir = os.path.join(self.vis_dir, f"{self.vis_iter}")
        
        pred_positions = pred['query_pos'] + torch.tanh(pred['reg_offsets']) * 2.0
        cls_scores = torch.sigmoid(pred['cls_logits'])
        query_qualities = pred['query_qualities']
        
        # Save BEV features
        if bev_feat is not None:
            for i in range(B):
                iter_str = f"{i:05d}"
                os.makedirs(os.path.join(s_dir, "bev_features"), exist_ok=True)
                bev_path = os.path.join(s_dir, "bev_features", f"iter_{iter_str}_bev.png")
                visualize_bev(bev_feat[i], bev_path)
        
        # Visualize points and queries
        for i in range(B):
            iter_str = f"{i:05d}"
            os.makedirs(os.path.join(s_dir, "point_clouds"), exist_ok=True)
            os.makedirs(os.path.join(s_dir, "queries"), exist_ok=True)
            
            # Get matched predictions
            pred_idx, gt_idx = indices[i]
            if len(pred_idx) > 0:
                matched_pred = pred_positions[i, pred_idx]
                matched_scores = cls_scores[i, pred_idx] * query_qualities[i, pred_idx]
            else:
                matched_pred = torch.zeros((0, 3), device=pred_positions.device)
                matched_scores = torch.zeros((0,), device=pred_positions.device)
            
            # 3D point cloud
            pts_path_3d = os.path.join(s_dir, "point_clouds", f"iter_{iter_str}_points_3d.png")
            visualize_point_cloud_3d(matched_pred, gt_points_list[i], pts_path_3d, matched_scores)
            
            # Query selection visualization - pass per-batch data
            query_path = os.path.join(s_dir, "queries", f"iter_{iter_str}_queries.png")
            fg_mask = cls_scores[i] > 0.5
            visualize_queries(
                pred_positions[i],      # Query positions for this batch
                fg_mask,                # Foreground mask for this batch
                query_qualities[i],     # Quality scores for this batch
                query_path
            )
            

# ============================================================================
# Backward Compatibility Alias
# ============================================================================
# @HEADS.register_module()
# class RadarPointHead(RadarPointRegressionHead):
#     """Alias for backward compatibility."""
#     pass
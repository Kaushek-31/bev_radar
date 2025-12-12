#!/usr/bin/env python
"""
Evaluate Radar Point Cloud Prediction with Confidence Visualization

Usage:
    python eval_radar.py \
        configs/bevdet/bevdet-r50-4d-cbgs-radar-frozen.py \
        work_dirs/radar_training/latest.pth \
        --eval radar_chamfer radar_coverage \
        --show-dir results_vis \
        --num-vis 50
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

from mmdet.utils import setup_multi_processes, compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate radar point cloud prediction')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--eval', type=str, nargs='+', 
                       help='evaluation metrics: "radar_chamfer", "radar_coverage"')
    parser.add_argument('--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--num-vis', type=int, default=50, help='number of samples to visualize')
    parser.add_argument('--num-samples', type=int, default=None, 
                       help='number of samples to evaluate (default: all)')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    return args


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image from ImageNet normalization."""
    img = img.copy()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).astype(np.uint8)


def visualize_radar_with_confidence(pred_points, pred_confidence, gt_points, 
                                    camera_images=None, title="", save_path=None):
    """Visualize radar predictions colored by confidence score."""
    print(f"Visualizing radar with confidence: {title}")
    XLIM, YLIM, ZLIM = 50, 50, 5
    
    # Handle empty predictions
    if pred_points is None or len(pred_points) == 0:
        pred_points = np.zeros((0, 3))
        pred_confidence = np.zeros(0)
    if gt_points is None or len(gt_points) == 0:
        gt_points = np.zeros((0, 3))
    
    # Split by confidence threshold
    if len(pred_confidence) > 0:
        high_conf_mask = pred_confidence > 0.5
        high_conf_points = pred_points[high_conf_mask]
        high_conf_scores = pred_confidence[high_conf_mask]
        low_conf_points = pred_points[~high_conf_mask]
        low_conf_scores = pred_confidence[~high_conf_mask]
    else:
        high_conf_points = high_conf_scores = np.zeros(0)
        low_conf_points, low_conf_scores = pred_points, np.zeros(len(pred_points))
    
    # Compute metrics
    chamfer_all = chamfer_high = coverage_1m_all = coverage_1m_high = 0.0
    
    if len(pred_points) > 0 and len(gt_points) > 0:
        pred_t = torch.from_numpy(pred_points).float()
        gt_t = torch.from_numpy(gt_points).float()
        dist = torch.cdist(pred_t, gt_t)
        chamfer_all = (dist.min(1)[0].mean() + dist.min(0)[0].mean()).item()
        coverage_1m_all = (dist.min(0)[0] < 1.0).float().mean().item()
        
        if len(high_conf_points) > 0:
            pred_high_t = torch.from_numpy(high_conf_points).float()
            dist_high = torch.cdist(pred_high_t, gt_t)
            chamfer_high = (dist_high.min(1)[0].mean() + dist_high.min(0)[0].mean()).item()
            coverage_1m_high = (dist_high.min(0)[0] < 1.0).float().mean().item()
    
    # Confidence stats
    if len(pred_confidence) > 0:
        conf_mean, conf_std = pred_confidence.mean(), pred_confidence.std()
        conf_min, conf_max = pred_confidence.min(), pred_confidence.max()
        conf_median = np.median(pred_confidence)
    else:
        conf_mean = conf_std = conf_min = conf_max = conf_median = 0.0
    
    # Metrics text
    metric_text = (
        f"CONFIDENCE STATS\n"
        f"{'='*20}\n"
        f"Mean:   {conf_mean:.3f}\n"
        f"Median: {conf_median:.3f}\n"
        f"Std:    {conf_std:.3f}\n"
        f"Range:  [{conf_min:.3f}, {conf_max:.3f}]\n"
        f"High (>0.5): {high_conf_mask.sum()}/{len(pred_confidence)}\n"
        f"\n"
        f"ALL PREDICTIONS\n"
        f"{'='*20}\n"
        f"Count:   {len(pred_points)}\n"
        f"Chamfer: {chamfer_all:.3f}m\n"
        f"Cov@1m:  {coverage_1m_all:.1%}\n"
        f"\n"
        f"HIGH CONF (>0.5)\n"
        f"{'='*20}\n"
        f"Count:   {len(high_conf_points)}\n"
        f"Chamfer: {chamfer_high:.3f}m\n"
        f"Cov@1m:  {coverage_1m_high:.1%}\n"
        f"\n"
        f"GROUND TRUTH\n"
        f"{'='*20}\n"
        f"Count:   {len(gt_points)}\n"
    )
    
    # Setup figure - CHANGED: Now 4 rows instead of 3
    if camera_images is not None and camera_images.shape[0] > 0:
        fig = plt.figure(figsize=(28, 32))  # Taller for extra row
        gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Camera views
        cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for cam_idx in range(min(6, camera_images.shape[0])):
            ax = fig.add_subplot(gs[cam_idx // 3, cam_idx % 3])
            ax.imshow(denormalize_image(camera_images[cam_idx]))
            ax.set_title(cam_names[cam_idx], fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Radar plots start at row 2
        row_offset = 2
    else:
        fig = plt.figure(figsize=(28, 24))  # Taller for extra row
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        row_offset = 0
    
    # 1. 3D All predictions (colored by confidence)
    ax1 = fig.add_subplot(gs[row_offset, 0], projection='3d')
    if len(gt_points) > 0:
        ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                   c='blue', s=15, alpha=0.4, label='GT', marker='o')
    if len(pred_points) > 0:
        scatter = ax1.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c=pred_confidence, cmap='viridis', s=25, alpha=0.8, 
                   vmin=0, vmax=1, marker='^', edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax1, label='Confidence', shrink=0.6)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.set_title('All Predictions (colored by confidence)', fontsize=12, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-XLIM, XLIM]); ax1.set_ylim([-YLIM, YLIM]); ax1.set_zlim([-ZLIM, ZLIM])
    
    # 2. BEV All predictions
    ax2 = fig.add_subplot(gs[row_offset, 1])
    if len(gt_points) > 0:
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], c='blue', s=20, alpha=0.4, label='GT')
    if len(pred_points) > 0:
        scatter = ax2.scatter(pred_points[:, 0], pred_points[:, 1], 
                   c=pred_confidence, cmap='viridis', s=40, alpha=0.8, vmin=0, vmax=1,
                   marker='^', edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax2, label='Confidence')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_title(f'BEV - All ({len(pred_points)} pts)', fontsize=12, fontweight='bold')
    ax2.axis('equal'); ax2.grid(True, alpha=0.3); ax2.legend()
    ax2.set_xlim([-XLIM, XLIM]); ax2.set_ylim([-YLIM, YLIM])
    
    # 3. BEV High confidence only
    ax3 = fig.add_subplot(gs[row_offset, 2])
    if len(gt_points) > 0:
        ax3.scatter(gt_points[:, 0], gt_points[:, 1], c='blue', s=20, alpha=0.4, label='GT')
    if len(high_conf_points) > 0:
        scatter = ax3.scatter(high_conf_points[:, 0], high_conf_points[:, 1], 
                   c=high_conf_scores, cmap='plasma', s=50, alpha=0.9,
                   vmin=0.5, vmax=1.0, marker='^', edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax3, label='Confidence')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
    ax3.set_title(f'High Conf >0.5 ({len(high_conf_points)} pts)', fontsize=12, fontweight='bold')
    ax3.axis('equal'); ax3.grid(True, alpha=0.3); ax3.legend()
    ax3.set_xlim([-XLIM, XLIM]); ax3.set_ylim([-YLIM, YLIM])
    
    # 4. 3D High confidence
    ax4 = fig.add_subplot(gs[row_offset+1, 0], projection='3d')
    if len(gt_points) > 0:
        ax4.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                   c='blue', s=15, alpha=0.4, label='GT')
    if len(high_conf_points) > 0:
        scatter = ax4.scatter(high_conf_points[:, 0], high_conf_points[:, 1], high_conf_points[:, 2], 
                   c=high_conf_scores, cmap='plasma', s=30, alpha=0.9,
                   vmin=0.5, vmax=1.0, marker='^', edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax4, label='Confidence', shrink=0.6)
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)'); ax4.set_zlabel('Z (m)')
    ax4.set_title('3D High Confidence', fontsize=12, fontweight='bold')
    ax4.legend(); ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-XLIM, XLIM]); ax4.set_ylim([-YLIM, YLIM]); ax4.set_zlim([-ZLIM, ZLIM])
    
    # 5. Confidence histogram
    ax5 = fig.add_subplot(gs[row_offset+1, 1])
    if len(pred_confidence) > 0:
        ax5.hist(pred_confidence, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax5.axvline(conf_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {conf_mean:.3f}')
        ax5.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold')
        ax5.set_xlabel('Confidence'); ax5.set_ylabel('Count')
        ax5.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax5.legend(); ax5.grid(True, alpha=0.3); ax5.set_xlim([0, 1])
    
    # 6. Metrics text
    ax6 = fig.add_subplot(gs[row_offset+1, 2])
    ax6.axis('off')
    ax6.text(0.05, 0.5, metric_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax6.set_title('Metrics', fontsize=12, fontweight='bold')
    
    # 7. Confidence vs Distance to GT
    ax7 = fig.add_subplot(gs[row_offset+2, 0])
    if len(pred_points) > 0 and len(gt_points) > 0:
        dist = torch.cdist(torch.from_numpy(pred_points).float(), 
                          torch.from_numpy(gt_points).float())
        min_dist = dist.min(1)[0].numpy()
        ax7.scatter(pred_confidence, min_dist, c=pred_confidence, cmap='viridis', 
                   s=20, alpha=0.6, vmin=0, vmax=1)
        ax7.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='1m')
        ax7.axhline(2.0, color='orange', linestyle='--', alpha=0.5, label='2m')
        ax7.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Conf=0.5')
        ax7.set_xlabel('Confidence'); ax7.set_ylabel('Dist to GT (m)')
        ax7.set_title('Confidence vs Accuracy', fontsize=12, fontweight='bold')
        ax7.legend(); ax7.grid(True, alpha=0.3); ax7.set_xlim([0, 1])
    
    # 8. Low confidence points
    ax8 = fig.add_subplot(gs[row_offset+2, 1])
    if len(low_conf_points) > 0:
        if len(gt_points) > 0:
            ax8.scatter(gt_points[:, 0], gt_points[:, 1], c='blue', s=20, alpha=0.4, label='GT')
        ax8.scatter(low_conf_points[:, 0], low_conf_points[:, 1], 
                   c=low_conf_scores, cmap='Reds_r', s=40, alpha=0.7,
                   vmin=0, vmax=0.5, marker='v', edgecolors='black', linewidths=0.5)
        ax8.set_xlabel('X (m)'); ax8.set_ylabel('Y (m)')
        ax8.set_title(f'Low Conf <0.5 ({len(low_conf_points)} pts)', fontsize=12, fontweight='bold')
        ax8.axis('equal'); ax8.grid(True, alpha=0.3); ax8.legend()
        ax8.set_xlim([-XLIM, XLIM]); ax8.set_ylim([-YLIM, YLIM])
    
    # 9. Cumulative confidence
    ax9 = fig.add_subplot(gs[row_offset+2, 2])
    if len(pred_confidence) > 0:
        sorted_conf = np.sort(pred_confidence)
        cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
        ax9.plot(sorted_conf, cumulative, linewidth=2, color='steelblue')
        ax9.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        pct_high = (pred_confidence > 0.5).mean()
        ax9.text(0.6, 0.3, f'{pct_high:.1%} >0.5', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax9.set_xlabel('Confidence'); ax9.set_ylabel('Cumulative Fraction')
        ax9.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
        ax9.legend(); ax9.grid(True, alpha=0.3)
        ax9.set_xlim([0, 1]); ax9.set_ylim([0, 1])
    
    # ========== NEW: CLOSE-UP VIEWS ==========
    
    # 10. BEV Close-up: ±5m
    ax10 = fig.add_subplot(gs[row_offset+3, 0])
    if len(gt_points) > 0:
        # Filter GT points within ±5m
        mask_5m = (np.abs(gt_points[:, 0]) <= 5) & (np.abs(gt_points[:, 1]) <= 5)
        gt_5m = gt_points[mask_5m]
        if len(gt_5m) > 0:
            ax10.scatter(gt_5m[:, 0], gt_5m[:, 1], c='blue', s=50, alpha=0.5, 
                        label=f'GT ({len(gt_5m)})', marker='o')
    
    if len(pred_points) > 0:
        # Filter predictions within ±5m
        mask_5m_pred = (np.abs(pred_points[:, 0]) <= 5) & (np.abs(pred_points[:, 1]) <= 5) & (np.abs(pred_confidence[:]) >= 0.7)
        pred_5m = pred_points[mask_5m_pred]
        conf_5m = pred_confidence[mask_5m_pred]
        if len(pred_5m) > 0:
            scatter = ax10.scatter(pred_5m[:, 0], pred_5m[:, 1], 
                       c=conf_5m, cmap='viridis', s=80, alpha=0.8, vmin=0, vmax=1,
                       marker='^', edgecolors='black', linewidths=0.8)
            plt.colorbar(scatter, ax=ax10, label='Confidence')
    
    ax10.set_xlabel('X (m)', fontsize=11); ax10.set_ylabel('Y (m)', fontsize=11)
    ax10.set_title(f'Close-up BEV (±5m)', fontsize=12, fontweight='bold')
    ax10.axis('equal'); ax10.grid(True, alpha=0.3); ax10.legend(fontsize=9)
    ax10.set_xlim([-5, 5]); ax10.set_ylim([-5, 5])
    
    # 11. BEV Close-up: ±10m
    ax11 = fig.add_subplot(gs[row_offset+3, 1])
    if len(gt_points) > 0:
        # Filter GT points within ±10m
        mask_10m = (np.abs(gt_points[:, 0]) <= 10) & (np.abs(gt_points[:, 1]) <= 10)
        gt_10m = gt_points[mask_10m]
        if len(gt_10m) > 0:
            ax11.scatter(gt_10m[:, 0], gt_10m[:, 1], c='blue', s=40, alpha=0.5, 
                        label=f'GT ({len(gt_10m)})', marker='o')
    
    if len(pred_points) > 0:
        # Filter predictions within ±10m
        mask_10m_pred = (np.abs(pred_points[:, 0]) <= 10) & (np.abs(pred_points[:, 1]) <= 10) & (np.abs(pred_confidence[:]) >= 0.7)
        pred_10m = pred_points[mask_10m_pred]
        conf_10m = pred_confidence[mask_10m_pred]
        if len(pred_10m) > 0:
            scatter = ax11.scatter(pred_10m[:, 0], pred_10m[:, 1], 
                       c=conf_10m, cmap='viridis', s=60, alpha=0.8, vmin=0, vmax=1,
                       marker='^', edgecolors='black', linewidths=0.7)
            plt.colorbar(scatter, ax=ax11, label='Confidence')
    
    ax11.set_xlabel('X (m)', fontsize=11); ax11.set_ylabel('Y (m)', fontsize=11)
    ax11.set_title(f'Close-up BEV (±10m)', fontsize=12, fontweight='bold')
    ax11.axis('equal'); ax11.grid(True, alpha=0.3); ax11.legend(fontsize=9)
    ax11.set_xlim([-10, 10]); ax11.set_ylim([-10, 10])
    
    # 12. Error heatmap or distance distribution
    ax12 = fig.add_subplot(gs[row_offset+3, 2])
    if len(pred_points) > 0 and len(gt_points) > 0:
        # Show distribution of distances to nearest GT
        dist_matrix = torch.cdist(torch.from_numpy(pred_points).float(), 
                                  torch.from_numpy(gt_points).float())
        min_dists = dist_matrix.min(1)[0].numpy()
        
        # Create 2D histogram of prediction locations colored by distance error
        # Bin the BEV space
        x_bins = np.linspace(-50, 50, 26)
        y_bins = np.linspace(-50, 50, 26)
        
        # For each bin, compute mean error
        H, xedges, yedges = np.histogram2d(pred_points[:, 0], pred_points[:, 1], 
                                           bins=[x_bins, y_bins])
        
        # Weight by distance error
        H_error, _, _ = np.histogram2d(pred_points[:, 0], pred_points[:, 1], 
                                       bins=[x_bins, y_bins], weights=min_dists)
        
        # Average error per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_error = H_error / H
            avg_error[~np.isfinite(avg_error)] = 0
        
        im = ax12.imshow(avg_error.T, origin='lower', extent=[-50, 50, -50, 50],
                        cmap='RdYlGn_r', aspect='equal', vmin=0, vmax=5)
        plt.colorbar(im, ax=ax12, label='Avg Error (m)')
        
        # Overlay GT points
        if len(gt_points) > 0:
            ax12.scatter(gt_points[:, 0], gt_points[:, 1], c='blue', s=10, 
                        alpha=0.3, marker='x', label='GT')
        
        ax12.set_xlabel('X (m)', fontsize=11); ax12.set_ylabel('Y (m)', fontsize=11)
        ax12.set_title('Spatial Error Distribution', fontsize=12, fontweight='bold')
        ax12.legend(fontsize=9)
    else:
        ax12.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center',
                 transform=ax12.transAxes, fontsize=12)
        ax12.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def extract_radar_gt(info):
    """Extract ground truth radar points."""
    if 'radar_points' in info and info['radar_points'] is not None:
        radar_data = info['radar_points']
        if isinstance(radar_data, dict) and 'xyz' in radar_data:
            return radar_data['xyz'][:, :3]
        elif isinstance(radar_data, np.ndarray) and radar_data.shape[-1] >= 3:
            return radar_data[:, :3]
    return np.zeros((0, 3), dtype=np.float32)


def single_gpu_test(model, data_loader, show_dir=None, num_vis=50):
    """Run evaluation and extract predictions with confidence."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    if show_dir:
        os.makedirs(show_dir, exist_ok=True)
    
    vis_count = 0
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # Extract inputs - handle DataContainer
            img_inputs_container = data.get('img_inputs', None)
            img_metas_container = data.get('img_metas', None)
            
            if img_inputs_container is None or img_metas_container is None:
                print(f"Warning: Missing data at sample {i}")
                prog_bar.update()
                continue
            
            # Unwrap DataContainer
            if hasattr(img_inputs_container, 'data'):
                img_inputs_list = img_inputs_container.data
            else:
                img_inputs_list = img_inputs_container
            
            if hasattr(img_metas_container, 'data'):
                img_metas_list = img_metas_container.data
            else:
                img_metas_list = img_metas_container
            
            # Unwrap batch dimension
            if isinstance(img_inputs_list, list) and len(img_inputs_list) > 0:
                first_elem = img_inputs_list[0]
                if isinstance(first_elem, list):
                    img_inputs = first_elem
                else:
                    img_inputs = img_inputs_list
            else:
                img_inputs = img_inputs_list
            
            if isinstance(img_metas_list, list) and len(img_metas_list) > 0:
                first_elem = img_metas_list[0]
                if isinstance(first_elem, dict):
                    img_metas = img_metas_list
                else:
                    img_metas = first_elem
            else:
                img_metas = img_metas_list
            
            # Get camera images for visualization
            camera_images = None
            if isinstance(img_inputs, list) and len(img_inputs) > 0:
                if torch.is_tensor(img_inputs[0]):
                    img_tensor = img_inputs[0]
                    if img_tensor.dim() == 5:
                        img_tensor = img_tensor[0]
                    camera_images = img_tensor.cpu().numpy()
            
            # Move to CUDA
            if isinstance(img_inputs, list):
                img_inputs = [x.cuda() if torch.is_tensor(x) else x for x in img_inputs]
            elif torch.is_tensor(img_inputs):
                img_inputs = img_inputs.cuda()
            
            # Forward
            try:
                result = model.module.simple_test(
                    points=None, 
                    img_metas=[img_metas] if not isinstance(img_metas, list) else img_metas, 
                    img=img_inputs
                )
            except Exception as e:
                print(f"Error at sample {i}: {e}")
                import traceback
                traceback.print_exc()
                prog_bar.update()
                continue
        
        # Extract GT
        info = dataset.get_data_info(i)
        if 'radar_info' in info and 'radar_points' in info['radar_info']:
            gt_points = info['radar_info']['radar_points']
        else:
            gt_points = extract_radar_gt(dataset.data_infos[i])
        
        # Extract predictions
        pred_points = pred_conf = None
        if len(result) > 0:
            if 'radar_points' in result[0]:
                pred_points = result[0]['radar_points']
                if isinstance(pred_points, torch.Tensor):
                    pred_points = pred_points.cpu().numpy()
            if 'confidence' in result[0]:
                print("Extracting prediction confidence...")
                pred_conf = result[0]['confidence']
                if isinstance(pred_conf, torch.Tensor):
                    pred_conf = pred_conf.cpu().numpy()
        
        # Store
        results.append({
            'pred_radar_points': pred_points,
            'pred_confidence': pred_conf,
            'gt_radar_points': gt_points,
            'sample_idx': info.get('sample_idx', str(i)),
            'camera_images': camera_images,
        })
        
        # Visualize
        if show_dir and vis_count < num_vis and pred_points is not None and pred_conf is not None:
            save_path = os.path.join(show_dir, f'{results[-1]["sample_idx"]}.png')
            visualize_radar_with_confidence(
                pred_points, pred_conf, gt_points,
                camera_images=camera_images,
                title=f'Sample {i}: {results[-1]["sample_idx"]}',
                save_path=save_path
            )
            vis_count += 1
        
        prog_bar.update()
    
    return results


def compute_metrics(pred_points, gt_points):
    """Compute Chamfer distance and coverage."""
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf'), float('inf'), 0.0, 0.0
    
    pred = torch.from_numpy(pred_points).float()
    gt = torch.from_numpy(gt_points).float()
    dist = torch.cdist(pred, gt)
    
    chamfer = (dist.min(1)[0].mean() + dist.min(0)[0].mean()).item()
    pred_to_gt = dist.min(1)[0].mean().item()
    cov_1m = (dist.min(0)[0] < 1.0).float().mean().item()
    cov_2m = (dist.min(0)[0] < 2.0).float().mean().item()
    
    return chamfer, pred_to_gt, cov_1m, cov_2m


def evaluate_results(results):
    """Compute evaluation metrics."""
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    chamfer_all, chamfer_high = [], []
    cov_1m_all, cov_1m_high = [], []
    confidences = []
    
    for result in tqdm(results, desc="Computing metrics"):
        pred = result['pred_radar_points']
        conf = result.get('pred_confidence')
        gt = result['gt_radar_points']
        
        if pred is None or len(pred) == 0:
            continue
        
        # All predictions
        ch, _, c1, c2 = compute_metrics(pred, gt)
        chamfer_all.append(ch)
        cov_1m_all.append(c1)
        
        # Confidence stats
        if conf is not None:
            confidences.extend(conf.tolist())
            
            # High confidence only
            high_mask = conf > 0.5
            if high_mask.sum() > 0:
                ch_high, _, c1_high, _ = compute_metrics(pred[high_mask], gt)
                chamfer_high.append(ch_high)
                cov_1m_high.append(c1_high)
    
    # Print results
    print(f"\n{'='*70}")
    if confidences:
        print(f"Confidence: mean={np.mean(confidences):.3f}, "
              f"med={np.median(confidences):.3f}, "
              f">0.5: {np.mean([c>0.5 for c in confidences]):.1%}")
    
    if chamfer_all:
        print(f"\nAll Predictions:")
        print(f"  Chamfer: {np.mean(chamfer_all):.4f}m (±{np.std(chamfer_all):.4f})")
        print(f"  Cov@1m:  {np.mean(cov_1m_all):.1%}")
    
    if chamfer_high:
        print(f"\nHigh Confidence (>0.5):")
        print(f"  Chamfer: {np.mean(chamfer_high):.4f}m")
        print(f"  Cov@1m:  {np.mean(cov_1m_high):.1%}")
    
    print("="*70 + "\n")
    
    return {
        'chamfer_mean': float(np.mean(chamfer_all)) if chamfer_all else 0,
        'coverage_1m': float(np.mean(cov_1m_all)) if cov_1m_all else 0,
        'conf_mean': float(np.mean(confidences)) if confidences else 0,
        'conf_high_pct': float(np.mean([c>0.5 for c in confidences])) if confidences else 0,
    }


def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]
    
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=True)
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset.data_infos = dataset.data_infos[:args.num_samples]
    
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False
    )
    
    # Build model
    if '4D' in cfg.model.type:
        cfg.model.align_after_view_transfromation = True
    cfg.model.train_cfg = None
    
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    
    print("\n" + "="*70)
    print("RADAR EVALUATION")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {len(dataset)}")
    print("="*70 + "\n")
    
    # Run evaluation
    results = single_gpu_test(model, data_loader, args.show_dir, args.num_vis)
    
    # Compute metrics
    if args.eval:
        eval_results = evaluate_results(results)
        
        # Save results
        if args.out:
            mmcv.dump(results, args.out)
            eval_path = args.out.replace('.pkl', '_eval.json')
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"✓ Results saved to: {args.out}")
            print(f"✓ Metrics saved to: {eval_path}")


if __name__ == '__main__':
    main()
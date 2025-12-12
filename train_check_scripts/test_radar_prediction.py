#!/usr/bin/env python
"""
Test Radar Point Cloud Prediction and Compare with Ground Truth

This script:
1. Loads trained model checkpoint
2. Runs inference on test samples
3. Compares predictions with ground truth
4. Visualizes results
5. Computes evaluation metrics

Usage:
    python test_radar_prediction.py \
        --config configs/bevdet/bevdet-r50-4d-cbgs-radar-frozen.py \
        --checkpoint work_dirs/radar_training/epoch_15.pth \
        --num-samples 10 \
        --visualize
"""

import argparse
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint


def load_model(config_path, checkpoint_path):
    """Load trained model from checkpoint.
    
    Args:
        config_path (str): Path to config file
        checkpoint_path (str): Path to checkpoint
        
    Returns:
        model: Loaded model in eval mode
        cfg: Config object
    """
    print(f"Loading config: {config_path}")
    cfg = Config.fromfile(config_path)
    
    print(f"Building model...")
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    model.eval()
    model.cuda()
    
    print("✓ Model loaded successfully")
    return model, cfg


def extract_radar_gt(info):
    """Extract ground truth radar points from info dict.
    
    Args:
        info (dict): Sample info
        
    Returns:
        np.ndarray: GT radar points (N, 3)
    """
    if 'radar_points' in info and info['radar_points'] is not None:
        radar_data = info['radar_points']
        if isinstance(radar_data, dict) and 'xyz' in radar_data:
            return radar_data['xyz']
        elif isinstance(radar_data, np.ndarray):
            return radar_data[:, :3]
    
    if 'radars' in info and info['radars'] is not None:
        all_points = []
        for sensor_data in info['radars'].values():
            if isinstance(sensor_data, dict) and 'xyz' in sensor_data:
                all_points.append(sensor_data['xyz'])
        if all_points:
            return np.concatenate(all_points, axis=0)
    
    return np.zeros((0, 3), dtype=np.float32)


def compute_chamfer_distance(pred_points, gt_points):
    """Compute Chamfer distance between prediction and GT.
    
    Args:
        pred_points (np.ndarray): Predicted points (N_pred, 3)
        gt_points (np.ndarray): GT points (N_gt, 3)
        
    Returns:
        dict: Chamfer distance metrics
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return {
            'chamfer_distance': float('inf'),
            'pred_to_gt': float('inf'),
            'gt_to_pred': float('inf'),
        }
    
    # Convert to torch tensors
    pred = torch.from_numpy(pred_points).float()
    gt = torch.from_numpy(gt_points).float()
    
    # Compute pairwise distances
    pred_expand = pred.unsqueeze(1)  # (N_pred, 1, 3)
    gt_expand = gt.unsqueeze(0)      # (1, N_gt, 3)
    
    dist = torch.sum((pred_expand - gt_expand) ** 2, dim=-1)  # (N_pred, N_gt)
    
    # Pred to GT: for each predicted point, find nearest GT
    min_dist_pred_to_gt = torch.min(dist, dim=1)[0]  # (N_pred,)
    pred_to_gt = torch.mean(torch.sqrt(min_dist_pred_to_gt)).item()
    
    # GT to Pred: for each GT point, find nearest prediction
    min_dist_gt_to_pred = torch.min(dist, dim=0)[0]  # (N_gt,)
    gt_to_pred = torch.mean(torch.sqrt(min_dist_gt_to_pred)).item()
    
    # Symmetric Chamfer distance
    chamfer = pred_to_gt + gt_to_pred
    
    return {
        'chamfer_distance': chamfer,
        'pred_to_gt': pred_to_gt,
        'gt_to_pred': gt_to_pred,
    }


def compute_coverage(pred_points, gt_points, threshold=1.0):
    """Compute coverage: % of GT points within threshold of any prediction.
    
    Args:
        pred_points (np.ndarray): Predicted points
        gt_points (np.ndarray): GT points
        threshold (float): Distance threshold (meters)
        
    Returns:
        float: Coverage percentage [0, 1]
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0
    
    pred = torch.from_numpy(pred_points).float()
    gt = torch.from_numpy(gt_points).float()
    
    pred_expand = pred.unsqueeze(1)
    gt_expand = gt.unsqueeze(0)
    
    dist = torch.sqrt(torch.sum((pred_expand - gt_expand) ** 2, dim=-1))
    min_dist_to_pred = torch.min(dist, dim=0)[0]
    
    coverage = (min_dist_to_pred < threshold).float().mean().item()
    return coverage


def visualize_comparison(pred_points, gt_points, title="Prediction vs Ground Truth", 
                        save_path=None, metrics=None):
    """Visualize predicted and GT point clouds side by side.
    
    Args:
        pred_points (np.ndarray): Predicted points (N_pred, 3)
        gt_points (np.ndarray): GT points (N_gt, 3)
        title (str): Plot title
        save_path (str): Path to save figure
        metrics (dict): Metrics to display
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create metric text
    metric_text = ""
    if metrics:
        metric_text = (f"Chamfer: {metrics['chamfer_distance']:.3f}m\n"
                      f"Pred→GT: {metrics['pred_to_gt']:.3f}m\n"
                      f"GT→Pred: {metrics['gt_to_pred']:.3f}m\n"
                      f"Coverage@1m: {metrics.get('coverage_1m', 0):.2%}")
    
    # 3D comparison
    ax1 = fig.add_subplot(231, projection='3d')
    if len(gt_points) > 0:
        ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                   c='blue', s=10, alpha=0.6, label='Ground Truth')
    if len(pred_points) > 0:
        ax1.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', s=10, alpha=0.6, label='Prediction')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Comparison')
    ax1.legend()
    
    # BEV (XY) comparison
    ax2 = fig.add_subplot(232)
    if len(gt_points) > 0:
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], c='blue', s=20, 
                   alpha=0.6, label=f'GT ({len(gt_points)} pts)')
    if len(pred_points) > 0:
        ax2.scatter(pred_points[:, 0], pred_points[:, 1], c='red', s=20, 
                   alpha=0.6, label=f'Pred ({len(pred_points)} pts)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Bird\'s Eye View (XY)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Side view (XZ)
    ax3 = fig.add_subplot(233)
    if len(gt_points) > 0:
        ax3.scatter(gt_points[:, 0], gt_points[:, 2], c='blue', s=20, alpha=0.6, label='GT')
    if len(pred_points) > 0:
        ax3.scatter(pred_points[:, 0], pred_points[:, 2], c='red', s=20, alpha=0.6, label='Pred')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # GT only (detailed)
    ax4 = fig.add_subplot(234, projection='3d')
    if len(gt_points) > 0:
        ax4.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                   c=gt_points[:, 2], cmap='viridis', s=15, alpha=0.8)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title(f'Ground Truth Only\n({len(gt_points)} points)')
    
    # Pred only (detailed)
    ax5 = fig.add_subplot(235, projection='3d')
    if len(pred_points) > 0:
        ax5.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c=pred_points[:, 2], cmap='plasma', s=15, alpha=0.8)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_zlabel('Z (m)')
    ax5.set_title(f'Prediction Only\n({len(pred_points)} points)')
    
    # Metrics text box
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    ax6.text(0.1, 0.5, metric_text, fontsize=14, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Evaluation Metrics')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def run_inference(model, dataset, indices, device='cuda'):
    """Run inference on dataset samples.
    
    Args:
        model: Trained model
        dataset: Test dataset
        indices: Sample indices to test
        device: Device to run on
        
    Returns:
        tuple: (predictions, ground_truths) - lists for each sample
    """
    predictions = []
    ground_truths = []
    
    print(f"\nRunning inference on {len(indices)} samples...")
    
    for idx in tqdm(indices):
        # Get GT from dataset info (NOT from pipeline)
        info = dataset.data_infos[idx]
        gt_points = extract_radar_gt(info)
        ground_truths.append(gt_points)
        
        # Get sample from dataset - this goes through the pipeline
        try:
            data = dataset[idx]
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            predictions.append(np.zeros((625, 3)))  # Empty prediction
            continue
        
        # Extract img_inputs and img_metas
        img_inputs_raw = data.get('img_inputs', None)
        img_metas_raw = data.get('img_metas', None)
        
        if img_inputs_raw is None:
            print(f"Warning: No img_inputs for sample {idx}")
            predictions.append(np.zeros((625, 3)))
            continue
        
        # Extract from DataContainer if wrapped, otherwise use directly
        if hasattr(img_inputs_raw, 'data'):
            img_inputs_list = img_inputs_raw.data
        else:
            img_inputs_list = img_inputs_raw
        
        if hasattr(img_metas_raw, 'data'):
            img_metas_list = img_metas_raw.data
        else:
            img_metas_list = img_metas_raw
        
        # Handle different formats:
        # 1. [[tensors...]] - batch wrapped
        # 2. [tensors...] - direct list
        # 3. Single tensor (less common)
        
        if isinstance(img_inputs_list, list) and len(img_inputs_list) > 0:
            first_elem = img_inputs_list[0]
            # If first element is a list, unwrap the batch
            if isinstance(first_elem, list):
                img_inputs = first_elem
            else:
                # First element is a tensor, use the list as is
                img_inputs = img_inputs_list
        else:
            img_inputs = img_inputs_list
        
        # Handle img_metas
        if isinstance(img_metas_list, list) and len(img_metas_list) > 0:
            first_elem = img_metas_list[0]
            if isinstance(first_elem, dict):
                img_metas = img_metas_list
            else:
                img_metas = first_elem
        else:
            img_metas = img_metas_list
        
        # Move tensors to device
        # img_inputs should be a list: [imgs, sensor2ego, ego2global, intrin, post_rot, post_tran, bda]
        if isinstance(img_inputs, list):
            img_inputs = [
                item.to(device) if torch.is_tensor(item) else item
                for item in img_inputs
            ]
        elif torch.is_tensor(img_inputs):
            img_inputs = img_inputs.to(device)
        else:
            print(f"Warning: Unexpected img_inputs type at sample {idx}: {type(img_inputs)}")
            predictions.append(np.zeros((625, 3)))
            continue
        
        # Ensure img_metas is a list
        if not isinstance(img_metas, list):
            img_metas = [img_metas]
        
        # Run inference
        try:
            with torch.no_grad():
                result = model.simple_test(
                    points=None,
                    img_metas=img_metas,
                    img=img_inputs
                )
            
            # Extract predicted radar points
            if len(result) > 0 and 'radar_points' in result[0]:
                pred_points = result[0]['radar_points']
                if isinstance(pred_points, torch.Tensor):
                    pred_points = pred_points.cpu().numpy()
                predictions.append(pred_points)
            else:
                print(f"Warning: No radar_points in result for sample {idx}")
                predictions.append(np.zeros((625, 3)))
                
        except Exception as e:
            print(f"Error during inference for sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append(np.zeros((625, 3)))
    
    return predictions, ground_truths


def main():
    parser = argparse.ArgumentParser(description='Test radar prediction model')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--data-root', default='data/nuscenes', help='Data root')
    parser.add_argument('--pkl-file', default='bevdetv3-nuscenes_infos_val.pkl',
                       help='Test data pkl file')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start index in dataset')
    parser.add_argument('--output-dir', default='test_results',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predicted point clouds')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    if args.save_predictions:
        pred_dir = os.path.join(args.output_dir, 'predictions')
        gt_dir = os.path.join(args.output_dir, 'ground_truth')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
    
    print("="*70)
    print("RADAR POINT CLOUD PREDICTION TEST")
    print("="*70)
    
    # Load model
    model, cfg = load_model(args.config, args.checkpoint)
    
    # Build dataset
    print(f"\nBuilding test dataset...")
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    print(f"✓ Dataset size: {len(dataset)}")
    
    # Get test indices
    end_index = min(args.start_index + args.num_samples, len(dataset))
    test_indices = list(range(args.start_index, end_index))
    print(f"Testing samples {args.start_index} to {end_index-1}")
    
    # Run inference
    predictions, ground_truths = run_inference(model, dataset, test_indices, args.device)
    
    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    all_metrics = []
    chamfer_distances = []
    coverages_1m = []
    coverages_2m = []
    
    for i, idx in enumerate(test_indices):
        # Get data
        data = dataset[idx]
        sample_token = data['img_metas'].data['sample_idx']
        
        # Get GT (already loaded from dataset.data_infos)
        gt_points = ground_truths[i]
        
        # Get prediction
        pred_points = predictions[i]
        
        print(f"Sample {i+1}/{len(test_indices)}: {sample_token}")
        print(f"  GT points: {len(gt_points)}")
        print(f"  Predicted points: {len(pred_points)}")
        
        # Compute metrics
        chamfer_metrics = compute_chamfer_distance(pred_points, gt_points)
        coverage_1m = compute_coverage(pred_points, gt_points, threshold=1.0)
        coverage_2m = compute_coverage(pred_points, gt_points, threshold=2.0)
        
        metrics = {
            **chamfer_metrics,
            'coverage_1m': coverage_1m,
            'coverage_2m': coverage_2m,
            'num_gt': len(gt_points),
            'num_pred': len(pred_points),
        }
        
        all_metrics.append(metrics)
        chamfer_distances.append(chamfer_metrics['chamfer_distance'])
        coverages_1m.append(coverage_1m)
        coverages_2m.append(coverage_2m)
        
        print(f"  Chamfer distance: {chamfer_metrics['chamfer_distance']:.3f}m")
        print(f"  Coverage@1m: {coverage_1m:.2%}")
        print(f"  Coverage@2m: {coverage_2m:.2%}")
        
        # Visualize
        if args.visualize:
            vis_path = os.path.join(vis_dir, f'{sample_token}.png')
            visualize_comparison(
                pred_points, gt_points,
                title=f'Sample {i+1}: {sample_token}',
                save_path=vis_path,
                metrics=metrics
            )
        
        # Save point clouds
        if args.save_predictions:
            np.save(os.path.join(pred_dir, f'{sample_token}.npy'), pred_points)
            np.save(os.path.join(gt_dir, f'{sample_token}.npy'), gt_points)
        
        print()
    
    # Summary statistics
    print(f"{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Chamfer Distance:")
    print(f"  Mean: {np.mean(chamfer_distances):.3f}m")
    print(f"  Std:  {np.std(chamfer_distances):.3f}m")
    print(f"  Min:  {np.min(chamfer_distances):.3f}m")
    print(f"  Max:  {np.max(chamfer_distances):.3f}m")
    print(f"  Median: {np.median(chamfer_distances):.3f}m")
    
    print(f"\nCoverage@1m:")
    print(f"  Mean: {np.mean(coverages_1m):.2%}")
    print(f"  Std:  {np.std(coverages_1m):.2%}")
    print(f"  Min:  {np.min(coverages_1m):.2%}")
    print(f"  Max:  {np.max(coverages_1m):.2%}")
    
    print(f"\nCoverage@2m:")
    print(f"  Mean: {np.mean(coverages_2m):.2%}")
    
    # Save results
    results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'num_samples': len(test_indices),
        'summary': {
            'chamfer_distance': {
                'mean': float(np.mean(chamfer_distances)),
                'std': float(np.std(chamfer_distances)),
                'min': float(np.min(chamfer_distances)),
                'max': float(np.max(chamfer_distances)),
                'median': float(np.median(chamfer_distances)),
            },
            'coverage_1m': {
                'mean': float(np.mean(coverages_1m)),
                'std': float(np.std(coverages_1m)),
            },
            'coverage_2m': {
                'mean': float(np.mean(coverages_2m)),
            },
        },
        'per_sample_metrics': all_metrics,
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Chamfer distance histogram
    axes[0, 0].hist(chamfer_distances, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(chamfer_distances), color='red', linestyle='--',
                      label=f'Mean: {np.mean(chamfer_distances):.3f}m')
    axes[0, 0].set_xlabel('Chamfer Distance (m)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Chamfer Distance Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coverage histogram
    axes[0, 1].hist(coverages_1m, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(coverages_1m), color='red', linestyle='--',
                      label=f'Mean: {np.mean(coverages_1m):.2%}')
    axes[0, 1].set_xlabel('Coverage@1m')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Coverage@1m Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Chamfer vs sample
    axes[1, 0].plot(chamfer_distances, 'o-')
    axes[1, 0].axhline(np.mean(chamfer_distances), color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Chamfer Distance (m)')
    axes[1, 0].set_title('Chamfer Distance per Sample')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coverage vs sample
    axes[1, 1].plot(coverages_1m, 'o-', color='green')
    axes[1, 1].axhline(np.mean(coverages_1m), color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Coverage@1m')
    axes[1, 1].set_title('Coverage@1m per Sample')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Evaluation Summary ({len(test_indices)} samples)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    summary_plot_path = os.path.join(args.output_dir, 'summary_plots.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Summary plots saved: {summary_plot_path}")
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    if args.visualize:
        print(f"  - Visualizations: {vis_dir}")
    if args.save_predictions:
        print(f"  - Predictions: {pred_dir}")
        print(f"  - Ground truth: {gt_dir}")
    print(f"  - Results: {results_path}")
    print(f"  - Summary plots: {summary_plot_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
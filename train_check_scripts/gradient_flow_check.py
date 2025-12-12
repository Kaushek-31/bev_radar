# tools/debug_gradient_flow.py
"""
Comprehensive gradient flow debugging for BEVDet4D_Radar model.
Tracks gradient propagation through all modules during training.
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from torch.utils.data import DataLoader
from mmcv.parallel import collate
import os

class GradientFlowTracker:
    """Track gradient flow through all modules."""
    
    def __init__(self, model, log_interval=1):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        
        # Storage for gradient statistics
        self.grad_stats = defaultdict(lambda: {
            'mean': [],
            'std': [],
            'max': [],
            'min': [],
            'norm': [],
            'zero_fraction': []
        })
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register backward hooks on all parameters."""
        print("\n" + "="*80)
        print("🔍 REGISTERING GRADIENT TRACKING HOOKS")
        print("="*80)
        
        module_count = 0
        param_count = 0
        
        for name, module in self.model.named_modules():
            # Skip empty modules
            if len(list(module.children())) > 0:
                continue
                
            # Register hook for parameters in this module
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{name}.{param_name}" if name else param_name
                    hook = param.register_hook(
                        self._make_hook(full_name, param)
                    )
                    self.hooks.append(hook)
                    param_count += 1
            
            module_count += 1
        
        print(f"✓ Registered hooks on {module_count} modules")
        print(f"✓ Tracking gradients for {param_count} parameters")
        print("="*80 + "\n")
    
    def _make_hook(self, name, param):
        """Create a gradient hook for a specific parameter."""
        def hook(grad):
            if grad is None:
                self.grad_stats[name]['zero_fraction'].append(1.0)
                return
            
            grad_data = grad.detach()
            
            # Compute statistics
            self.grad_stats[name]['mean'].append(grad_data.mean().item())
            self.grad_stats[name]['std'].append(grad_data.std().item())
            self.grad_stats[name]['max'].append(grad_data.max().item())
            self.grad_stats[name]['min'].append(grad_data.min().item())
            self.grad_stats[name]['norm'].append(grad_data.norm().item())
            
            # Fraction of zero gradients
            zero_frac = (grad_data.abs() < 1e-10).float().mean().item()
            self.grad_stats[name]['zero_fraction'].append(zero_frac)
            
        return hook
    
    def step(self):
        """Call after each backward pass."""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self._log_gradient_flow()
    
    def _log_gradient_flow(self):
        """Log gradient flow statistics."""
        print("\n" + "="*100)
        print(f"📊 GRADIENT FLOW ANALYSIS - Step {self.step_count}")
        print("="*100)
        
        # Categorize modules
        categories = {
            'img_backbone': [],
            'img_neck': [],
            'img_view_transformer': [],
            'img_bev_encoder_backbone': [],
            'img_bev_encoder_neck': [],
            'radar_head': [],
            'other': []
        }
        
        for name in self.grad_stats.keys():
            categorized = False
            for category in categories.keys():
                if category in name:
                    categories[category].append(name)
                    categorized = True
                    break
            if not categorized:
                categories['other'].append(name)
        
        # Print statistics by category
        for category, params in categories.items():
            if not params:
                continue
            
            print(f"\n{'='*100}")
            print(f"📦 {category.upper()} ({len(params)} parameters)")
            print(f"{'='*100}")
            
            # Compute aggregate statistics
            all_norms = []
            all_means = []
            all_stds = []
            all_zero_fracs = []
            no_grad_count = 0
            
            for param_name in params:
                stats = self.grad_stats[param_name]
                if stats['norm']:
                    norm = stats['norm'][-1]
                    all_norms.append(norm)
                    all_means.append(stats['mean'][-1])
                    all_stds.append(stats['std'][-1])
                    all_zero_fracs.append(stats['zero_fraction'][-1])
                else:
                    no_grad_count += 1
            
            # Print aggregate stats
            if all_norms:
                print(f"\n🔢 Aggregate Statistics:")
                print(f"  Mean gradient norm: {np.mean(all_norms):.6e}")
                print(f"  Median gradient norm: {np.median(all_norms):.6e}")
                print(f"  Max gradient norm: {np.max(all_norms):.6e}")
                print(f"  Min gradient norm: {np.min(all_norms):.6e}")
                print(f"  Mean zero fraction: {np.mean(all_zero_fracs):.4f}")
                print(f"  Parameters with NO gradients: {no_grad_count}/{len(params)}")
                
                # Check for issues
                if no_grad_count > 0:
                    print(f"\n⚠️  WARNING: {no_grad_count} parameters have NO gradients!")
                
                if np.mean(all_zero_fracs) > 0.9:
                    print(f"\n⚠️  WARNING: {np.mean(all_zero_fracs)*100:.1f}% of gradients are zero!")
                
                if np.max(all_norms) > 100:
                    print(f"\n⚠️  WARNING: Large gradient detected! Max norm = {np.max(all_norms):.2e}")
                
                if np.max(all_norms) < 1e-6:
                    print(f"\n⚠️  WARNING: Vanishing gradients! Max norm = {np.max(all_norms):.2e}")
                
                # Show top 5 parameters by gradient norm
                if len(params) > 5:
                    param_norms = [(p, self.grad_stats[p]['norm'][-1]) 
                                   for p in params if self.grad_stats[p]['norm']]
                    param_norms.sort(key=lambda x: x[1], reverse=True)
                    
                    print(f"\n🔝 Top 5 parameters by gradient norm:")
                    for i, (p, norm) in enumerate(param_norms[:5], 1):
                        # Shorten name for readability
                        short_name = p.split('.')[-2] + '.' + p.split('.')[-1] if '.' in p else p
                        print(f"  {i}. {short_name:40s}: {norm:.6e}")
                    
                    print(f"\n🔻 Bottom 5 parameters by gradient norm:")
                    for i, (p, norm) in enumerate(param_norms[-5:], 1):
                        short_name = p.split('.')[-2] + '.' + p.split('.')[-1] if '.' in p else p
                        print(f"  {i}. {short_name:40s}: {norm:.6e}")
            else:
                print(f"\n❌ NO GRADIENTS for any parameters in this category!")
        
        print("\n" + "="*100 + "\n")
    
    def get_summary(self):
        """Get summary of gradient flow over all steps."""
        summary = {}
        
        for name, stats in self.grad_stats.items():
            if stats['norm']:
                summary[name] = {
                    'mean_norm': np.mean(stats['norm']),
                    'max_norm': np.max(stats['norm']),
                    'min_norm': np.min(stats['norm']),
                    'mean_zero_fraction': np.mean(stats['zero_fraction']),
                    'num_steps': len(stats['norm'])
                }
        
        return summary
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def check_requires_grad(model):
    """Check which parameters require gradients."""
    print("\n" + "="*80)
    print("🔍 CHECKING REQUIRES_GRAD STATUS")
    print("="*80)
    
    categories = {
        'img_backbone': {'trainable': 0, 'frozen': 0},
        'img_neck': {'trainable': 0, 'frozen': 0},
        'img_view_transformer': {'trainable': 0, 'frozen': 0},
        'img_bev_encoder_backbone': {'trainable': 0, 'frozen': 0},
        'img_bev_encoder_neck': {'trainable': 0, 'frozen': 0},
        'radar_head': {'trainable': 0, 'frozen': 0},
        'other': {'trainable': 0, 'frozen': 0}
    }
    
    for name, param in model.named_parameters():
        categorized = False
        for category in categories.keys():
            if category in name:
                if param.requires_grad:
                    categories[category]['trainable'] += param.numel()
                else:
                    categories[category]['frozen'] += param.numel()
                categorized = True
                break
        
        if not categorized:
            if param.requires_grad:
                categories['other']['trainable'] += param.numel()
            else:
                categories['other']['frozen'] += param.numel()
    
    # Print summary
    total_trainable = 0
    total_frozen = 0
    
    for category, counts in categories.items():
        if counts['trainable'] + counts['frozen'] > 0:
            total = counts['trainable'] + counts['frozen']
            trainable_pct = 100 * counts['trainable'] / total if total > 0 else 0
            
            print(f"\n📦 {category}:")
            print(f"  Trainable: {counts['trainable']:>12,} ({trainable_pct:>5.1f}%)")
            print(f"  Frozen:    {counts['frozen']:>12,} ({100-trainable_pct:>5.1f}%)")
            print(f"  Total:     {total:>12,}")
            
            total_trainable += counts['trainable']
            total_frozen += counts['frozen']
    
    print(f"\n{'='*80}")
    print(f"🎯 TOTAL SUMMARY:")
    total = total_trainable + total_frozen
    print(f"  Trainable: {total_trainable:>12,} ({100*total_trainable/total:>5.1f}%)")
    print(f"  Frozen:    {total_frozen:>12,} ({100*total_frozen/total:>5.1f}%)")
    print(f"  Total:     {total:>12,}")
    print("="*80 + "\n")


def main():
    """Run gradient flow debugging."""
    
    # Config
    config_path = 'configs/bevdet/bevdet-r50-4d-cbgs-radar-frozen.py'
    checkpoint_path = 'work_dirs/bevdet-r50-4d-cbgs-radar-frozen/latest.pth'
    
    print("\n" + "="*80)
    print("🚀 GRADIENT FLOW DEBUGGING")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*80 + "\n")
    
    # Load config
    cfg = Config.fromfile(config_path)
    cfg.model.train_cfg = None
    
    # Build model
    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Using freshly initialized model")
    
    model = model.cuda()
    model.train()
    
    # Check requires_grad status
    check_requires_grad(model)
    
    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(cfg.data.train)
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: collate(x, samples_per_gpu=1)
    )
    
    # Create gradient tracker
    tracker = GradientFlowTracker(model, log_interval=1)
    
    # Run a few training iterations
    num_iterations = 3
    print(f"\n🏃 Running {num_iterations} training iterations...\n")
    
    data_iter = iter(data_loader)
    
    for i in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"⚙️  ITERATION {i+1}/{num_iterations}")
        print(f"{'='*80}")
        
        try:
            # Get a batch
            data_batch = next(data_iter)
        except StopIteration:
            print("⚠️  End of dataset reached")
            break
        
        # Extract data components
        img_inputs = data_batch.get('img_inputs', None)
        img_metas = data_batch.get('img_metas', None)
        gt_radar_points = data_batch.get('gt_radar_points', None)
        
        print(f"\n📦 Batch info:")
        if img_inputs is not None:
            if isinstance(img_inputs, list):
                print(f"  img_inputs: list of {len(img_inputs)} tensors")
                for j, inp in enumerate(img_inputs):
                    if isinstance(inp, torch.Tensor):
                        print(f"    [{j}]: {inp.shape} {inp.dtype} device={inp.device}")
            elif isinstance(img_inputs, torch.Tensor):
                print(f"  img_inputs: {img_inputs.shape} {img_inputs.dtype} device={img_inputs.device}")
        
        if img_metas is not None:
            if isinstance(img_metas, list):
                print(f"  img_metas: list of {len(img_metas)} items")
            else:
                print(f"  img_metas: {type(img_metas)}")
        
        if gt_radar_points is not None:
            if isinstance(gt_radar_points, torch.Tensor):
                print(f"  gt_radar_points: {gt_radar_points.shape} {gt_radar_points.dtype}")
            elif isinstance(gt_radar_points, list):
                print(f"  gt_radar_points: list of {len(gt_radar_points)} items")
                for j, pts in enumerate(gt_radar_points[:3]):  # Show first 3
                    if isinstance(pts, torch.Tensor):
                        print(f"    [{j}]: {pts.shape}")
                    else:
                        print(f"    [{j}]: {type(pts)}")
        
        # Move data to GPU
        if img_inputs is not None:
            if isinstance(img_inputs, ( list, tuple )):
                img_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x 
                             for x in img_inputs]
            elif isinstance(img_inputs, torch.Tensor):
                img_inputs = img_inputs.cuda()
        
        if gt_radar_points is not None:
            if isinstance(gt_radar_points, torch.Tensor):
                gt_radar_points = gt_radar_points.cuda()
            elif isinstance(gt_radar_points, list):
                gt_radar_points = [
                    torch.as_tensor(x, dtype=torch.float32).cuda() if not isinstance(x, torch.Tensor)
                    else x.cuda()
                    for x in gt_radar_points
                ]
        
        # Forward pass
        print(f"\n🔄 Forward pass...")
        
        try:
            losses = model.forward_train(
                img_inputs=img_inputs,
                gt_radar_points=gt_radar_points,
                img_metas=img_metas,
                points=None
            )
            
            # Print losses
            print(f"\n💰 Losses:")
            total_loss = 0
            for key, value in losses.items():
                loss_val = value.item() if isinstance(value, torch.Tensor) else value
                print(f"  {key}: {loss_val:.6f}")
                if 'loss' in key and isinstance(value, torch.Tensor):
                    total_loss = total_loss + value if isinstance(total_loss, torch.Tensor) else value
            
            if isinstance(total_loss, torch.Tensor):
                print(f"  TOTAL: {total_loss.item():.6f}")
            else:
                print(f"  TOTAL: {total_loss:.6f}")
            
            # Backward pass
            print(f"\n⬅️  Backward pass...")
            if isinstance(total_loss, torch.Tensor):
                total_loss.backward()
            else:
                print("⚠️  No tensor to backpropagate!")
                continue
            
            # Track gradients
            tracker.step()
            
            # Clear gradients
            model.zero_grad()
            
        except Exception as e:
            print(f"\n❌ Error during iteration {i+1}:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Print final summary
    print("\n" + "="*80)
    print("📈 GRADIENT FLOW SUMMARY OVER ALL ITERATIONS")
    print("="*80)
    
    summary = tracker.get_summary()
    
    if not summary:
        print("\n⚠️  No gradient statistics collected!")
    else:
        # Group by category
        categories = defaultdict(list)
        for name, stats in summary.items():
            categorized = False
            for category in ['img_backbone', 'img_neck', 'img_view_transformer',
                            'img_bev_encoder_backbone', 'img_bev_encoder_neck', 'radar_head']:
                if category in name:
                    categories[category].append((name, stats))
                    categorized = True
                    break
            if not categorized:
                categories['other'].append((name, stats))
        
        for category, params in categories.items():
            if not params:
                continue
            
            print(f"\n{category.upper()}:")
            avg_norms = [stats['mean_norm'] for _, stats in params]
            if avg_norms:
                print(f"  Avg gradient norm: {np.mean(avg_norms):.6e}")
                print(f"  Max gradient norm: {np.max(avg_norms):.6e}")
                print(f"  Min gradient norm: {np.min(avg_norms):.6e}")
    
    # Cleanup
    tracker.remove_hooks()
    
    print("\n" + "="*80)
    print("✅ GRADIENT FLOW DEBUGGING COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
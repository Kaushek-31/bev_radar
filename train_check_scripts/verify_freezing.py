#!/usr/bin/env python
"""
Verification script to check if model freezing is working correctly.

Usage:
    python verify_freezing.py configs/bevdet4d/config_bevdet4d_radar_frozen.py
"""

import argparse
import torch
from mmcv import Config
from mmdet3d.models import build_model


def verify_freezing(config_path):
    """Verify that freezing is working correctly.
    
    Args:
        config_path (str): Path to config file
    """
    print("=" * 70)
    print("VERIFYING MODEL FREEZING")
    print("=" * 70)
    
    # Load config
    print(f"\n1. Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Check config settings
    print(f"\n2. Config settings:")
    print(f"   - freeze_backbone: {cfg.model.get('freeze_backbone', False)}")
    print(f"   - pretrained: {cfg.model.get('pretrained', 'None')}")
    
    # Build model
    print(f"\n3. Building model...")
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    
    # IMPORTANT: Initialize weights (this triggers freezing)
    print(f"\n4. Initializing weights (this will load pretrained and freeze)...")
    model.init_weights()
    
    # Analyze parameters
    print(f"\n5. Analyzing parameters...")
    print("=" * 70)
    
    # Categorize parameters
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))
    
    # Summary statistics
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen = total_params - total_trainable
    
    print(f"\n{'='*70}")
    print(f"PARAMETER SUMMARY")
    print(f"{'='*70}")
    print(f"Total parameters:      {total_params:>15,}")
    print(f"Trainable parameters:  {total_trainable:>15,} ({100*total_trainable/total_params:>5.2f}%)")
    print(f"Frozen parameters:     {total_frozen:>15,} ({100*total_frozen/total_params:>5.2f}%)")
    print(f"{'='*70}")
    
    # Check trainable parameters
    print(f"\n{'='*70}")
    print(f"TRAINABLE PARAMETERS (should be radar_head only)")
    print(f"{'='*70}")
    
    radar_head_found = False
    non_radar_trainable = []
    
    for name, numel in trainable_params:
        if 'radar_head' in name:
            radar_head_found = True
        else:
            non_radar_trainable.append(name)
    
    if radar_head_found:
        print(f"✓ radar_head is trainable ({sum(n for _, n in trainable_params if 'radar_head' in _):,} params)")
    else:
        print(f"✗ WARNING: radar_head is NOT trainable!")
    
    if non_radar_trainable:
        print(f"\n✗ WARNING: Found {len(non_radar_trainable)} non-radar trainable parameters:")
        for name in non_radar_trainable[:10]:
            print(f"    - {name}")
        if len(non_radar_trainable) > 10:
            print(f"    ... and {len(non_radar_trainable) - 10} more")
    else:
        print(f"✓ No non-radar trainable parameters (correct!)")
    
    # Check frozen parameters
    print(f"\n{'='*70}")
    print(f"FROZEN PARAMETERS (should be backbone, neck, etc.)")
    print(f"{'='*70}")
    
    # Group frozen params by module
    frozen_by_module = {}
    for name, numel in frozen_params:
        module = name.split('.')[0]
        if module not in frozen_by_module:
            frozen_by_module[module] = 0
        frozen_by_module[module] += numel
    
    expected_frozen = [
        'img_backbone',
        'img_neck',
        'img_view_transformer',
        'img_bev_encoder_backbone',
        'img_bev_encoder_neck',
        'pre_process'
    ]
    
    for module in expected_frozen:
        if module in frozen_by_module:
            print(f"✓ {module:30s}: {frozen_by_module[module]:>12,} params frozen")
        else:
            print(f"✗ {module:30s}: NOT FOUND (may not exist in model)")
    
    # Check for unexpected frozen modules
    unexpected_frozen = set(frozen_by_module.keys()) - set(expected_frozen)
    if 'radar_head' in unexpected_frozen:
        print(f"\n✗ ERROR: radar_head is frozen! It should be trainable!")
    
    # Verification result
    print(f"\n{'='*70}")
    print(f"VERIFICATION RESULT")
    print(f"{'='*70}")
    
    success = True
    
    # Check 1: radar_head is trainable
    if not radar_head_found:
        print(f"✗ FAIL: radar_head is not trainable")
        success = False
    else:
        print(f"✓ PASS: radar_head is trainable")
    
    # Check 2: No non-radar params are trainable
    if non_radar_trainable:
        print(f"✗ FAIL: Found {len(non_radar_trainable)} unexpected trainable parameters")
        success = False
    else:
        print(f"✓ PASS: Only radar_head is trainable")
    
    # Check 3: Expected modules are frozen
    all_frozen = all(m in frozen_by_module for m in expected_frozen if hasattr(model, m))
    if all_frozen:
        print(f"✓ PASS: All expected modules are frozen")
    else:
        print(f"✗ FAIL: Some expected modules are not frozen")
        success = False
    
    # Check 4: Reasonable ratio
    if total_trainable < total_params * 0.1:  # Less than 10% trainable
        print(f"✓ PASS: Trainable params ratio is reasonable ({100*total_trainable/total_params:.2f}%)")
    else:
        print(f"✗ WARNING: Trainable params ratio seems high ({100*total_trainable/total_params:.2f}%)")
    
    print(f"{'='*70}")
    if success:
        print(f"✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print(f"Model is correctly configured for training with frozen backbone!")
    else:
        print(f"✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print(f"Please review the warnings above.")
    print(f"{'='*70}\n")
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Verify model freezing')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    
    verify_freezing(args.config)


if __name__ == '__main__':
    main()
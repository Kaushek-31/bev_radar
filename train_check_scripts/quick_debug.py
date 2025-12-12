import torch
import numpy as np
from mmdet3d.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint

cfg = Config.fromfile('configs/bevdet/bevdet-r50-4d-cbgs-radar-frozen.py')
cfg.model.train_cfg = None

model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, 'work_dirs/bevdet-r50-4d-cbgs-radar-frozen/latest.pth', map_location='cpu')

print("\n" + "="*70)
print("🔍 QUICK DEBUG - Model Output Check")
print("="*70)

# Create dummy input with correct format
B = 1
N_cams = 6
H, W = 256, 704

# BEVDet4D expects temporal frames
num_frames = 2  # Current + 1 previous

# Images for all frames
img = torch.randn(B, N_cams * num_frames, 3, H, W)

# Camera parameters (per frame, per camera)
sensor2ego = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N_cams * num_frames, 1, 1).float()
ego2global = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N_cams * num_frames, 1, 1).float()
intrinsic = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N_cams * num_frames, 1, 1).float()
intrinsic[:, :, 0, 0] = 1000.0  # fx
intrinsic[:, :, 1, 1] = 1000.0  # fy
intrinsic[:, :, 0, 2] = W / 2   # cx
intrinsic[:, :, 1, 2] = H / 2   # cy

post_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N_cams * num_frames, 1, 1).float()
post_tran = torch.zeros(B, N_cams * num_frames, 3).float()

# BDA matrix (4x4 for BEVDet4D)
bda = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).float()

img_inputs = [img, sensor2ego, ego2global, intrinsic, post_rot, post_tran, bda]
img_inputs = [item.cuda() for item in img_inputs]
model = model.cuda()
print(f"\n📦 Input shapes:")
print(f"  img: {img.shape}")
print(f"  sensor2ego: {sensor2ego.shape}")
print(f"  ego2global: {ego2global.shape}")
print(f"  intrinsic: {intrinsic.shape}")
print(f"  post_rot: {post_rot.shape}")
print(f"  post_tran: {post_tran.shape}")
print(f"  bda: {bda.shape}")

# Minimal img_metas
img_metas = [{
    'sample_idx': 'test_sample',
    'timestamp': 0.0,
}]

model.eval()
print("\n🚀 Running inference...")

try:
    with torch.no_grad():
        result = model.simple_test(points=None, img_metas=img_metas, img=img_inputs)
        
        print("\n✓ Inference successful!")
        print(f"\n📊 Result structure:")
        print(f"  Type: {type(result)}")
        print(f"  Length: {len(result)}")
        print(f"  Keys: {result[0].keys()}")
        
        if 'radar_points' in result[0]:
            radar_points = result[0]['radar_points']
            print(f"\n🎯 Radar predictions:")
            print(f"  Type: {type(radar_points)}")
            
            if radar_points is None:
                print(f"  ⚠️  Radar points are None!")
            elif isinstance(radar_points, torch.Tensor):
                print(f"  Shape: {radar_points.shape}")
                print(f"  Device: {radar_points.device}")
                print(f"  Dtype: {radar_points.dtype}")
                
                if radar_points.numel() == 0:
                    print(f"  ⚠️  Tensor is empty (0 elements)!")
                else:
                    print(f"  Range: [{radar_points.min():.3f}, {radar_points.max():.3f}]")
                    print(f"  Mean: {radar_points.mean(dim=0)}")
                    print(f"  Std: {radar_points.std(dim=0)}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(radar_points).any():
                        print(f"  🚨 Contains NaN values!")
                        print(f"     NaN count: {torch.isnan(radar_points).sum().item()}")
                    
                    if torch.isinf(radar_points).any():
                        print(f"  🚨 Contains Inf values!")
                        print(f"     Inf count: {torch.isinf(radar_points).sum().item()}")
                    
                    # Sample some points
                    if radar_points.shape[0] > 0:
                        print(f"\n  First 5 predictions:")
                        for i in range(min(5, radar_points.shape[0])):
                            pt = radar_points[i]
                            print(f"    Point {i}: [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]")
            else:
                print(f"  Unexpected type: {type(radar_points)}")
                print(f"  Value: {radar_points}")
        else:
            print(f"\n⚠️  'radar_points' not in result!")
            print(f"  Available keys: {list(result[0].keys())}")

except Exception as e:
    print(f"\n❌ Error during inference:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)

# Additional diagnostics
print("\n🔧 Model diagnostics:")
if hasattr(model, 'radar_head'):
    print(f"  Has radar_head: ✓")
    radar_head = model.radar_head
    
    if hasattr(radar_head, 'num_points'):
        print(f"  Num points: {radar_head.num_points}")
    
    if hasattr(radar_head, 'confidence_target'):
        print(f"  Confidence target: {radar_head.confidence_target}")
    
    # Check if radar head is in eval mode
    print(f"  Radar head training mode: {radar_head.training}")
else:
    print(f"  Has radar_head: ✗")

print("="*70)
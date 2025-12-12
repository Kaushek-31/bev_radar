# check_checkpoint.py
import torch

ckpt = torch.load('work_dirs/bevdet-r50-4d-cbgs-radar-frozen/latest.pth', map_location='cpu')

print("State dict keys:")
radar_keys = [k for k in ckpt['state_dict'].keys() if 'radar_head' in k]

if len(radar_keys) == 0:
    print("❌ NO RADAR HEAD WEIGHTS IN CHECKPOINT!")
    print("   The model was never trained or checkpoint doesn't include radar head")
else:
    print(f"✓ Found {len(radar_keys)} radar head parameters")
    
    # Check a few weights
    for key in radar_keys[:5]:
        param = ckpt['state_dict'][key]
        print(f"  {key}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
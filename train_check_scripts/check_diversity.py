import torch
import numpy as np
from mmdet3d.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint

cfg = Config.fromfile('configs/bevdet/bevdet-r50-4d-cbgs-radar-frozen.py')
cfg.model.train_cfg = None

model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, 'work_dirs/bevdet-r50-4d-cbgs-radar-frozen/latest.pth', map_location='cpu')
model = model.cuda()
model.eval()

print("\n" + "="*70)
print("🔄 DIVERSITY TEST - Multiple Inputs")
print("="*70)

if hasattr(model, 'radar_head'):
    B = 1
    C = 256
    H = 128
    W = 128
    
    # Test with 3 different random inputs
    predictions = []
    
    for test_id in range(3):
        print(f"\n🧪 Test {test_id + 1}:")
        
        # Different random seed for each test
        torch.manual_seed(test_id)
        dummy_bev_feat = torch.randn(B, C, H, W).cuda()
        
        print(f"  BEV feat stats: mean={dummy_bev_feat.mean():.4f}, std={dummy_bev_feat.std():.4f}")
        
        with torch.no_grad():
            result = model.radar_head([dummy_bev_feat])
            
            if len(result) > 0:
                pred_dict = result[0]
                if isinstance(pred_dict, list):
                    pred_dict = pred_dict[0]
                
                if 'radar_points' in pred_dict:
                    radar_points = pred_dict['radar_points']
                    predictions.append(radar_points.cpu().numpy())
                    radar_points = radar_points.detach().cpu().numpy()
                    print(f"  Radar points shape: {radar_points.shape}")
                    # mean_vals = radar_points.mean(axis=0).astype(float)
                    # std_vals  = radar_points.std(axis=0).astype(float)

                    print("  Mean:", radar_points.mean(axis=0))
                    print("  Std:", radar_points.std(axis=0))
    
    # Compare predictions
    if len(predictions) >= 2:
        print("\n" + "="*70)
        print("📊 DIVERSITY ANALYSIS")
        print("="*70)
        
        for i in range(len(predictions) - 1):
            pred1 = predictions[i]
            pred2 = predictions[i + 1]
            
            # Compare first N points
            N = min(pred1.shape[0], pred2.shape[0], 100)
            
            diff_mean = np.abs(pred1[:N] - pred2[:N]).mean()
            diff_max = np.abs(pred1[:N] - pred2[:N]).max()
            
            print(f"\nTest {i+1} vs Test {i+2}:")
            print(f"  Mean difference: {diff_mean:.6f}")
            print(f"  Max difference:  {diff_max:.6f}")
            
            if diff_mean < 0.001:
                print(f"  🚨 CRITICAL: Predictions are identical!")
                print(f"     The model is not responding to input variations.")
                print(f"     This confirms MODEL COLLAPSE.")
            elif diff_mean < 0.1:
                print(f"  ⚠️  WARNING: Predictions are very similar")
            else:
                print(f"  ✓ Predictions show healthy variation")

print("\n" + "="*70)
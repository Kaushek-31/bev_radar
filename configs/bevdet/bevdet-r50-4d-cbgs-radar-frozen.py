# Copyright (c) Phigent Robotics. All rights reserved.

# BEVDet4D Radar Point Regression - WITH PRETRAINED FREEZING
# Loads pretrained BEVDet4D backbone, freezes it, only trains radar_head

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']


pc_x_lim = [-51.2, 51.2]
pc_y_lim = [-51.2, 51.2]
pc_z_lim = [-5.0, 3.0]

# ============ Global Settings ============
point_cloud_range = [pc_x_lim[0], pc_y_lim[0], pc_z_lim[0], pc_x_lim[1], pc_y_lim[1], pc_z_lim[1]]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# ============ Data Config ============
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 1+1, 1)

# ============ Model Config ============
RADAR_POINT_COUNT = 2000  # Number of radar points to predict
GT_RADAR_POINT_COUNT = 500  # Number of radar points to predict

model = dict(
    type='BEVDet4D_Radar',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    
    # ========== PRETRAINED & FREEZING ==========
    freeze_backbone=True,  # Freeze all except radar_head
    pretrained='./checkpoints/bevdet-dev2.1/bevdet-r50-4d-cbgs.pth',  # Path to pretrained BEVDet4D
    # NOTE: Change this path to your actual pretrained checkpoint!
    # ==========================================
    
    img_backbone=dict(
        pretrained='torchvision://resnet50',  # Only used if training from scratch
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),

    radar_head=dict(
        type='SimpleCNNRadarHead',
        in_channels=256,
        num_points=RADAR_POINT_COUNT,
        point_cloud_range=point_cloud_range,
        hidden_channels=[512, 256, 128],
        use_multi_scale=False,
        fusion_method='concat',
        use_spatial_attention=True,
        max_offset=3.0,  # INCREASED from 1.5
        use_probabilistic=False,
        confidence_target=0.3,  # NEW: Target average confidence
        confidence_reg_weight=0.1,  # NEW: Confidence regularization weight
        loss_weight=0.02,
        diversity_loss_weight=0.1,  # INCREASED from 0.0
        visualize_training=True,
        vis_interval=100
    )

)

# ============ Dataset Config ============
dataset_type = 'NuScenesDatasetRadar'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# ============ Training Pipeline ============
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),

    dict(type='LoadAnnotations'),

    dict(
        type='LoadRadarPointsXYZ',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=[0, 1, 2]),
    
    dict(
        type='RadarPointsRangeFilter',
        point_cloud_range=point_cloud_range),
    
    dict(
        type='RadarPointsSampler',
        num_points=GT_RADAR_POINT_COUNT,
        sample_method='random'),
    
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_radar_points'],
        meta_keys=['sample_idx', 'timestamp'])
]

# ============ Testing Pipeline ============
test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=data_config,
        sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='LoadRadarPointsXYZ',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=[0, 1, 2]),
    
    dict(
        type='RadarPointsRangeFilter',
        point_cloud_range=point_cloud_range),
    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs'])
        ]),
]

# ============ Input Modality ============
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)

# ============ Data Loaders ============
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR',
            img_info_prototype='bevdet4d',
            multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
            load_radar_data=True,
            radar_format='merged',
            max_radar_points=GT_RADAR_POINT_COUNT,
        )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        load_radar_data=True,
        radar_format='merged',
        max_radar_points=GT_RADAR_POINT_COUNT,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        load_radar_data=True,
        radar_format='merged',
        max_radar_points=GT_RADAR_POINT_COUNT,
    ))

# ============ Optimization ============
# Higher LR since only training radar_head (small portion of model)
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # Increased from 2e-4 since we're only training radar_head
    weight_decay=1e-2)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# Faster LR decay since training converges faster with frozen backbone
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=20)


# ============ Hooks ============
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=2,
    ),
]

# ============ Evaluation ============
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    metric='radar_points',
)

# ============ Logging ============
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# ============ Checkpoint ============
checkpoint_config = dict(interval=5)

# ============ FP16 (Optional) ============
# Faster training with frozen backbone
# fp16 = dict(loss_scale='dynamic')

# ============ Task Flags ============
use_radar = True
radar_regression = True
multi_task_learning = False
use_pretrained_frozen = True
find_unused_parameters = True








    # # Radar point regression head - ONLY TRAINABLE PART
    # radar_head=dict(
    #     type='RadarPointRegressionHead',
    #     in_channels=256,
    #     num_points=RADAR_POINT_COUNT,
    #     point_dim=3,
    #     hidden_channels=[512, 256, 128],
    #     point_cloud_range=point_cloud_range,
    #     loss_type='chamfer',
    #     loss_weight=1.0,
    #     diversity_weight=0.1,       # Spread predictions
    #     coverage_weight=0.5,        # Cover GT points
    #     l1_weight=0.1,              # L1 loss weight
    #     l2_weight=0.1,              # L2 loss weight
    #     min_distance=0.5,           # Min separation (meters)
    #     coverage_threshold=1.5,     # Coverage radius (meters)
    #     bev_h=128, bev_w=128,       # NEW
    #     use_spatial=True,            # NEW (key feature!)
    #     num_spatial_groups=64,       # NEW (8×8 grid)
    # ),

    # radar_head = dict(
    #     type='RadarPointRegressionHead',
    #     in_channels=256,
    #     num_points=1024,
    #     point_dim=3,
    #     bev_h=128,
    #     bev_w=128,
    #     hidden_channels=[512, 256, 128],
    #     point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    #     loss_type='chamfer',
    #     loss_weight=1.0,
        
    #     # Anti-clustering losses
    #     diversity_weight=0.1,
    #     coverage_weight=0.5,
    #     min_distance=0.5,
    #     coverage_threshold=1.5,
        
    #     # Spatial mode
    #     use_spatial=True,
    #     num_spatial_groups=64,  # 8x8 grid
        
    #     enable_viz=False,  # Set True to save visualizations
    # )

    # ============================================================================
    # APPROACH 2: Heatmap-Based (Recommended)
    # ============================================================================
    # Good for: Best performance, production, variable points
    # Speed: Fast (95 FPS)
    # Points: Variable (50-800 per scene)

    # radar_head = dict(
    #     type='RadarPointHeatmapHead',
    #     in_channels=256,
    #     bev_h=128,
    #     bev_w=128,
    #     point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        
    #     # Point extraction
    #     min_confidence=0.3,      # Lower = more points, Higher = fewer points
    #     nms_radius=2,            # NMS kernel size
    #     gaussian_radius=2,       # Gaussian for targets
        
    #     # Loss weights
    #     loss_conf_weight=1.0,
    #     loss_offset_weight=0.5,
    #     loss_height_weight=0.5,
        
    #     enable_viz=True,
    #     visualize_training=True,
    # )

    # ============================================================================
    # APPROACH 3: Attention-Based
    # ============================================================================
    # Good for: Research, complex scenes, long-range dependencies
    # Speed: Slower (60 FPS)
    # Points: Variable (100-800 per scene)
    # Memory: High

    # radar_head = dict(
    #     type='RadarPointAttentionHead',
    #     in_channels=256,
    #     hidden_dim=256,
    #     num_attention_heads=8,
    #     max_points=1000,
    #     topk_ratio=0.1,          # Select top 10% of locations
    #     confidence_threshold=0.5,
    #     point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        
    #     enable_viz=True,
    #     visualize_training=True,
    # )

    # # ============================================================================
    # # APPROACH 4: Hybrid (Attention + Heatmap)
    # # ============================================================================
    # # Good for: Best results, research papers, when you want it all
    # # Speed: Medium (80 FPS)
    # # Points: Variable (50-800 per scene)
    # # Memory: Medium-High

    # radar_head = dict(
    #     type='RadarPointHybridHead',
    #     in_channels=256,
    #     hidden_dim=256,
    #     num_attention_heads=8,
    #     bev_h=128,
    #     bev_w=128,
    #     point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        
    #     # Heatmap extraction
    #     min_confidence=0.3,
    #     nms_radius=2,
    #     gaussian_radius=2,
        
    #     # Loss weights
    #     loss_conf_weight=1.0,
    #     loss_offset_weight=0.5,
    #     loss_height_weight=0.5,
        
    #     enable_viz=False,
    # )


    # ============================================================================
    # APPROACH 5: DETR-Style Query-Based Head
    # ============================================================================

    # radar_head = dict(
    #     type='RadarPointQueryHead',
    #     in_channels=256,
    #     embed_dims=256,
    #     num_decoder_layers=6,
    #     num_heads=8,
    #     num_foreground_queries=1000,
    #     point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    #     visualize_training=True,  # See what's happening!
    # )

    # radar_head = dict(
    #     type='SimpleQueryRadarPointHead',
    #     in_channels=256,
    #     embed_dims=256,
    #     num_queries=500,
    #     visualize_training=True,  # Enable visualization
    # )
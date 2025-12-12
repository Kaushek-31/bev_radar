# Copyright (c) Phigent Robotics. All rights reserved.

# BEVDet4D Radar Point Regression Config - RADAR ONLY
# Camera images → BEV features → Predict 625 radar points (x,y,z)
# NO 3D detection head (radar prediction only)

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# ============ Global Settings ============
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

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
model = dict(
    type='BEVDet4D_Radar',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    
    # Image backbone (ResNet50)
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    
    # FPN neck
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    
    # LSS view transformer (image → BEV)
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    
    # BEV encoder backbone
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    
    # BEV encoder neck
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    
    # Pre-process net for temporal fusion
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    
    # Radar point regression head (ONLY HEAD)
    radar_head=dict(
        type='RadarPointRegressionHead',
        in_channels=256,                    # From BEV neck output
        num_points=625,                     # Fixed number of points to predict
        point_dim=3,                        # XYZ coordinates
        hidden_channels=[512, 256, 128],    # Hidden layer dimensions
        point_cloud_range=point_cloud_range,
        loss_type='chamfer',                # Chamfer distance loss
        loss_weight=1.0,
    ),
    # NO pts_bbox_head - radar prediction only
    # NO train_cfg/test_cfg - not needed for regression only
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
    # Load images
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadAnnotations'),
    # Load radar points (XYZ only)
    dict(
        type='LoadRadarPointsXYZ',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=[0, 1, 2]),
    
    # Filter radar points within range
    dict(
        type='RadarPointsRangeFilter',
        point_cloud_range=point_cloud_range),
    
    # Sample fixed number of radar points
    dict(
        type='RadarPointsSampler',
        num_points=625,  # Match model output size
        sample_method='random'),  # or 'fps' for farthest point sampling
    
    # BEV augmentation
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    
    # Format bundle
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    
    # Collect keys for model input (ONLY radar, no detection)
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
                keys=['img_inputs', 'gt_radar_points'])
        ])
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
    samples_per_gpu=4,
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
            max_radar_points=1000,
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
        max_radar_points=1000,
    ),
    test=dict(
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
        max_radar_points=1000,
    ))

# ============ Optimization ============
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-2)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[16, 19])

# ============ Runtime ============
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
    metric='radar_points',  # ONLY radar evaluation
)

# ============ Logging ============
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# ============ Checkpoint ============
checkpoint_config = dict(interval=1)

# ============ FP16 ============
# fp16 = dict(loss_scale='dynamic')

# ============ Task Flags ============
use_radar = True
radar_regression = True
multi_task_learning = False  # Radar prediction only
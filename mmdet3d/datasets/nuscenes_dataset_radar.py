# Copyright (c) OpenMMLab. All rights reserved.
"""
NuScenes Dataset with Radar Point Ground Truth Support
Loads radar XYZ coordinates for training radar point regression models.
"""

import numpy as np
from mmdet.datasets import DATASETS
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesDatasetRadar(NuScenesDataset):
    """NuScenes Dataset for Radar Point Regression.
    
    Extends standard NuScenesDataset to load radar point cloud ground truth.
    Radar points are loaded as (N, 3) arrays containing XYZ coordinates.
    
    Args:
        load_radar_data (bool): Whether to load radar point clouds
        radar_format (str): Format of radar data ('merged' or 'individual')
        max_radar_points (int): Maximum number of radar points to load per sample
        All other args inherited from NuScenesDataset
    """
    
    def __init__(self,
                 load_radar_data=True,
                 radar_format='merged',
                 max_radar_points=1000,
                 **kwargs):
        self.load_radar_data = load_radar_data
        self.radar_format = radar_format
        self.max_radar_points = max_radar_points
        
        super(NuScenesDatasetRadar, self).__init__(**kwargs)
        
        # Update modality
        if self.load_radar_data and self.modality is not None:
            self.modality['use_radar'] = True
    
    def get_data_info(self, index):
        """Get data info with radar points.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Data information including radar_info
        """
        info = self.data_infos[index]
        
        # Standard information
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info.get('lidar_path', ''),
            sweeps=info.get('sweeps', []),
            timestamp=info['timestamp'] / 1e6,
        )
        
        # Add annotation info if present
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        
        # ===== Load Radar Point Cloud =====
        if self.load_radar_data:
            radar_info = {}
            
            # Load merged radar points (all 5 sensors combined)
            if self.radar_format in ['merged', 'both']:
                if 'radar_points' in info and info['radar_points'] is not None:
                    radar_data = info['radar_points']
                    
                    # Extract XYZ coordinates only
                    if isinstance(radar_data, dict) and 'xyz' in radar_data:
                        radar_xyz = radar_data['xyz']  # (N, 3)
                    elif isinstance(radar_data, np.ndarray):
                        radar_xyz = radar_data[:, :3]  # Assume first 3 cols are XYZ
                    else:
                        radar_xyz = np.zeros((0, 3), dtype=np.float32)
                    
                    # Limit number of points if needed
                    if len(radar_xyz) > self.max_radar_points:
                        # Randomly sample points
                        indices = np.random.choice(
                            len(radar_xyz), self.max_radar_points, replace=False
                        )
                        radar_xyz = radar_xyz[indices]
                    
                    radar_info['radar_points'] = radar_xyz.astype(np.float32)
                else:
                    # No radar data available
                    radar_info['radar_points'] = np.zeros((0, 3), dtype=np.float32)
            
            # Load individual radar sensors
            if self.radar_format in ['individual', 'both']:
                if 'radars' in info and info['radars'] is not None:
                    radars_dict = {}
                    for sensor_name, sensor_data in info['radars'].items():
                        if isinstance(sensor_data, dict) and 'xyz' in sensor_data:
                            radars_dict[sensor_name] = sensor_data['xyz'].astype(np.float32)
                        else:
                            radars_dict[sensor_name] = np.zeros((0, 3), dtype=np.float32)
                    radar_info['radars'] = radars_dict
                else:
                    radar_info['radars'] = {}
            
            input_dict['radar_info'] = radar_info
        
        # Camera information
        if self.modality.get('use_camera', False):
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    
                    # Lidar to image transformation
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = viewpad @ lidar2cam_rt.T
                    lidar2img_rts.append(lidar2img_rt)
                
                input_dict.update(dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                ))
                
                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                # BEVDet format
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
        
        return input_dict
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        
        Args:
            idx (int): Index of data
            
        Returns:
            dict: Training/test data (with radar_points as Tensor)
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def prepare_train_data(self, index):
        """Prepare training data.
        
        Args:
            index (int): Index for accessing the target data
            
        Returns:
            dict: Training data dict with radar_points
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        # Apply pipeline
        example = self.pipeline(input_dict)
        
        # For radar-only mode, check if we have radar points
        # For detection mode, check if we have valid labels
        if self.filter_empty_gt:
            if 'gt_labels_3d' in example:
                # Detection mode - filter empty detections
                if example is None or ~(example['gt_labels_3d'] != -1).any():
                    return None
            elif 'gt_radar_points' in example:
                # Radar-only mode - check if we have any radar points
                if example is None or len(example['gt_radar_points']) == 0:
                    return None
            else:
                # No GT data at all
                if example is None:
                    return None
        
        return example
    
    def prepare_test_data(self, index):
        """Prepare test data.
        
        Args:
            index (int): Index for accessing the target data
            
        Returns:
            dict: Test data dict with radar_points
        """
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example
    
    def _build_default_pipeline(self):
        """Build default pipeline with radar loading."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        
        # Add radar loading
        if self.load_radar_data:
            pipeline.insert(2, dict(
                type='LoadRadarPointsXYZ',
                coord_type='LIDAR',
                load_dim=3,
                use_dim=[0, 1, 2]
            ))
        
        from .pipelines import Compose
        return Compose(pipeline)
    
    def evaluate(self,
                 results,
                 metric='radar_points',
                 logger=None,
                 **kwargs):
        """Evaluate radar point predictions.
        
        Args:
            results (list[dict]): Results containing 'radar_points' predictions
            metric (str): Evaluation metric
            logger: Logger for printing
            
        Returns:
            dict: Evaluation metrics
        """
        if metric == 'radar_points':
            return self.evaluate_radar_points(results, logger=logger)
        else:
            # Fall back to standard detection evaluation
            return super().evaluate(results, metric=metric, logger=logger, **kwargs)
    
    def evaluate_radar_points(self, results, logger=None):
        """Evaluate radar point cloud predictions.
        
        Args:
            results (list[dict]): Predictions with 'radar_points' key
            logger: Logger
            
        Returns:
            dict: Evaluation metrics (Chamfer distance, coverage, etc.)
        """
        import torch
        
        chamfer_distances = []
        coverage_scores = []
        
        for i, result in enumerate(results):
            if 'radar_points' not in result:
                continue
            
            pred_points = result['radar_points']  # (N_pred, 3)
            
            # Get GT radar points
            info = self.data_infos[i]
            if 'radar_points' in info and info['radar_points'] is not None:
                radar_data = info['radar_points']
                if isinstance(radar_data, dict) and 'xyz' in radar_data:
                    gt_points = radar_data['xyz']
                elif isinstance(radar_data, np.ndarray):
                    gt_points = radar_data[:, :3]
                else:
                    continue
            else:
                continue
            
            # Convert to tensors
            pred_points = torch.from_numpy(np.array(pred_points)).float()
            gt_points = torch.from_numpy(gt_points).float()
            
            # Compute Chamfer distance
            chamfer = self._compute_chamfer(pred_points, gt_points)
            chamfer_distances.append(chamfer.item())
            
            # Compute coverage (% of GT points within threshold of prediction)
            coverage = self._compute_coverage(pred_points, gt_points, threshold=1.0)
            coverage_scores.append(coverage)
        
        # Aggregate metrics
        metrics = {}
        if chamfer_distances:
            metrics['chamfer_distance'] = np.mean(chamfer_distances)
            metrics['coverage@1m'] = np.mean(coverage_scores)
        
        if logger is not None:
            logger.info(f'Radar Point Evaluation:')
            for k, v in metrics.items():
                logger.info(f'  {k}: {v:.4f}')
        
        return metrics
    
    def _compute_chamfer(self, pred, gt):
        """Compute Chamfer distance."""
        # pred: (N_pred, 3), gt: (N_gt, 3)
        pred = pred.unsqueeze(1)  # (N_pred, 1, 3)
        gt = gt.unsqueeze(0)      # (1, N_gt, 3)
        
        dist = torch.sum((pred - gt) ** 2, dim=-1)  # (N_pred, N_gt)
        
        min_dist_pred = torch.min(dist, dim=1)[0].mean()
        min_dist_gt = torch.min(dist, dim=0)[0].mean()
        
        return min_dist_pred + min_dist_gt
    
    def _compute_coverage(self, pred, gt, threshold=1.0):
        """Compute coverage score."""
        pred = pred.unsqueeze(1)
        gt = gt.unsqueeze(0)
        
        dist = torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1))
        min_dist_gt = torch.min(dist, dim=0)[0]
        
        coverage = (min_dist_gt < threshold).float().mean()
        return coverage.item()
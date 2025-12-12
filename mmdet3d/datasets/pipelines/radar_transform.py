# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom pipeline transforms for radar point cloud loading and processing.
"""

import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations


@PIPELINES.register_module()
class LoadRadarPointsXYZ:
    """Load radar points (XYZ coordinates only) from info dict.
    
    Args:
        coord_type (str): Coordinate system ('LIDAR', 'DEPTH', 'CAMERA')
        load_dim (int): Dimension of loaded points (usually 3 for XYZ)
        use_dim (list[int]): Dimensions to use (e.g., [0, 1, 2] for XYZ)
    """
    
    def __init__(self,
                 coord_type='LIDAR',
                 load_dim=3,
                 use_dim=[0, 1, 2]):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim if isinstance(use_dim, list) else list(use_dim)
    
    def __call__(self, results):
        """Load radar points from results dict.
        
        Args:
            results (dict): Result dict containing radar_info
            
        Returns:
            dict: Updated results with 'radar_points' key
        """
        if 'radar_info' not in results:
            # No radar data, create empty array
            results['radar_points'] = np.zeros((0, len(self.use_dim)), dtype=np.float32)
            return results
        
        radar_info = results['radar_info']
        
        # Get radar points from merged data
        if 'radar_points' in radar_info:
            radar_xyz = radar_info['radar_points']  # (N, 3)
        else:
            # Try to merge from individual sensors
            if 'radars' in radar_info and radar_info['radars']:
                radar_list = []
                for sensor_name, sensor_data in radar_info['radars'].items():
                    if len(sensor_data) > 0:
                        radar_list.append(sensor_data)
                
                if radar_list:
                    radar_xyz = np.concatenate(radar_list, axis=0)
                else:
                    radar_xyz = np.zeros((0, 3), dtype=np.float32)
            else:
                radar_xyz = np.zeros((0, 3), dtype=np.float32)
        
        # Select dimensions
        if radar_xyz.shape[0] > 0:
            radar_xyz = radar_xyz[:, self.use_dim]
        
        results['radar_points'] = radar_xyz.astype(np.float32)
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(coord_type={self.coord_type}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class RadarPointsRangeFilter:
    """Filter radar points within a specific range.
    
    Args:
        point_cloud_range (list): [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    
    def __init__(self, point_cloud_range):
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    
    def __call__(self, results):
        """Filter radar points.
        
        Args:
            results (dict): Result dict with 'radar_points'
            
        Returns:
            dict: Updated results with filtered radar_points
        """
        if 'radar_points' not in results:
            return results
        
        radar_points = results['radar_points']
        
        if len(radar_points) == 0:
            return results
        
        # Get point cloud range
        pc_min = self.point_cloud_range[:3]
        pc_max = self.point_cloud_range[3:6]
        
        # Filter points
        mask = np.all(
            (radar_points >= pc_min) & (radar_points <= pc_max),
            axis=1
        )
        
        results['radar_points'] = radar_points[mask]
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.point_cloud_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class RadarPointsSampler:
    """Sample a fixed number of radar points.
    
    If more points than num_points: sample randomly or use FPS
    If fewer points: pad with zeros or repeat points
    
    Args:
        num_points (int): Target number of points
        sample_method (str): 'random', 'fps' (farthest point sampling), or 'repeat'
        padding_value (float): Value to use for padding if not enough points
    """
    
    def __init__(self,
                 num_points=625,
                 sample_method='random',
                 padding_value=0.0):
        self.num_points = num_points
        self.sample_method = sample_method
        self.padding_value = padding_value
    
    def __call__(self, results):
        """Sample radar points to fixed size.
        
        Args:
            results (dict): Result dict with 'radar_points'
            
        Returns:
            dict: Updated results with sampled radar_points (num_points, 3)
        """
        if 'radar_points' not in results:
            # Create empty points
            results['radar_points'] = np.full(
                (self.num_points, 3), self.padding_value, dtype=np.float32
            )
            results['radar_points_mask'] = np.zeros(self.num_points, dtype=bool)
            return results
        
        radar_points = results['radar_points']
        N = len(radar_points)
        
        if N == 0:
            # No points, pad with padding value
            sampled_points = np.full(
                (self.num_points, 3), self.padding_value, dtype=np.float32
            )
            mask = np.zeros(self.num_points, dtype=bool)
        
        elif N >= self.num_points:
            # More points than needed, sample
            if self.sample_method == 'random':
                indices = np.random.choice(N, self.num_points, replace=False)
                sampled_points = radar_points[indices]
            elif self.sample_method == 'fps':
                indices = self._farthest_point_sample(radar_points, self.num_points)
                sampled_points = radar_points[indices]
            else:
                # Take first num_points
                sampled_points = radar_points[:self.num_points]
            
            mask = np.ones(self.num_points, dtype=bool)
        
        else:
            # Fewer points than needed, pad
            if self.sample_method == 'repeat':
                # Repeat points to reach num_points
                repeat_times = self.num_points // N + 1
                repeated = np.tile(radar_points, (repeat_times, 1))
                sampled_points = repeated[:self.num_points]
                mask = np.ones(self.num_points, dtype=bool)
            else:
                # Pad with padding value
                padding = np.full(
                    (self.num_points - N, 3), self.padding_value, dtype=np.float32
                )
                sampled_points = np.concatenate([radar_points, padding], axis=0)
                mask = np.concatenate([
                    np.ones(N, dtype=bool),
                    np.zeros(self.num_points - N, dtype=bool)
                ])
        
        results['radar_points'] = sampled_points.astype(np.float32)
        results['radar_points_mask'] = mask
        
        # Also create gt_radar_points for training
        # This will be picked up by Collect3D
        results['gt_radar_points'] = sampled_points.astype(np.float32)
        
        return results
    
    def _farthest_point_sample(self, points, num_samples):
        """Farthest point sampling.
        
        Args:
            points (np.ndarray): Points array (N, 3)
            num_samples (int): Number of samples
            
        Returns:
            np.ndarray: Sampled indices
        """
        N = len(points)
        indices = np.zeros(num_samples, dtype=np.int32)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        
        for i in range(num_samples):
            indices[i] = farthest
            centroid = points[farthest]
            dist = np.sum((points - centroid) ** 2, axis=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        
        return indices
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points}, '
        repr_str += f'sample_method={self.sample_method})'
        return repr_str


@PIPELINES.register_module()
class RadarPointsToTensor:
    """Convert radar points numpy array to tensor.
    
    This ensures gt_radar_points is in the correct format for the model.
    """
    
    def __call__(self, results):
        """Convert to tensor.
        
        Args:
            results (dict): Result dict
            
        Returns:
            dict: Updated results with tensor radar_points
        """
        if 'radar_points' in results:
            if not isinstance(results['radar_points'], torch.Tensor):
                results['radar_points'] = torch.from_numpy(
                    results['radar_points']
                ).float()
        
        if 'gt_radar_points' in results:
            if not isinstance(results['gt_radar_points'], torch.Tensor):
                results['gt_radar_points'] = torch.from_numpy(
                    results['gt_radar_points']
                ).float()
        
        if 'radar_points_mask' in results:
            if not isinstance(results['radar_points_mask'], torch.Tensor):
                results['radar_points_mask'] = torch.from_numpy(
                    results['radar_points_mask']
                ).bool()
        
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module()
class RadarPointsAugmentation:
    """Apply augmentation to radar points (rotation, scaling, flipping).
    
    Should be used after BEVAug to apply same transformations.
    
    Args:
        Apply transformations from BEV augmentation to radar points.
    """
    
    def __call__(self, results):
        """Apply augmentation.
        
        Args:
            results (dict): Result dict
            
        Returns:
            dict: Augmented results
        """
        if 'radar_points' not in results:
            return results
        
        # Get BEV augmentation params if available
        if 'pcd_rotation' in results:
            radar_points = results['radar_points']
            
            # Apply rotation
            rotation_mat = results['pcd_rotation']
            radar_points = radar_points @ rotation_mat.T
            
            # Apply scaling
            if 'pcd_scale_factor' in results:
                radar_points *= results['pcd_scale_factor']
            
            # Apply flipping
            if 'pcd_horizontal_flip' in results and results['pcd_horizontal_flip']:
                radar_points[:, 1] = -radar_points[:, 1]
            
            if 'pcd_vertical_flip' in results and results['pcd_vertical_flip']:
                radar_points[:, 0] = -radar_points[:, 0]
            
            results['radar_points'] = radar_points
        
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
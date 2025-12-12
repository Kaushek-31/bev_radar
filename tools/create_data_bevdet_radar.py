# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter_radar as nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def add_ann_adj_info(extra_tag):
    """Add annotation and adjacent frame information to the dataset.
    
    This function enriches the dataset with:
    - Sample annotations with velocities
    - Ground truth boxes and labels
    - Scene tokens
    - Occupancy paths
    
    Note: The original info dict already contains radar data (individual sensors
    and merged point cloud) from the nuscenes_converter.create_nuscenes_infos() call.
    This function adds additional annotation metadata.
    """
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            
            # Verify radar data is present (optional sanity check)
            if 'radars' not in info:
                print(f"Warning: No radar data found for sample {info.get('token', 'unknown')}")
            if 'radar_points' not in info:
                print(f"Warning: No merged radar_points found for sample {info.get('token', 'unknown')}")
            
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
        
        with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v1.0-trainval'
    # version = 'v1.0-test'
    # version = 'v1.0-mini'
    root_path = './data/nuscenes'
    extra_tag = 'bevdetv3-nuscenes'
    
    print('='*50)
    print('Step 1: Creating nuScenes infos with Camera, LiDAR, and Radar data')
    print('='*50)
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=version,
        max_sweeps=10)

    print('='*50)
    print('Step 2: Adding annotation and adjacent frame info')
    print('='*50)
    add_ann_adj_info(extra_tag)

    print('='*50)
    print('Step 3: Creating groundtruth database')
    print('='*50)
    create_groundtruth_database('NuScenesDataset',
                                root_path,
                                extra_tag,
                                f'{root_path}/{extra_tag}_infos_train.pkl')
    
    print('='*50)
    print('Data preparation complete!')
    print('Output files:')
    print(f'  - {root_path}/{extra_tag}_infos_train.pkl (with radar data)')
    print(f'  - {root_path}/{extra_tag}_infos_val.pkl (with radar data)')
    print('='*50)

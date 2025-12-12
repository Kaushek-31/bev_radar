# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union, Dict, Any

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDatasetRadar as NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

# Radar sensors to extract from nuScenes sample
RADAR_SENSORS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT'
]


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscenes dataset including radar returns.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc: NuScenes) -> List[dict]:
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (NuScenes): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc: NuScenes,
                         train_scenes: set,
                         val_scenes: set,
                         test: bool = False,
                         max_sweeps: int = 10) -> Tuple[List[dict], List[dict]]:
    """Generate the train/val infos from the raw data including radar.

    Returns:
        (train_infos, val_infos)
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info: Dict[str, Any] = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'radars': dict(),
            'radar_points': None,  # <-- MERGED RADAR CLOUD
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        # prepare rotation / translation matrices used for transforming sensors to lidar_top
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # ---- Extract radar data for the 5 radars and transform into lidar-top frame ----
        merged_xyz = []
        merged_vel = []
        merged_rcs = []
        merged_dyn = []
        merged_amb = []

        for radar_name in RADAR_SENSORS:
            try:
                radar_token = sample['data'].get(radar_name, None)
                if radar_token is None or radar_token == '':
                    # no radar for this sample, skip
                    info['radars'][radar_name] = {
                        'xyz': np.zeros((0, 3), dtype=np.float32),
                        'vel_xy': np.zeros((0, 2), dtype=np.float32),
                        'rcs': np.zeros((0,), dtype=np.float32),
                        'dyn_prop': np.zeros((0,), dtype=np.float32),
                        'ambig_state': np.zeros((0,), dtype=np.float32),
                        'timestamp': None,
                        'path': None,
                        'token': None
                    }
                else:
                    radar_info = _extract_radar_points(nusc, radar_token,
                                                       l2e_t, l2e_r_mat,
                                                       e2g_t, e2g_r_mat)
                    info['radars'][radar_name] = radar_info
                    
                    # Merge for the combined radar point cloud
                    merged_xyz.append(radar_info['xyz'])
                    merged_vel.append(radar_info['vel_xy'])
                    merged_rcs.append(radar_info['rcs'])
                    merged_dyn.append(radar_info['dyn_prop'])
                    merged_amb.append(radar_info['ambig_state'])
                    
            except Exception as e:
                # If something goes wrong with this radar sensor, log and store empty arrays
                mmcv.print_log(f'Warning: could not extract radar {radar_name} for sample {sample["token"]}: {e}', 'create_nuscenes_infos')
                info['radars'][radar_name] = {
                    'xyz': np.zeros((0, 3), dtype=np.float32),
                    'vel_xy': np.zeros((0, 2), dtype=np.float32),
                    'rcs': np.zeros((0,), dtype=np.float32),
                    'dyn_prop': np.zeros((0,), dtype=np.float32),
                    'ambig_state': np.zeros((0,), dtype=np.float32),
                    'timestamp': None,
                    'path': None,
                    'token': None
                }

        # ---- Final merged radar cloud ----
        if merged_xyz:
            info['radar_points'] = dict(
                xyz=np.concatenate(merged_xyz, axis=0),
                vel_xy=np.concatenate(merged_vel, axis=0),
                rcs=np.concatenate(merged_rcs, axis=0),
                dyn_prop=np.concatenate(merged_dyn, axis=0),
                ambig_state=np.concatenate(merged_amb, axis=0),
            )
        else:
            info['radar_points'] = dict(
                xyz=np.zeros((0, 3), dtype=np.float32),
                vel_xy=np.zeros((0, 2), dtype=np.float32),
                rcs=np.zeros((0,), dtype=np.float32),
                dyn_prop=np.zeros((0,), dtype=np.float32),
                ambig_state=np.zeros((0,), dtype=np.float32)
            )

        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # convert box size to the format of our lidar coordinate system
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        # append to train or val list
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def _load_radar_file(path: str) -> np.ndarray:
    """Load radar binary file; returns Nx18 float32 or NxK depending on file.

    Try common shapes. If fail, return empty Nx18.
    """
    if not mmcv.is_filepath(path):
        return np.zeros((0, 18), dtype=np.float32)
    try:
        pts = np.fromfile(path, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((0, 18), dtype=np.float32)
        # try reshape to common formats
        for k in [18, 9, 7, 4]:
            if pts.size % k == 0:
                return pts.reshape(-1, k).astype(np.float32)
        # fallback: return empty
        return np.zeros((0, 18), dtype=np.float32)
    except Exception:
        # fallback: try numpy load (if .npy)
        try:
            pts = np.load(path)
            return pts.astype(np.float32)
        except Exception:
            return np.zeros((0, 18), dtype=np.float32)


def _extract_radar_points(nusc: NuScenes,
                          radar_token: str,
                          lidar_l2e_t: np.ndarray,
                          lidar_l2e_r_mat: np.ndarray,
                          lidar_e2g_t: np.ndarray,
                          lidar_e2g_r_mat: np.ndarray) -> Dict[str, Any]:
    """Load and transform radar points into lidar-top frame.

    Args:
        nusc: NuScenes instance
        radar_token: sample_data token for the radar
        lidar_l2e_t, lidar_l2e_r_mat: lidar->ego translation/rotation (from this sample)
        lidar_e2g_t, lidar_e2g_r_mat: ego->global translation/rotation (from this sample)

    Returns:
        radar_dict with keys (xyz, vel_xy, rcs, dyn_prop, ambig_state, timestamp, path, token)
    """
    # get sample_data record and file path
    radar_path, _, _ = nusc.get_sample_data(radar_token)
    radar_sample = nusc.get('sample_data', radar_token)
    cs_record = nusc.get('calibrated_sensor', radar_sample['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', radar_sample['ego_pose_token'])

    # try load raw radar binary file
    raw = _load_radar_file(radar_path)  # shape (N, K)
    N = 0 if raw is None else raw.shape[0]
    if N == 0:
        return {
            'xyz': np.zeros((0, 3), dtype=np.float32),
            'vel_xy': np.zeros((0, 2), dtype=np.float32),
            'rcs': np.zeros((0,), dtype=np.float32),
            'dyn_prop': np.zeros((0,), dtype=np.float32),
            'ambig_state': np.zeros((0,), dtype=np.float32),
            'timestamp': radar_sample.get('timestamp', None),
            'path': radar_path,
            'token': radar_token
        }

    # interpret raw fields robustly
    pts = raw
    # handle common case Nx18
    if pts.shape[1] >= 8:
        xyz = pts[:, :3].astype(np.float32)
        dyn_prop = pts[:, 3].astype(np.float32)
        ambig_state = pts[:, 4].astype(np.float32)
        vel_x = pts[:, 5].astype(np.float32)
        vel_y = pts[:, 6].astype(np.float32)
        rcs = pts[:, 7].astype(np.float32)
    else:
        # fallback mapping: if shape is Nx4 assume [x,y,z,intensity]
        if pts.shape[1] == 4:
            xyz = pts[:, :3].astype(np.float32)
            rcs = pts[:, 3].astype(np.float32)
            dyn_prop = np.zeros((pts.shape[0],), dtype=np.float32)
            ambig_state = np.zeros((pts.shape[0],), dtype=np.float32)
            vel_x = np.zeros((pts.shape[0],), dtype=np.float32)
            vel_y = np.zeros((pts.shape[0],), dtype=np.float32)
        else:
            # unknown layout: attempt to take first 3 columns as xyz
            xyz = pts[:, :3].astype(np.float32)
            rcs = np.zeros((pts.shape[0],), dtype=np.float32)
            dyn_prop = np.zeros((pts.shape[0],), dtype=np.float32)
            ambig_state = np.zeros((pts.shape[0],), dtype=np.float32)
            vel_x = np.zeros((pts.shape[0],), dtype=np.float32)
            vel_y = np.zeros((pts.shape[0],), dtype=np.float32)

    # Transform points: sensor -> ego -> global -> ego_at_lidar_time -> lidar_top
    # 1) sensor -> ego (use calibrated sensor record)
    r2e_t = np.array(cs_record['translation'])
    r2e_r = Quaternion(cs_record['rotation']).rotation_matrix
    pts_ego = xyz @ r2e_r.T + r2e_t  # (N, 3)

    # 2) ego -> global (use radar sample ego pose)
    e2g_t_s = np.array(pose_record['translation'])
    e2g_r_s = Quaternion(pose_record['rotation']).rotation_matrix
    pts_global = pts_ego @ e2g_r_s.T + e2g_t_s

    # 3) global -> ego (at lidar time): invert lidar ego->global
    pts_ego_lidar = (pts_global - np.array(lidar_e2g_t)) @ np.linalg.inv(lidar_e2g_r_mat).T

    # 4) ego_lidar -> lidar_top (invert lidar sensor2ego)
    pts_lidar_top = (pts_ego_lidar - np.array(lidar_l2e_t)) @ np.linalg.inv(lidar_l2e_r_mat).T

    # assemble metadata arrays
    vel_xy = np.stack([vel_x, vel_y], axis=1).astype(np.float32)
    radar_dict = {
        'xyz': pts_lidar_top.astype(np.float32),
        'vel_xy': vel_xy,
        'rcs': rcs.astype(np.float32),
        'dyn_prop': dyn_prop.astype(np.float32),
        'ambig_state': ambig_state.astype(np.float32),
        'timestamp': radar_sample.get('timestamp', None),
        'path': radar_path,
        'token': radar_token
    }
    return radar_dict


def obtain_sensor2top(nusc: NuScenes,
                      sensor_token: str,
                      l2e_t: np.ndarray,
                      l2e_r_mat: np.ndarray,
                      e2g_t: np.ndarray,
                      e2g_r_mat: np.ndarray,
                      sensor_type: str = 'lidar') -> Dict[str, Any]:
    """Obtain the info with RT matrix from general sensor to Top LiDAR.

    Returns a sweep dict with transformation so that:
      points @ sweep['sensor2lidar_rotation'].T + sweep['sensor2lidar_translation']
    will map points in sensor frame to lidar_top frame.

    Args:
        nusc: NuScenes instance
        sensor_token: sample_data token for this sensor
        l2e_t, l2e_r_mat: lidar->ego translation/rotation (from keyframe)
        e2g_t, e2g_r_mat: ego->global translation/rotation (from keyframe)
        sensor_type: 'lidar' or other
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Path of the data root.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc: NuScenes,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        nusc (NuScenes): NuScenes instance.
        sample_data_token (str): Sample data token.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get mono3d annotation.

    Returns:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        final_coords = post_process_coords(corner_coords)

        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
from typing import Union

import numpy as np
import torch

from . import BodyModel

betas_params_names = ['betas', 'shape']
pose_params_names = ['poses', 'fullpose', 'pose', 'body_pose']
global_ori_params_names = ['global_ori', 'global_orient', 'root_orient']
trans_params_names = ['trans', 'transl']


def find_param_key(params_dict, param_names):
    res = None
    for key in param_names:
        res = params_dict.get(key, None)
        if res is not None:
            break
    return res


def get_model_params(body_model: BodyModel, params: dict):
    betas = find_param_key(params, betas_params_names)

    if betas is None:
        raise ValueError('No betas found in params dict with names: ' + str(betas_params_names))

    pose = find_param_key(params, pose_params_names)

    if pose is None:
        raise ValueError('No pose found in params dict with names: ' + str(pose_params_names))

    if pose.shape[-1] != body_model.num_pose_params:
        if pose.shape[-1] == body_model.num_pose_params - 3:
            global_ori = find_param_key(params, global_ori_params_names)
            if global_ori is None:
                raise ValueError('No global_ori found in params dict with names: ' + str(global_ori_params_names))
            pose = torch.cat([global_ori, pose], dim=-1)
        else:
            raise ValueError('pose has shape ' + str(pose.shape))

    trans = find_param_key(params, trans_params_names)

    if trans is None:
        raise ValueError('No trans found in params dict with names: ' + str(trans))

    return betas, pose, trans


def rotate_points_around_axis(v: np.ndarray, deg: float, axis: Union[int, str]):
    """
    Takes a set of points and rotates them by the given angle around the given axis.

    Args:
      v: the points to rotate
      deg: the angle to rotate by
      axis: the axis to rotate around. Can be 0 (=x), 1 (=y) or 2 (=z)

    Returns:
      the rotated points.
    """
    rot = np.radians(deg)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis = axis_map[axis.lower()]

    if axis == 0:
        rot_mat = np.array([[1, 0, 0],
                            [0, np.cos(rot), -np.sin(rot)],
                            [0, np.sin(rot), np.cos(rot)]])
    elif axis == 1:
        rot_mat = np.array([[np.cos(rot), 0, np.sin(rot)],
                            [0, 1, 0],
                            [-np.sin(rot), 0, np.cos(rot)]])
    elif axis == 2:
        rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1]])
    else:
        raise ValueError("axis must be 0 (=x), 1 (=y) or 2 (=z)")

    if len(v.shape) == 2:
        return v @ rot_mat.T
    elif len(v.shape) == 3:
        return np.einsum('nvj, ij -> nvi', v, rot_mat)
    else:
        raise ValueError("v must be 2 or 3 dimensional")

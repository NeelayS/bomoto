from typing import Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
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
                            [0, np.sin(rot), np.cos(rot)]], dtype=np.float32)
    elif axis == 1:
        rot_mat = np.array([[np.cos(rot), 0, np.sin(rot)],
                            [0, 1, 0],
                            [-np.sin(rot), 0, np.cos(rot)]], dtype=np.float32)
    elif axis == 2:
        rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1]], dtype=np.float32)
    else:
        raise ValueError("axis must be 0 (=x), 1 (=y) or 2 (=z)")

    if len(v.shape) == 1:
        return (v[None, ...] @ rot_mat.T).squeeze()
    if len(v.shape) == 2:
        return v @ rot_mat.T
    elif len(v.shape) == 3:
        return np.einsum('nvj, ij -> nvi', v, rot_mat)
    else:
        raise ValueError("v must be 2 or 3 dimensional")


def rotvec_slerp(initial, target, n_frames_interp=None, interp_coeffs=None):
    if interp_coeffs is None:
        interp_coeffs = np.linspace(0, 1, n_frames_interp, dtype=np.float32)[..., None]
    initial, target = initial.reshape(-1, 3), target.reshape(-1, 3)
    joint_rots = [np.stack([r_initial, r_target]) for r_initial, r_target in zip(initial, target)]
    joint_rots = [Rotation.from_rotvec(jr) for jr in joint_rots]
    slerps = [Slerp([0, 1], jr)(interp_coeffs.squeeze()).as_rotvec() for jr in joint_rots]
    return np.concatenate(slerps, axis=1)


def lerp(initial, target, n_frames_interp=None, interp_coeffs=None):
    if interp_coeffs is None:
        interp_coeffs = np.linspace(0, 1, n_frames_interp, dtype=np.float32)[..., None]
    interp_vector = target - initial
    transition = initial + interp_coeffs * interp_vector[None, ...]
    return transition


def interpolate_parameters(target_betas: np.ndarray,
                           target_pose: np.ndarray,
                           target_trans: np.ndarray,
                           initial_betas: np.ndarray = None,
                           initial_pose: np.ndarray = None,
                           initial_trans: np.ndarray = None,
                           n_frames_interp: int = 100):
    """
    Interpolates SMPL-X parameters using linear interpolation for betas and translation and spherical linear
    interpolation for pose.

    Args:
        target_betas:
        target_pose:
        target_trans:
        initial_betas:
        initial_pose:
        initial_trans:
        n_frames_interp:

    Returns:
        interpolated SMPL-X parameters

    """
    # interp_coeffs = (np.arange(n_frames_interp).astype(np.float32)[..., None] / n_frames_interp)
    interp_coeffs = np.linspace(0, 1, n_frames_interp, dtype=np.float32)[..., None]

    initial_betas = target_betas if initial_betas is None else initial_betas
    initial_pose = target_pose if initial_pose is None else initial_pose
    initial_trans = target_trans if initial_trans is None else initial_trans

    transition_poses = rotvec_slerp(initial_pose, target_pose, n_frames_interp=None, interp_coeffs=interp_coeffs)

    transition_betas = lerp(initial_betas, target_betas, n_frames_interp=None, interp_coeffs=interp_coeffs)
    transition_trans = lerp(initial_trans, target_trans, n_frames_interp=None, interp_coeffs=interp_coeffs)

    return transition_betas, transition_poses, transition_trans

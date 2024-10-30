import os
import os.path as osp

import numpy as np
from bomoto.body_models import BodyModel
from bomoto.body_models import interpolate_parameters


def generate_smpl_params_sequence():
    """
    Generates a simple example sequence where, in 100 frames, the body lowers the arms and moves slightly forward
    """
    betas = np.random.rand(10) * 2 - 0.5
    initial_pose = np.zeros((72,), dtype=np.float32)
    initial_trans = np.zeros((3,), dtype=np.float32)

    target_pose = initial_pose.copy()
    target_pose[[50, 53]] = [-np.pi / 4, np.pi / 4]  # this rotates the arms to make an A-Pose
    target_trans = np.array([0, 0, 1], dtype=np.float32)

    betas, poses, trans = interpolate_parameters(target_betas=betas,
                                                 target_pose=target_pose,
                                                 target_trans=target_trans,
                                                 initial_betas=betas,
                                                 initial_pose=initial_pose,
                                                 initial_trans=initial_trans,
                                                 n_frames_interp=100)
    return betas, poses, trans


def main():
    betas, poses, trans = generate_smpl_params_sequence()

    out_data = {
        'betas': betas,
        'poses': poses,
        'trans': trans
    }

    sample_motion_seq_fname = osp.join(osp.dirname(__file__), 'sample_data', 'sample_motion_seq.npz')
    os.makedirs(osp.dirname(sample_motion_seq_fname), exist_ok=True)
    np.savez_compressed(sample_motion_seq_fname, **out_data)

    print(f'Sample motion sequence saved to {sample_motion_seq_fname}')


if __name__ == '__main__':
    main()

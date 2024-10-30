import os
import os.path as osp

import numpy as np
import torch.cuda

from bomoto.body_models import BodyModel
from bomoto.body_models import interpolate_parameters

import trimesh

import argparse


def generate_smpl_params_sequence():
    """
    Generates a simple example sequence where, in 100 frames, the body lowers the arms and moves slightly forward
    """
    betas = np.random.rand(300) * 2 - 0.5
    initial_pose = np.zeros((165,), dtype=np.float32)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    assert osp.exists(args.model_path), f'Model file not found: {args.model_path}'

    out_path = osp.join(osp.dirname(__file__), 'sample_data')
    betas, poses, trans = generate_smpl_params_sequence()

    bmargs = {
        'model_path': args.model_path,
        'gender': 'neutral',
        'n_betas': 300,
        'batch_size': poses.shape[0],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    body_model = BodyModel.instantiate('smplx', **bmargs)

    v_seq = body_model.forward(betas, poses, trans).detach().cpu().numpy()

    for i, v in enumerate(v_seq):
        mesh = trimesh.Trimesh(vertices=v, faces=body_model.faces.cpu().numpy())
        out_fname = osp.join(out_path, f'frame_{str(i).zfill(5)}.obj')
        os.makedirs(osp.dirname(out_fname), exist_ok=True)
        mesh.export(out_fname)

    print(f'Sample meshes saved to {out_path}')


if __name__ == '__main__':
    main()

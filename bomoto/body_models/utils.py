import torch

from . import BodyModel


def fix_params_keys(body_model: BodyModel, params: dict):
    betas = params.get("betas", None)

    for key in ['poses', 'fullpose', 'pose', 'body_pose']:
        pose = params.get(key, None)
        if pose is not None:
            break

    if pose.shape[-1] != body_model.num_pose_params:
        if pose.shape[-1] == body_model.num_pose_params - 3:
            for key in ['global_ori', 'global_orient', 'root_orient']:
                global_ori = params.get(key, None)
                if global_ori is not None:
                    break
            if global_ori is None:
                raise ValueError
            pose = torch.cat([global_ori, pose], dim=-1)
        else:
            raise ValueError

    for key in ['trans', 'transl']:
        trans = params.get(key, None)
        if trans is not None:
            break

    return betas, pose, trans

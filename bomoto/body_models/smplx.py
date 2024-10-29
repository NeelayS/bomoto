from __future__ import annotations

from typing import Union

import numpy as np
import torch
from smplx import SMPLX

from . import BodyModel


class SMPLXWrapper(BodyModel):
    def __init__(self, model_path: str,
                 gender: str,
                 n_betas: int,
                 batch_size: int,
                 device: Union[str, torch.device],
                 flat_hand_mean: bool = True,
                 # use_pca: bool = False,
                 # num_pca_comps: int = 45,
                 v_template=None,
                 **misc_args):
        body_model = SMPLX(model_path=model_path,
                           gender=gender,
                           num_betas=n_betas,
                           batch_size=batch_size,
                           device=device,
                           flat_hand_mean=flat_hand_mean,
                           use_pca=False,
                           num_pca_comps=45,
                           v_template=v_template,
                           **misc_args)
        super().__init__(model=body_model)

    @property
    def num_vertices(self) -> int:
        return 10475

    @property
    def num_pose_params(self) -> int:
        # return 168 if self.body_models.extra_root_joint_at_origin else 165
        return 165

    @staticmethod
    def get_body_model_params_info():
        return {"pose": 162, "global_orient": 3, "transl": 3}

    @staticmethod
    def full_pose_to_parts(full_pose: Union[torch.tensor, np.ndarray], hand_pca_comps: int = 45):
        return {
            "global_orient": full_pose[:, :3],
            "body_pose": full_pose[:, 3:66],
            "jaw_pose": full_pose[:, 66:69],
            "leye_pose": full_pose[:, 69:72],
            "reye_pose": full_pose[:, 72:75],
            "left_hand_pose": full_pose[:, 75: 75 + hand_pca_comps],
            "right_hand_pose": full_pose[:, 75 + hand_pca_comps:]
        }

    def forward(self,
                betas: Union[torch.tensor, np.ndarray, None] = None,
                pose: Union[torch.tensor, np.ndarray, None] = None,
                trans: Union[torch.tensor, np.ndarray, None] = None,
                return_full_model_output: bool = False,
                **kwargs):
        betas, pose, trans, kwargs = super()._preprocess_params(betas, pose, trans, **kwargs)
        output = self.model.forward(betas=betas,
                                    transl=trans,
                                    **SMPLXWrapper.full_pose_to_parts(pose),
                                    **kwargs)
        return output.vertices if not return_full_model_output else output

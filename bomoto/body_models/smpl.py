from __future__ import annotations

from typing import Union

import numpy as np
import torch
from smplx import SMPL

from . import BodyModel


class SMPLWrapper(BodyModel):
    def __init__(self, model_path: str,
                 gender: str,
                 n_betas: int,
                 batch_size: int,
                 device: Union[str, torch.device],
                 v_template=None,
                 **misc_args):
        body_model = SMPL(model_path=model_path,
                          gender=gender,
                          num_betas=n_betas,
                          batch_size=batch_size,
                          device=device,
                          v_template=v_template,
                          **misc_args)
        super().__init__(model=body_model)

    @property
    def num_vertices(self) -> int:
        return 6890

    @property
    def num_pose_params(self) -> int:
        return 72

    @staticmethod
    def get_body_model_params_info():
        return {"body_pose": 69, "global_orient": 3, "transl": 3}

    @staticmethod
    def full_pose_to_parts(full_pose: Union[torch.tensor, np.ndarray]):
        return {
            "global_orient": full_pose[:, :3],
            "body_pose": full_pose[:, 3:],
        }

    def forward(self,
                betas: Union[torch.tensor, np.ndarray, None] = None,
                pose: Union[torch.tensor, np.ndarray, None] = None,
                trans: Union[torch.tensor, np.ndarray, None] = None,
                **kwargs):
        betas, pose, trans, kwargs = super()._preprocess_params(betas, pose, trans, **kwargs)
        return self.model.forward(betas=betas,
                                  transl=trans,
                                  **SMPLWrapper.full_pose_to_parts(pose)).vertices

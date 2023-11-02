from __future__ import annotations

from typing import Union, Optional

import torch
from smplx import SMPL

from . import BodyModel


class SMPLWrapper(BodyModel):
    def __init__(self, model_path: str,
                 gender: str,
                 n_betas: int,
                 batch_size: int,
                 device: Union[str, torch.device],
                 **misc_args):
        body_model = SMPL(model_path=model_path,
                          gender=gender,
                          num_betas=n_betas,
                          batch_size=batch_size,
                          device=device,
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
    def full_pose_to_parts(full_pose: torch.Tensor):
        return {
            "global_orient": full_pose[:, :3],
            "body_pose": full_pose[:, 3:],
        }

    def forward(self,
                betas: Optional[torch.tensor] = None,
                pose: Optional[torch.tensor] = None,
                trans: Optional[torch.tensor] = None):
        betas, pose, trans = super()._replace_none_params(betas, pose, trans)
        return self.model.forward(betas=betas,
                                  transl=trans,
                                  **SMPLWrapper.full_pose_to_parts(pose)).vertices

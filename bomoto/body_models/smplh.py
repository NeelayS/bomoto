from __future__ import annotations

from functools import cached_property
from typing import Union, Optional

import torch
from smplx import SMPLH

from . import BodyModel


class SMPLHWrapper(BodyModel):
    def __init__(self, model_path: str,
                 gender: str,
                 n_betas: int,
                 batch_size: int,
                 device: Union[str, torch.device],
                 flat_hand_mean: bool = True,
                 # use_pca: bool = False,
                 # num_pca_comps: int = 45,
                 **misc_args):
        body_model = SMPLH(model_path=model_path,
                           gender=gender,
                           num_betas=n_betas,
                           batch_size=batch_size,
                           use_pca=False,
                           num_pca_comps=45,
                           flat_hand_mean=flat_hand_mean,
                           device=device,
                           **misc_args)
        super().__init__(model=body_model)

    @property
    def num_vertices(self) -> int:
        return 6890

    @cached_property
    def num_pose_params(self) -> int:
        return 156

    @staticmethod
    def get_body_model_params_info():
        return {"pose": 153, "global_orient": 3, "transl": 3}

    @staticmethod
    def full_pose_to_parts(full_pose: torch.Tensor, hand_pca_comps: int = 45):
        # TODO: add possibility of using PCA hand components
        return {
            "global_orient": full_pose[:, :3],
            "body_pose": full_pose[:, 3:66],
            "left_hand_pose": full_pose[:, 66: 66 + hand_pca_comps],
            "right_hand_pose": full_pose[:, 66 + hand_pca_comps:]
        }

    def forward(self,
                betas: Optional[torch.tensor] = None,
                pose: Optional[torch.tensor] = None,
                trans: Optional[torch.tensor] = None):
        betas, pose, trans = super()._replace_none_params(betas, pose, trans)
        return self.model.forward(betas=betas,
                                  transl=trans,
                                  **SMPLHWrapper.full_pose_to_parts(pose)).vertices
from __future__ import annotations

from functools import cached_property
from typing import Union

import numpy as np
import torch

from skel.skel_model import SKEL
from . import BodyModel


class SKELWrapper(BodyModel):
    def __init__(self, model_path: str,
                 gender: str,
                 device: Union[str, torch.device],
                 **misc_args):
        body_model = SKEL(model_path=model_path,
                          gender=gender,
                          **misc_args).to(device)

        super().__init__(model=body_model)

    @cached_property
    def faces(self) -> torch.tensor:
        return self.model.skin_f

    @property
    def num_vertices(self) -> int:
        return 6890

    @property
    def num_betas(self) -> int:
        return 10

    @property
    def num_pose_params(self) -> int:
        return 46

    @staticmethod
    def get_body_model_params_info():
        return {"pose": 43, "global_orient": 3, "transl": 3}

    @property
    def batch_size(self):
        return 1

    def forward(self,
                betas: Union[torch.tensor, np.ndarray, None] = None,
                pose: Union[torch.tensor, np.ndarray, None] = None,
                trans: Union[torch.tensor, np.ndarray, None] = None,
                return_full_model_output: bool = False,
                **kwargs):
        betas, pose, trans, kwargs = super()._preprocess_params(betas, pose, trans, **kwargs)
        betas = betas.repeat(pose.shape[0], 1) if betas.shape[0] != pose.shape[0] else betas
        output = self.model.forward(betas=betas,
                                    trans=trans,
                                    poses=pose,
                                    **kwargs)
        return output.skin_verts if not return_full_model_output else output

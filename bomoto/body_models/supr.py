from __future__ import annotations

from functools import cached_property
from typing import Union

import numpy as np
import torch

from SUPR.supr.pytorch.supr import SUPR
from . import BodyModel


class SUPRWrapper(BodyModel):
    def __init__(self, model_path: str,
                 n_betas: int,
                 device: Union[str, torch.device],
                 v_template=None,
                 **misc_args):
        body_model = SUPR(path_model=model_path,
                          num_betas=n_betas,
                          v_template=v_template,
                          device=device)
        super().__init__(model=body_model)

    @cached_property
    def faces(self):
        return self.model.faces.type(torch.long).to(self.device)

    @property
    def num_vertices(self) -> int:
        return 10475

    @property
    def num_pose_params(self) -> int:
        return 225

    @staticmethod
    def get_body_model_params_info():
        return {"pose": 225, "trans": 3}

    @property
    def batch_size(self) -> int:
        return 1

    def forward(self,
                betas: Union[torch.tensor, np.ndarray, None] = None,
                pose: Union[torch.tensor, np.ndarray, None] = None,
                trans: Union[torch.tensor, np.ndarray, None] = None):
        betas, pose, trans = super()._preprocess_params(betas, pose, trans)
        return self.model.forward(betas=betas,
                                  pose=pose,
                                  trans=trans)["vertices"]
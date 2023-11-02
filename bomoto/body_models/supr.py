from __future__ import annotations

from functools import cached_property
from typing import Union, Optional

import torch

from SUPR.supr.pytorch.supr import SUPR
from . import BodyModel


class SUPRWrapper(BodyModel):
    def __init__(self, model_path: str,
                 n_betas: int,
                 device: Union[str, torch.device],
                 **misc_args):
        body_model = SUPR(path_model=model_path,
                          num_betas=n_betas,
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
                betas: Optional[torch.tensor] = None,
                pose: Optional[torch.tensor] = None,
                trans: Optional[torch.tensor] = None):
        betas, pose, trans = super()._replace_none_params(betas, pose, trans)
        return self.model.forward(betas=betas,
                                  pose=pose,
                                  trans=trans)["vertices"]

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import torch
from functools import cached_property


class BodyModel(ABC):
    body_models = None

    def __init__(self, model):
        self.model: torch.nn.Module = model
        # self.faces: torch.tensor = None

    @cached_property
    def faces(self):
        # the SUPRWrapper class overrides this method
        return torch.as_tensor(self.model.faces.astype(np.int64), dtype=torch.long, device=self.device)

    @property
    @abstractmethod
    def num_vertices(self):
        return NotImplementedError()

    @property
    @abstractmethod
    def num_pose_params(self):
        return NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_body_model_params_info():
        return NotImplementedError()

    @property
    def device(self):
        return self.model.shapedirs.device

    @property
    def batch_size(self) -> int:
        return self.model.batch_size

    def set_v_template(self, v_template: torch.tensor):
        # TODO: modify SUPR code to allow setting a v_template
        raise NotImplementedError()

    def _replace_none_params(self,
                             betas: Optional[torch.tensor] = None,
                             pose: Optional[torch.tensor] = None,
                             trans: Optional[torch.tensor] = None):

        if betas is None:
            betas = torch.zeros((self.batch_size, self.model.num_betas), device=self.device)
        else:
            # if betas.shape[-1] != self.model.num_betas:
            #     raise ValueError(f"betas must have shape (batch_size, {self.model.num_betas}), got {betas.shape}")
            betas = betas[..., :self.model.num_betas]
            if betas.ndim == 1:
                betas = betas[None, ...].repeat(self.batch_size, 1)
        if pose is None:
            pose = torch.zeros((self.batch_size, self.num_pose_params), device=self.device)
        if trans is None:
            trans = torch.zeros((self.batch_size, 3), device=self.device)
        return betas, pose, trans

    @abstractmethod
    def forward(self,
                betas: Optional[torch.tensor] = None,
                pose: Optional[torch.tensor] = None,
                trans: Optional[torch.tensor] = None):
        raise NotImplementedError()

    def to(self, device: Union[str, torch.device]) -> BodyModel:
        self.model = self.model.to(device)
        return self

    def eval(self) -> BodyModel:
        self.model = self.model.eval()
        return self

    @staticmethod
    def instantiate(model_type: str, **kwargs) -> BodyModel:
        if not isinstance(model_type, str):
            raise ValueError(f"model_type must be a string, got {type(model_type)}")
        model_type = model_type.lower()
        model_type = ''.join([c for c in model_type.lower() if c.isalnum()])
        return BodyModel.body_models[model_type](**kwargs).to(kwargs['device'])

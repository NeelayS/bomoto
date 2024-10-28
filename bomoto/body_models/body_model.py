from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple

import numpy as np
import torch
from functools import cached_property
from ..utils import params2torch


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
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_pose_params(self):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_body_model_params_info():
        raise NotImplementedError()

    @property
    def device(self):
        return self.model.shapedirs.device

    @property
    def batch_size(self) -> int:
        return self.model.batch_size

    # def set_v_template(self, v_template: torch.tensor):
    #     # TODO: modify SUPR code to allow setting a v_template
    #     raise NotImplementedError()

    def _preprocess_params(self,
                           betas: Optional[torch.tensor] = None,
                           pose: Optional[torch.tensor] = None,
                           trans: Optional[torch.tensor] = None,
                           **kwargs) -> Tuple[torch.tensor, torch.tensor, torch.tensor, dict]:

        if betas is None:
            betas = torch.zeros((self.batch_size, self.model.num_betas), dtype=torch.float32, device=self.device)
        else:
            betas = betas[..., :self.model.num_betas]
            betas = torch.as_tensor(betas, dtype=torch.float32, device=self.device).squeeze()
            if betas.ndim == 1:
                betas = betas[None, ...].repeat(self.batch_size, 1)
            # elif betas.shape[0] == 1:
            #     betas = betas[None, ...].repeat(self.batch_size, 1)

        if pose is None:
            pose = torch.zeros((self.batch_size, self.num_pose_params), dtype=torch.float32, device=self.device)
        else:
            pose = torch.as_tensor(pose, dtype=torch.float32, device=self.device)
            if pose.ndim == 1:
                pose = pose[None, ...].repeat(self.batch_size, 1)

        if trans is None:
            trans = torch.zeros((self.batch_size, 3), dtype=torch.float32, device=self.device)
        else:
            trans = torch.as_tensor(trans, dtype=torch.float32, device=self.device)
            if trans.ndim == 1:
                trans = trans[None, ...].repeat(self.batch_size, 1)

        return betas, pose, trans, params2torch(kwargs, device=self.device)

    @abstractmethod
    def forward(self,
                betas: Union[torch.tensor, np.ndarray, None] = None,
                pose: Union[torch.tensor, np.ndarray, None] = None,
                trans: Union[torch.tensor, np.ndarray, None] = None,
                **kwargs):
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

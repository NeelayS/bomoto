import os

import numpy as np
import torch
import trimesh
from .body_models import BodyModel, get_model_params

from . import numpy2torch_types_to_convert, numpy_float_types


class MeshDirDataset(torch.utils.data.Dataset):
    """
    Dataset class to load meshes stored in either .obj or .ply format from a directory.

    Parameters
    ----------
    mesh_dir : str
        Path to the directory containing the meshes.
    mesh_format: str, optional
        Format of the meshes. Can be either 'obj' or 'ply'. By default, loads meshes present in either format.
    """

    def __init__(self, mesh_dir: str, mesh_format: str = None):
        super().__init__()

        assert isinstance(
            mesh_dir, str
        ), "mesh_dir must be a string denoting the path to the directory containing the meshes"
        assert os.path.isdir(
            mesh_dir
        ), f"mesh_dir must be a valid directory. {mesh_dir} is not a valid directory"

        if mesh_format is None:
            mesh_format = (".obj", ".ply")
        else:
            assert isinstance(
                mesh_format, str
            ), "if specified, mesh_format must be a string denoting the format of the meshes"
            mesh_format = mesh_format.lower()
            assert mesh_format in [
                "obj",
                "ply",
            ], "mesh_format must be either 'obj' or 'ply'"
            mesh_format = (f".{mesh_format}",)

        self.mesh_paths = sorted(
            [
                os.path.join(mesh_dir, f)
                for f in os.listdir(mesh_dir)
                if any(f.endswith(ext) for ext in mesh_format)
            ]
        )

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):

        mesh_path = self.mesh_paths[idx]
        mesh = trimesh.load(mesh_path, process=False)

        return {
            "vertices": torch.tensor(mesh.vertices, dtype=torch.float32),
            "faces": torch.tensor(
                mesh.faces,
            ),
        }


class NPZParamsFileDataset(torch.utils.data.Dataset):
    """
    Dataset class used to load parameters stored in .npz files and
    generate meshes using a body model.

    NOTE: When using this dataset class, the batch size in the dataloader
    must be set to 1 and the batch size for the body model must be equal
    to the number of frames/bodies in each .npz file.

    Parameters
    ----------
    body_model : BodyModel
        A body model wrapper instance.
    npz_files_dir : str
        Path to the directory containing the .npz files.
    n_betas : int
        Number of shape parameters.
    device : torch.device, optional
        Device on which to perform the forward pass. By default, uses the CPU.
    """

    def __init__(
            self,
            body_model: BodyModel,
            body_model_batch_size: int,
            npz_files_dir: str,
            n_betas: int,
            betas_override: np.ndarray = None,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        assert isinstance(npz_files_dir, str), "npz_files_dir must be a string"
        assert os.path.isdir(npz_files_dir), f"{npz_files_dir} is not a valid directory"

        self.npz_file_paths = sorted(
            [
                os.path.join(npz_files_dir, f)
                for f in os.listdir(npz_files_dir)
                if f.endswith(".npz")
            ]
        )

        self.body_model = body_model.to(device)
        # self.body_model_type = check_body_model_type(body_model_type)
        self.body_model_batch_size = body_model_batch_size
        self.body_model_faces = self.body_model.faces
        if not isinstance(self.body_model_faces, torch.Tensor):
            self.body_model_faces = torch.tensor(self.body_model_faces.astype(np.int64))
        self.body_model_faces = self.body_model_faces.type(torch.long).to(device)
        self.n_betas = n_betas
        self.betas_override = betas_override
        self.device = device

    def __len__(self):
        return len(self.npz_file_paths)

    def __getitem__(self, idx):

        npz_file_path = self.npz_file_paths[idx]
        params = dict(np.load(npz_file_path, allow_pickle=True))

        params = {k: v for k, v in params.items() if v.dtype in numpy2torch_types_to_convert}

        for key in params.keys():
            v = params[key]
            if v.dtype in numpy_float_types:
                v = v.astype(np.float32)
            params[key] = torch.tensor(v).to(self.device)

        betas, pose, trans = get_model_params(self.body_model, params)

        if self.betas_override is not None:
            betas = torch.as_tensor(self.betas_override, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            vertices = self.body_model.forward(betas=betas, pose=pose, trans=trans)

        return {
            "vertices": vertices,
            "faces": self.body_model_faces,
        }


def get_dataset(input_data_type: str, dataloader_batch_size: int):
    assert isinstance(input_data_type, str), "input_data_type must be a string"
    input_data_type = input_data_type.lower()
    assert input_data_type in [
        "meshes",
        "params",
    ], "input_data_type must be either 'meshes' or 'params'"

    assert (
            type(dataloader_batch_size) == int
    ), "dataloader_batch_size must be an integer"

    if input_data_type == "meshes":
        dataset = MeshDirDataset
    elif input_data_type == "params":
        dataset = NPZParamsFileDataset
        dataloader_batch_size = 1

    return {
        "input_data_type": input_data_type,
        "dataset": dataset,
        "dataloader_batch_size": dataloader_batch_size,
    }

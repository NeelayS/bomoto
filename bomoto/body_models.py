import os

import torch
from smplx import SMPL, SMPLH, SMPLX

from SUPR.supr.pytorch.supr import SUPR

BODY_MODELS = ("smpl", "smplh", "smplx", "supr")


def check_body_model_type(body_model_type: str):

    assert isinstance(body_model_type, str), "body_model_type must be a string"

    body_model_type = body_model_type.lower()

    if body_model_type not in BODY_MODELS:
        raise ValueError(f"Unknown body model type: {body_model_type}")

    return body_model_type


def get_body_model(
    body_model_type: str,
):

    body_models = {"smpl": SMPL, "smplh": SMPLH, "smplx": SMPLX, "supr": SUPR}

    body_model_type = check_body_model_type(body_model_type)

    return body_models[body_model_type]


def instantiate_body_model(
    body_model_type: str,
    body_model_class: torch.nn.Module,
    body_model_path: str,
    gender: str,
    n_betas: int,
    body_model_batch_size: int,
    misc_args: dict = None,
    device: torch.device = torch.device("cpu"),
):
    if misc_args is None: misc_args = {}
    body_model_type = check_body_model_type(body_model_type)

    assert isinstance(body_model_path, str), "body_model_path must be a string"
    assert os.path.isfile(
        body_model_path
    ), f"{body_model_path} is not a valid file path"

    assert isinstance(gender, str), "gender must be a string"
    gender = gender.lower()
    assert gender in ("male", "female", "neutral")

    assert type(n_betas) == int, "n_betas must be an integer"
    assert (
        type(body_model_batch_size) == int
    ), "body_model_batch_size must be an integer"
    assert isinstance(misc_args, dict), "misc_args must be a dictionary"
    assert isinstance(device, torch.device), "device must be a torch.device"

    if body_model_type == "smpl":
        body_model = body_model_class(
            model_path=body_model_path,
            gender=gender,
            num_betas=n_betas,
            batch_size=body_model_batch_size,
            device=device,
            **misc_args,
        )
    elif body_model_type == "smplh" or body_model_type == "smplx":
        body_model = body_model_class(
            model_path=body_model_path,
            gender=gender,
            num_betas=n_betas,
            batch_size=body_model_batch_size,
            use_pca=False,
            flat_hand_mean=True,
            device=device,
            **misc_args,
        )
    elif body_model_type == "supr":
        body_model = body_model_class(
            path_model=body_model_path,
            num_betas=n_betas,
            device=device,
        )

    return body_model


def get_body_model_params_info(body_model_type: str):

    body_model_type = check_body_model_type(body_model_type)

    if body_model_type == "smpl":
        return {"body_pose": 69, "global_orient": 3, "transl": 3}
    elif body_model_type == "smplh":
        return {"pose": 153, "global_orient": 3, "transl": 3}
    elif body_model_type == "smplx":
        return {"pose": 162, "global_orient": 3, "transl": 3}
    elif body_model_type == "supr":
        return {
            "pose": 225,
            "trans": 3,
        }


def perform_model_forward_pass(
    body_model_type: str,
    body_model: torch.nn.Module,
    params: dict,
    n_betas: int,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
):

    body_model_type = check_body_model_type(body_model_type)

    body_model = body_model.to(device)
    for param_name, param_value in params.items():
        params[param_name] = param_value.to(device)

    if params["betas"].shape[0] == 1 and batch_size != 1:
        betas = params["betas"].repeat(batch_size, 1)
    else:
        betas = params["betas"]

    betas = betas[..., :n_betas]

    if len(betas.shape) == 1:
        betas = betas.unsqueeze(0)

    if body_model_type == "smpl":

        if params["body_pose"].shape[-1] == 72:
            params["body_pose"] = params["body_pose"][..., 3:72]

        return body_model(
            betas=betas,
            body_pose=params["body_pose"],
            transl=params["transl"],
            global_orient=params["global_orient"],
        ).vertices

    elif body_model_type == "smplh":

        if params["pose"].shape[-1] == 156:
            params["pose"] = params["pose"][..., 3:156]

        return body_model(
            betas=betas,
            transl=params["transl"],
            global_orient=params["global_orient"],
            body_pose=params["pose"][..., :63],
            left_hand_pose=params["pose"][..., 63:108],
            right_hand_pose=params["pose"][..., 108:],
        ).vertices

    elif body_model_type == "smplx":

        if params["pose"].shape[-1] == 165:
            params["pose"] = params["pose"][..., 3:165]

        return body_model(
            betas=betas,
            transl=params["transl"],
            global_orient=params["global_orient"],
            body_pose=params["pose"][..., :63],
            jaw_pose=params["pose"][..., 63:66],
            leye_pose=params["pose"][..., 66:69],
            reye_pose=params["pose"][..., 69:72],
            left_hand_pose=params["pose"][..., 72:117],
            right_hand_pose=params["pose"][..., 117:],
        ).vertices

    elif body_model_type == "supr":
        return body_model(
            betas=betas,
            pose=params["pose"],
            trans=params["trans"],
        )["vertices"]

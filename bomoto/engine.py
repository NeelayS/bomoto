# import numpy as np
import torch

from bomoto.body_models import get_body_model  # perform_model_forward_pass,
from bomoto.body_models import get_body_model_params_info, instantiate_body_model
from bomoto.config import get_config
from bomoto.data import get_dataset

# from bomoto.losses import edge_loss, vertex_loss
from bomoto.utils import seed_everything, validate_device

# import trimesh
# from tqdm import tqdm


class Engine:
    """
    Main worker class to either fit body models or to convert
    between different types of body models.
    """

    def __init__(
        self,
        cfg_path: str,
    ):

        self.cfg = get_config(cfg_path)

    def _setup(self):

        seed_everything(self.cfg.seed)

        self._setup_device()

        self.input_body_model = None
        self.setup_dataloader()

        self._setup_output_body_model()

        self.params_info = get_body_model_params_info(self.cfg.output.body_model.type)
        self.output_body_model_params = {}
        self.output_body_model_params["betas"] = None
        for params_name in self.params_info.keys():
            self.output_body_model_params[params_name] = None
        self._init_params()

        self._setup_deformation()
        self._setup_vertex_masking()

    def _setup_device(self):

        self.device = validate_device(self.cfg.device)

    def _setup_deformation(
        self,
    ):
        pass

    def _setup_vertex_masking(
        self,
    ):
        pass

    def _init_params(
        self,
    ):

        if self.cfg.single_set_of_betas is True:
            self.output_body_models_params["betas"] = torch.zeros(
                (1, self.cfg.output.body_model.n_betas),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
        else:
            self.output_body_models_params["betas"] = torch.zeros(
                (self.cfg.batch_size, self.cfg.output.body_model.n_betas),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )

    def _setup_input_body_model(
        self,
    ):

        if self.cfg.input.body_model.misc_args is None:
            misc_args = {}
        else:
            misc_args = self.cfg.input.body_model.misc_args.to_dict()

        self.input_body_model = self._setup_model(
            body_model_type=self.cfg.input.body_model.type,
            body_model_path=self.cfg.input.body_model.path,
            gender=self.cfg.input.body_model.gender,
            n_betas=self.cfg.input.body_model.n_betas,
            body_model_batch_size=self.cfg.batch_size,
            misc_args=misc_args,
        )

    def _setup_output_body_model(
        self,
    ):

        if self.cfg.output.body_model.misc_args is None:
            misc_args = {}
        else:
            misc_args = self.cfg.output.body_model.misc_args.to_dict()

        self.output_body_model = self._setup_model(
            body_model_type=self.cfg.output.body_model.type,
            body_model_path=self.cfg.output.body_model.path,
            gender=self.cfg.output.body_model.gender,
            n_betas=self.cfg.output.body_model.n_betas,
            body_model_batch_size=self.cfg.batch_size,
            misc_args=misc_args,
        )

        self.output_body_model_faces = torch.tensor(
            self.output_body_model.faces, dtype=torch.long, device=self.device
        )

    def _setup_model(
        self,
        body_model_type: str,
        body_model_path: str,
        gender: str,
        n_betas: int,
        body_model_batch_size: int,
        misc_args: dict = {},
    ):
        body_model_class = get_body_model(body_model_type)
        body_model = instantiate_body_model(
            body_model_type=body_model_type,
            body_model_class=body_model_class,
            body_model_path=body_model_path,
            gender=gender,
            n_betas=n_betas,
            body_model_batch_size=body_model_batch_size,
            misc_args=misc_args,
            device=self.device,
        )

        return body_model

    def setup_dataloader(
        self,
    ):

        input_data_type, dataset_class, dataloader_batch_size = get_dataset(
            input_data_type=self.cfg.input.data.type,
            dataloader_batch_size=self.cfg.batch_size,
        )

        if input_data_type == "mesh_dir":
            self.dataset = dataset_class(
                mesh_dir=self.cfg.input.data.mesh_dir,
                mesh_format=self.cfg.input.data.mesh_format,
            )

        elif input_data_type == "npz_file":
            if self.input_body_model is None:
                self._setup_input_body_model()

            self.dataset = dataset_class(
                body_model=self.input_body_model,
                body_model_type=self.cfg.input.body_model.type,
                npz_file_dir=self.cfg.input.data.npz_file_dir,
                n_betas=self.cfg.input.body_model.n_betas,
                device=self.device,
            )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=dataloader_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataloader_num_workers,
        )


# During first iteration, check that the batch size obtained from datalaoader is
# the same as the class batch size
# if not, raise error
# if NPZ file dataset is being used, the model batch size must
# be equal to the number of frames in a file and the dataloader
# batch size must be 1

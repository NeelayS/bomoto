from .body_model import BodyModel
from .smpl import SMPLWrapper
from .smplh import SMPLHWrapper
from .smplx import SMPLXWrapper
from .supr import SUPRWrapper
from .utils import get_model_params

BodyModel.body_models = {'smpl': SMPLWrapper,
                         'smplh': SMPLHWrapper,
                         'smplx': SMPLXWrapper,
                         'supr': SUPRWrapper}

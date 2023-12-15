from .body_model import BodyModel
from .smpl import SMPLWrapper
from .smplh import SMPLHWrapper
from .smplx import SMPLXWrapper
from .supr import SUPRWrapper
from .utils import get_model_params, rotate_points_around_axis, interpolate_parameters, rotvec_slerp, lerp

BodyModel.body_models = {'smpl': SMPLWrapper,
                         'smplh': SMPLHWrapper,
                         'smplx': SMPLXWrapper,
                         'supr': SUPRWrapper}

from .body_model import BodyModel
from .smpl import SMPLWrapper
from .smplh import SMPLHWrapper
from .smplx import SMPLXWrapper


class MissingBodyModel:
    def __init__(self, *args, **kwargs):
        raise ImportError(f'The requested body model is not installed')


try:
    from .supr import SUPRWrapper
except ImportError:
    SUPRWrapper = MissingBodyModel

try:
    from .skel import SKELWrapper
except ImportError:
    SKELWrapper = MissingBodyModel

from .utils import get_model_params, rotate_points_around_axis, interpolate_parameters, rotvec_slerp, lerp

BodyModel.body_models = {'smpl': SMPLWrapper,
                         'smplh': SMPLHWrapper,
                         'smplx': SMPLXWrapper,
                         'supr': SUPRWrapper,
                         'skel': SKELWrapper}

from .base import ALIGNERS, AlignResult, FaceAligner, NoAlign, get_aligner, template_5pts  # noqa: F401
from . import insightface_2d106  # noqa: F401  レジストリへ登録される（順番が画面の並び順）
from . import insightface_scrfd  # noqa: F401
from . import retinaface_r50  # noqa: F401
from . import mtcnn  # noqa: F401
from . import cvlface_dfa  # noqa: F401

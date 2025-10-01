from .trainer import trainer_challenge_1, trainer_challenge_2
from .setup_logger import setup_logger

from .challenges import challenge1, challenge2
from .utils import list_of_str, str2bool

__all__ = ["trainer_challenge_1", 
            "trainer_challenge_2", 
            "setup_logger", 
            "list_of_str"]

print(__all__)
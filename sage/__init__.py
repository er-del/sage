"""
SAGE — Self-Adaptive General Engine
A complete mini-LLM system built from scratch.
"""

__version__ = "1.0.0"

from .model import SageModel
from .config import SageConfig
from .data import SageTokenizer
from .inference import generate
from .memory import ConversationHistory, RAGManager
from .train import train
from .finetune import finetune_instruction as finetune
from .utils import get_compatible_device

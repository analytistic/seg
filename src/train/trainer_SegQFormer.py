from collections.abc import Callable
from typing import Any

from torch._tensor import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers import BaseImageProcessor, FeatureExtractionMixin, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction



class SegQFormerTrainer(Trainer):
    """Custom Trainer for SegFormer model."""
    
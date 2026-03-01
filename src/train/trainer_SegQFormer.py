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
    def __init__(self, 
                 model: PreTrainedModel | Module | None = None, 
                 args: TrainingArguments | None = None, 
                 data_collator: Callable[[list[Any]], dict[str, Any]] | None = None, 
                 train_dataset: Dataset | IterableDataset | Any | None = None, 
                 eval_dataset: Dataset | dict[str, Dataset] | Any | None = None, 
                 processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None = None, 
                 model_init: Callable[..., PreTrainedModel] | None = None, 
                 compute_loss_func: Callable[..., Any] | None = None, 
                 compute_metrics: Callable[[EvalPrediction], dict] | None = None, 
                 callbacks: list[TrainerCallback] | None = None, 
                 optimizers: tuple[Optimizer | None, LambdaLR | None] = (None, None), 
                 optimizer_cls_and_kwargs: tuple[type[Optimizer], dict[str, Any]] | None = None, 
                 preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)

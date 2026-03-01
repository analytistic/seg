from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers

@dataclass
class DataArguments:
    datasets: str = field(
        default="",
        metadata={
            "help": "data name "
        },
    )
    datasets_path: str = field(
        default="",
        metadata={
            "help": "data path"
        },
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="",
        metadata={
            "help": "model name or path"
        },
    )
    use_cache: bool = field(
        default=True,
        metadata={
            "help": "Whether to use cache"
        },
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze the backbone"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class MultimodalArguments:
    image_processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained image processor or model identifier from huggingface.co/models."
        },
    )
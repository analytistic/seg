from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers
import toml


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

    @classmethod
    def from_toml(cls, toml_path: Optional[str] = None):
        if toml_path is not None:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
                args = data.get('data_args', {})
            return cls(**args)
        else:
            return cls()
    

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
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
    @classmethod
    def from_toml(cls, toml_path: Optional[str] = None):
        if toml_path is not None:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
                args = data.get('model_args', {})
            return cls(**args)
        else:
            return cls()


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
    @classmethod
    def from_toml(cls, toml_path: Optional[str] = None):
        if toml_path is not None:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
                args = data.get('training_args', {})
            return cls(**args)
        else:
            return cls()

@dataclass
class MultimodalArguments:
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained image processor or model identifier from huggingface.co/models."
        },
    )
    @classmethod
    def from_toml(cls, toml_path: Optional[str] = None):
        if toml_path is not None:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
                args = data.get('multimodal_args', {})
            return cls(**args)
        else:
            return cls()
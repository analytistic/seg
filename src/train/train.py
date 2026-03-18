from src.train.trainer_SegQFormer import SegQFormerTrainer
from src.utils.arguments import TrainingArguments, ModelArguments, DataArguments, MultimodalArguments
from transformers import HfArgumentParser
from src.model.SegQFormer.modeling_SegQFormer import SegQFormerForSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
import torch
from PIL import Image
import requests
from src.datasets.gis_dataset import GisSegDataset
from src.datasets import DataCollatorForSupervisedDataset
from src.train.eval import MetricsComputer
from transformers import set_seed




def train():
    """Train the SegFormer model."""
    # args
    # parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    config_path = 'src/config/segqformer.toml'
    model_args = ModelArguments.from_toml(config_path)
    data_args = DataArguments.from_toml(config_path)
    training_args = TrainingArguments.from_toml(config_path)
    multimodal_args = MultimodalArguments.from_toml(config_path)
    set_seed(training_args.seed)

    model = SegQFormerForSegmentation.from_pretrained(
        model_args.pretrained_model_name_or_path,
        ignore_mismatched_sizes=True,
        device_map='auto',
        dtype='auto',
    )
    dataset = GisSegDataset(
        datasets=data_args.datasets,
        datasets_path=data_args.datasets_path,
        multimodal_args=multimodal_args
    )
    

    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    trainer = SegQFormerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSupervisedDataset(),
        compute_metrics=MetricsComputer(dataset.processor),
        processing_class=dataset.processor
    )

    trainer.train()


    # dataset

    # trainer train and save

if __name__ == "__main__":
    train()


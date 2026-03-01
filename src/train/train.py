from train.trainer_SegQFormer import SegQFormerTrainer
from src.utils.arguments import TrainingArguments, ModelArguments, DataArguments
from transformers import HfArgumentParser, TrainingArguments
from transformers import Mask2FormerModel, AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
import torch
from PIL import Image
import requests




def train():
    """Train the SegFormer model."""
    # args
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    

    # model
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    inputs = processor(images=image, return_tensors="pt")

    # dataset

    # trainer train and save

if __name__ == "__main__":
    train()


from torch.utils.data import Dataset
from src.utils.arguments import MultimodalArguments
from src.model.SegQFormer.image_processing_SegQFormer import SegQFormerImageProcessor


class BaseDataset(Dataset):
    def __init__(
        self,
        datasets: str,
        multimodal_args: MultimodalArguments,
        label2semantic_id: dict
    ):
        super().__init__()
        self.multimodal_args = multimodal_args
        self.label2semantic_id = label2semantic_id


    def get_processor(self, multimodal_args: MultimodalArguments):
        processor_kwargs = vars(self.multimodal_args)
        processor = SegQFormerImageProcessor.from_pretrained(**processor_kwargs)
        return processor
    
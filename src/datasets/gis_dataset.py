from src.datasets.base_dataset import BaseDataset
from src.datasets.utils.get_data import get_gid_data
from src.utils.arguments import MultimodalArguments
from src.datasets import GIDSuperviseData
import json 
from PIL import Image
import torch
import numpy as np
from typing import Dict 

class GisSegDataset(BaseDataset):
    def __init__(self, 
                 datasets: str, 
                 datasets_path: str, 
                 multimodal_args: MultimodalArguments, 
                 label2semantic_id: Dict = {
                    'Built_up': 1,
                    'Farmland': 2,
                    'Water': 3,
                    'Background': 0
                 }
                ):
        super().__init__(datasets, multimodal_args, label2semantic_id)
        self.datasets = datasets
        self.datasets_path = datasets_path
        self.data = json.load(open(datasets_path, 'r'))
        self.processor = self.get_processor(multimodal_args)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data: GIDSuperviseData = get_gid_data(data, transform=None)
        inputs = self.processor(
            images=data.image,
            segmentation_maps=data.segmentation_maps,
            instance_id_to_semantic_id=None,
            return_tensors="pt"
        )

        return dict(
            pixel_values=inputs.data['pixel_values'],
            pixel_mask=inputs.data['pixel_mask'],
            multi_mask_labels=inputs.data['multi_mask_labels'],
            binary_masks_labels=inputs.data['binary_masks_labels'],
        )
    

if __name__ == "__main__":
    dataset = GisSegDataset(
        datasets='gis',
        datasets_path='/Users/alex/project/seg/data/gis_dataset/train.json',
        multimodal_args=MultimodalArguments()
    )
    print(dataset[0])
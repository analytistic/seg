from PIL import Image
import numpy as np
from typing import Callable, Optional, Dict
from src.datasets import GIDSuperviseData

def get_gid_data(data, transform: Optional[Callable] = None):

    image = np.array(Image.open(data['image_path']))
    label = (np.array(Image.open(data['label_path']).convert('RGB'))).astype(np.uint8)
    classes = data['classes']
    if transform:
        image, label = transform(image, label)
    
    # Convert label to class indices
    inputs = GIDSuperviseData(
        image=image,
        segmentation_maps=label,
        classes=classes
    )
    label2pixel = GIDSuperviseData.label2pixel
    label2semantic_id = GIDSuperviseData.label2semantic_id
    segmentation_maps = []
    for key, value in label2pixel.items():
        mask = np.all(label == value, axis=-1).astype(np.uint8) * label2semantic_id[key]
        segmentation_maps.append(mask)
    segmentation_maps = np.stack(segmentation_maps, axis=0).sum(axis=0)

    return GIDSuperviseData(image=image, segmentation_maps=segmentation_maps, classes=classes)
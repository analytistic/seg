from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, ClassVar
import numpy as np
import torch

@dataclass
class GIDSuperviseData:
    image: np.ndarray 
    segmentation_maps: np.ndarray 
    classes: Dict[str, bool]
    label2pixel: ClassVar[Dict[str, np.ndarray]] = {
        'Built_up': np.array([255, 0, 0]),
        'Farmland': np.array([0, 255, 0]),
        'Water': np.array([0, 0, 255]),
    }
    label2semantic_id: ClassVar[Dict[str, int]] = {
        'Built_up': 1,
        'Farmland': 2,
        'Water': 3,
    }
    

@dataclass
class DataCollatorForSupervisedDataset:
    def __call__(self, instances) -> Dict[str, Any]:
        
        batch = dict(
            pixel_values=torch.cat([instance['pixel_values'] for instance in instances], dim=0),
            pixel_mask=torch.cat([instance['pixel_mask'] for instance in instances], dim=0),
            multi_mask_labels=torch.cat([instance['multi_mask_labels'] for instance in instances], dim=0),
            binary_masks_labels=torch.cat([instance['binary_masks_labels'] for instance in instances], dim=0),
        )
        return batch

import numpy as np
from transformers import EvalPrediction
from src.model.SegQFormer.image_processing_SegQFormer import SegQFormerImageProcessor
def compute_confusion_matrix(pred_flat, label_flat, num_labels: int, ignore_index=None):
    mask = (label_flat >= 0) & (label_flat != ignore_index) if ignore_index is not None else (label_flat >= 0)
    return np.bincount(
        num_labels * label_flat[mask].astype(int) + pred_flat[mask].astype(int),
        minlength=num_labels**2
    ).reshape(num_labels, num_labels)


def compute_miou(confusion_matrix):
    miou = {}
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        denom = tp + fp + fn
        miou[f'{i}'] = tp / denom if denom > 0 else 0.0
    return miou

def compute_pa(confusion_matrix):
    tp = np.diag(confusion_matrix)
    total = confusion_matrix.sum()
    return tp.sum() / total if total > 0 else 0.0
    
class MetricsComputer:
    def __init__(self, processor: SegQFormerImageProcessor):
        self.processor = processor
    def __call__(self, eval_pred: EvalPrediction):
        outputs = eval_pred.predictions
        loss_dict = outputs[0]
        labels = eval_pred.label_ids[0]
        semantic_segmentation = self.processor.post_process_semantic_segmentation(
            outputs=outputs[1],
            target_sizes=[(label.shape[0], label.shape[1]) for label in labels]
        )
        confusion_matrix = compute_confusion_matrix(
            pred_flat=semantic_segmentation.flatten(),
            label_flat=labels.flatten(),
            num_labels=self.processor.num_labels + 1,
            ignore_index=self.processor.ignore_index
        )
        miou = compute_miou(confusion_matrix)
        pa = compute_pa(confusion_matrix)

        results = {
            'pa': pa
        }
        results.update({f'miou_class_{key}': value for key, value in miou.items()})
        results['miou_mean'] = np.mean(list(miou.values()))
        results.update({f'{key}': value.mean() for key, value in loss_dict.items()})

        return results


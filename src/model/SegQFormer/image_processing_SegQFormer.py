from transformers import TensorType
from transformers.image_processing_base import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, ImageInput, valid_images, validate_preprocess_arguments, make_flat_list_of_images, to_numpy_array, infer_channel_dimension_format
from transformers.image_processing_utils import get_size_dict
from transformers.models.mask2former.image_processing_mask2former import Mask2FormerImageProcessor
from transformers.utils.generic import filter_out_non_signature_kwargs

from transformers import AutoConfig, AutoImageProcessor
from src.model.SegQFormer.configuration_SegQFormer import SegQFormerConfig
from src.model.SegQFormer.modeling_SegQFormer import SegQFormerForSegmentationOutput
import torch
import numpy as np
from typing import Iterable, Any

def max_across_indices(values: Iterable[Any]) -> list[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]

# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
def get_max_height_width(
    images: list[np.ndarray], input_data_format: str | ChannelDimension | None = None
) -> list[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return [max_height, max_width]


def convert_segmentation_map_to_binary_masks(
    segmentation_map: np.ndarray,
    num_labels: int,
    ignore_index: int | None = None,
    do_reduce_labels: bool = False,
):
    if do_reduce_labels and ignore_index is None:
        raise ValueError("If `do_reduce_labels` is True, `ignore_index` must be provided.")

    if do_reduce_labels:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)
        all_labels = np.arange(num_labels)
    # Get unique ids (class or instance ids based on input)
    else:
        all_labels = np.arange(num_labels+1)

    # Drop background label if applicable
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # Generate a binary mask for each object instance
    binary_masks = [(segmentation_map == i) for i in all_labels]

    # Stack the binary masks
    if binary_masks:
        binary_masks = np.stack(binary_masks, axis=0)
    else:
        binary_masks = np.zeros((0, *segmentation_map.shape))


    return binary_masks.astype(np.float32)

class SegQFormerImageProcessor(Mask2FormerImageProcessor):
    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(self, id2label: dict[int, str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.id2label = id2label

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        instance_id_to_semantic_id: dict[int, int] | None = None,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        size_divisor: int | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        ignore_index: int | None = None,
        do_reduce_labels: bool | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: str | ChannelDimension = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
        pad_size: dict[str, int] | None = None,
    )-> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False, max_size=self._max_size)
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        pad_size = pad_size if pad_size is not None else self.pad_size

        if not valid_images(images):
            raise ValueError(
                "Invalid image(s) provided. The images have to be of one of the following types: "
                "PIL Image, Tensor, np.ndarray, list of PIL Images, list of Tensors or list of np.ndarrays."
            )
        
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        
        if segmentation_maps is not None and not valid_images(segmentation_maps):
            raise ValueError(
                "Invalid segmentation map(s) provided. The segmentation maps have to be of one of the following types: "
                "PIL Image, Tensor, np.ndarray, list of PIL Images, list of Tensors or list of np.ndarrays."
            )
        
        images = make_flat_list_of_images(images)
        if segmentation_maps is not None:
            segmentation_maps = make_flat_list_of_images(segmentation_maps, expected_ndims=2)

        if segmentation_maps is not None and len(images) != len(segmentation_maps): # type: ignore
            raise ValueError(f"Got {len(images)} images and {len(segmentation_maps)} segmentation maps, but they should be of the same length.") # type: ignore 

        images = [
            self._preprocess_image(
                image,
                do_resize=do_resize,
                size=size,
                size_divisor=size_divisor,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in images # type: ignore
        ]

        if segmentation_maps is not None:
            segmentation_maps = [
                self._preprocess_mask(
                    segmentation_maps,
                    do_resize=do_resize,
                    size=size,
                    size_divisor=size_divisor,
                    input_data_format=input_data_format,
                )
                for segmentation_maps in segmentation_maps # type: ignore
            ]
        
        encoded_inputs = self.encode_inputs(
            images,
            segmentation_maps=segmentation_maps,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            do_reduce_labels=do_reduce_labels,
            return_tensors=return_tensors,
            input_data_format=data_format,
            pad_size=pad_size,
        )
        return encoded_inputs
    
    def encode_inputs(
        self,
        pixel_values_list: ImageInput,
        segmentation_maps: ImageInput | None = None,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = None,
        ignore_index: int | None = None,
        do_reduce_labels: bool = False,
        return_tensors: str | TensorType | None = None,
        input_data_format: str | ChannelDimension | None = None,
        pad_size: dict[str, int] | None = None,
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        Mask2Former addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps
        will be converted to lists of binary masks and their respective labels. Let's see an example, assuming
        `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels =
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for
        each mask.

        Args:
            pixel_values_list (`ImageInput`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape `(channels, height,
                width)`.

            segmentation_maps (`ImageInput`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.

             (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            instance_id_to_semantic_id (`list[dict[int, int]]` or `dict[int, int]`, *optional*):
                A mapping between object instance ids and class ids. If passed, `segmentation_maps` is treated as an
                instance segmentation map where each pixel represents an instance id. Can be provided as a single
                dictionary with a global/dataset-level mapping or as a list of dictionaries (one per image), to map
                instance ids in each image separately.

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

            pad_size (`Dict[str, int]`, *optional*):
                The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
                provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
                height and width in the batch.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `=True` or if `pixel_mask` is in
              `self.model_input_names`).
            - **mask_labels** -- Optional list of mask labels of shape `(labels, height, width)` to be fed to a model
              (when `annotations` are provided).
            - **class_labels** -- Optional list of class labels of shape `(labels)` to be fed to a model (when
              `annotations` are provided). They identify the labels of `mask_labels`, e.g. the label of
              `mask_labels[i][j]` if `class_labels[i][j]`.
        """
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        do_reduce_labels = self.do_reduce_labels if do_reduce_labels is None else do_reduce_labels

        pixel_values_list = [to_numpy_array(pixel_values) for pixel_values in pixel_values_list]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(pixel_values_list[0])

        encoded_inputs = self.pad(
            pixel_values_list, return_tensors=return_tensors, input_data_format=input_data_format, pad_size=pad_size
        )

        if segmentation_maps is not None:
            mask_labels = []
            pad_size = get_max_height_width(pixel_values_list, input_data_format=input_data_format)
            # Convert to list of binary masks and labels
            for idx, segmentation_map in enumerate(segmentation_maps):
                segmentation_map = to_numpy_array(segmentation_map)
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id
                # Use instance2class_id mapping per image
                masks = self.convert_segmentation_map_to_binary_masks(
                    segmentation_map, num_labels=self.num_labels, ignore_index=ignore_index, do_reduce_labels=do_reduce_labels
                )
                # We add an axis to make them compatible with the transformations library
                # this will be removed in the future
                if masks.shape[0] > 0:
                    masks = [mask[None, ...] for mask in masks]
                    masks = [
                        self._pad_image(image=mask, output_size=pad_size, constant_values=ignore_index)
                        for mask in masks
                    ]
                    masks = np.concatenate(masks, axis=0)
                else:
                    masks = np.zeros((0, *pad_size), dtype=np.float32)
                mask_labels.append(torch.from_numpy(masks))

            # we cannot batch them since they don't share a common class size
            encoded_inputs["multi_mask_labels"] = torch.from_numpy(np.stack(segmentation_maps, axis=0)).to(torch.float32) # type: ignore
            encoded_inputs["binary_masks_labels"] = torch.stack(mask_labels, dim=0)

        return encoded_inputs
    
    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: np.ndarray,
        num_labels: int,
        ignore_index: int | None = None,
        do_reduce_labels: bool = False,
    ):
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        return convert_segmentation_map_to_binary_masks(
            segmentation_map=segmentation_map,
            num_labels=num_labels,
            ignore_index=ignore_index,
            do_reduce_labels=do_reduce_labels,
        )
    
    def post_process_semantic_segmentation(
        self, outputs: SegQFormerForSegmentationOutput | torch.Tensor | np.ndarray, target_sizes: list[tuple[int, int]] | None = None
    ) -> np.ndarray:
        masks_queries_logits = outputs.masks_queries_logits if isinstance(outputs, SegQFormerForSegmentationOutput) else outputs
        segmentation = torch.from_numpy(masks_queries_logits) if isinstance(masks_queries_logits, np.ndarray) else masks_queries_logits
        segmentation = segmentation.softmax(dim=1)
        batch_size = segmentation.shape[0]
        if target_sizes is not None:
            assert batch_size == len(target_sizes), "Make sure the batch size of the outputs and the target sizes match."
            semantic_segmentation = []
            for idx in range(batch_size):
                semantic_segmentation.append(
                    torch.nn.functional.interpolate(
                        segmentation[idx:idx+1], size=target_sizes[idx], mode="bilinear", align_corners=False
                    )[0].argmax(dim=0)
                )
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return torch.stack(semantic_segmentation, dim=0).cpu().detach().numpy()
    
    def decode(
        self,
        outputs: SegQFormerForSegmentationOutput | torch.Tensor,
        target_sizes: list[tuple[int, int]] | None = None,
        do_reduce_labels: bool = False,
    ):
        do_reduce_labels = self.do_reduce_labels if do_reduce_labels is None else do_reduce_labels
        results = []
        semantic_segmentation = self.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        for label_id, segmentation in enumerate(semantic_segmentation):
            if do_reduce_labels:
                label_id += 1
            results.append(
                {
                    "label": self.id2label[label_id] if self.id2label is not None else label_id,
                    "mask": (segmentation == label_id).to(torch.int32)
                }
            )
        return results
    

AutoConfig.register("segqformer", SegQFormerConfig)
AutoImageProcessor.register(SegQFormerConfig, SegQFormerImageProcessor)
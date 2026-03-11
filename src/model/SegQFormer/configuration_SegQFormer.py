from transformers.backbone_utils import consolidate_backbone_kwargs_to_config
from transformers.configuration_utils import PreTrainedConfig
from transformers import Mask2FormerConfig
from transformers.utils import logging
from transformers.models import AutoConfig


logger = logging.get_logger(__name__)


class SegQFormerConfig(Mask2FormerConfig):
    r"""
    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `SwinConfig()`):
            The configuration of the backbone model. If unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        feature_size (`int`, *optional*, defaults to 256):
            The features (channels) of the resulting feature maps.
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
            Dimension of feedforward network for deformable detr encoder used as part of pixel decoder.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of layers in the deformable detr encoder used as part of pixel decoder.
        decoder_layers (`int`, *optional*, defaults to 10):
            Number of layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feature dimension in feedforward network for transformer decoder.
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to use pre-LayerNorm or not for transformer decoder.
        enforce_input_projection (`bool`, *optional*, defaults to `False`):
            Whether to add an input projection 1x1 convolution even if the input channels and hidden dim are identical
            in the Transformer decoder.
        common_stride (`int`, *optional*, defaults to 4):
            Parameter used for determining number of FPN levels used as part of pixel decoder.
        ignore_value (`int`, *optional*, defaults to 255):
            Category id to be ignored during training.
        num_queries (`int`, *optional*, defaults to 100):
            Number of queries for the decoder.
        no_object_weight (`int`, *optional*, defaults to 0.1):
            The weight to apply to the null (no object) class.
        class_weight (`int`, *optional*, defaults to 2.0):
            The weight for the cross entropy loss.
        mask_weight (`int`, *optional*, defaults to 5.0):
            The weight for the mask loss.
        dice_weight (`int`, *optional*, defaults to 5.0):
            The weight for the dice loss.
        train_num_points (`str` or `function`, *optional*, defaults to 12544):
            Number of points used for sampling during loss calculation.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Oversampling parameter used for calculating no. of sampled points
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        use_auxiliary_loss (`boolean``, *optional*, defaults to `True`):
            If `True` [`Mask2FormerForUniversalSegmentationOutput`] will contain the auxiliary losses computed using
            the logits from each decoder's stage.
        feature_strides (`list[int]`, *optional*, defaults to `[4, 8, 16, 32]`):
            Feature strides corresponding to features generated from backbone network.
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.
    """

    model_type = "segqformer"
    sub_configs = {"backbone_config": AutoConfig} # type: ignore
    backbones_supported = ["swin"]
    attribute_map = {"hidden_size": "hidden_dim"}

    def __init__(
        self,
        backbone_config: dict | PreTrainedConfig | None = None,
        feature_size: int = 256,
        mask_feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_feedforward_dim: int = 1024,
        activation_function: str = "relu",
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        enforce_input_projection: bool = False,
        common_stride: int = 4,
        ignore_value: int = 255,
        num_queries: int = 100,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        cross_entropy_weight: float = 5.0,
        bce_weight: float = 25.0,
        water_dice_weight: float = 30.0,
        dice_weight: float = 5.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        use_auxiliary_loss: bool = True,
        feature_strides: list[int] = [4, 8, 16, 32],
        output_auxiliary_logits: bool | None = None,
        **kwargs,
    ):
        super().__init__(
        backbone_config=backbone_config,
        feature_size=feature_size,
        mask_feature_size=mask_feature_size,
        hidden_dim=hidden_dim,
        encoder_feedforward_dim=encoder_feedforward_dim,
        activation_function=activation_function,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        pre_norm=pre_norm,
        enforce_input_projection=enforce_input_projection,
        common_stride=common_stride,
        ignore_value=ignore_value,
        num_queries=num_queries,
        no_object_weight=no_object_weight,
        class_weight=class_weight,
        mask_weight=mask_weight,
        dice_weight=dice_weight,
        train_num_points=train_num_points,
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
        init_std=init_std,
        init_xavier_std=init_xavier_std,
        use_auxiliary_loss=use_auxiliary_loss,
        feature_strides=feature_strides,
        output_auxiliary_logits=output_auxiliary_logits,
        **kwargs,
        )
        self.cross_entropy_weight = cross_entropy_weight
        self.bce_weight = bce_weight
        self.water_dice_weight = water_dice_weight


__all__ = ["SegQFormerConfig"]

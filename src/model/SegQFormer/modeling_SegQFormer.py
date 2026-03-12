from transformers import Mask2FormerPreTrainedModel, GradientCheckpointingLayer
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoder, Mask2FormerSinePositionEmbedding, Mask2FormerTransformerModule, Mask2FormerMaskedAttentionDecoder, Mask2FormerAttention
from transformers.backbone_utils import load_backbone
from transformers.utils.generic import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.utils.auto_docstring import auto_docstring
from transformers.activations import ACT2FN
from src.model.SegQFormer.configuration_SegQFormer import SegQFormerConfig
from torch import nn
import torch
from dataclasses import dataclass



def sigmoid_bce_loss(inputs: torch.Tensor, labels: torch.Tensor, class_id: int) -> torch.Tensor:
    r"""
    transform the multi-class predict into binary predict to fix on class_id
    
    """
    prods = inputs.softmax(dim=1)[:, class_id, ...]
    labels_binary = labels[:, class_id, ...].to(torch.float32)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(prods, labels_binary)
    return loss



# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid()
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.mean(0)
    if weight is not None:
        loss = loss * weight
    loss = loss.sum()
    return loss


def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """

    criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean")
    loss = criterion(inputs, labels.to(dtype=torch.long))
    return loss



# copy from https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs
) -> torch.Tensor:
    """
    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features

@dataclass
class SegQFormerForSegmentationOutput(ModelOutput):
    r"""
    loss (`torch.Tensor`, *optional*):
        The computed loss, returned when labels are present.
    masks_queries_logits (`torch.Tensor`):
        A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
        query.
    auxiliary_logits (`list[Dict(str, torch.Tensor)]`, *optional*):
        List of class and mask predictions from each layer of the transformer decoder.
    encoder_last_hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
        Last hidden states (final feature map) of the last stage of the encoder model (backbone).
    pixel_decoder_last_hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
        Last hidden states (final feature map) of the last stage of the pixel decoder model.
    transformer_decoder_last_hidden_state (`tuple(torch.Tensor)`):
        Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
    encoder_hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
        model at the output of each stage.
    pixel_decoder_hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
        decoder model at the output of each stage.
    transformer_decoder_hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
        transformer decoder at the output of each stage.
    attentions (`tuple(tuple(torch.Tensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `tuple(torch.Tensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """

    loss: torch.Tensor | None = None
    loss_dict: dict[str, torch.Tensor] | None = None
    masks_queries_logits: torch.Tensor | None = None
    auxiliary_logits: list[dict[str, torch.Tensor]] | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    pixel_decoder_last_hidden_state: torch.Tensor | None = None
    transformer_decoder_last_hidden_state: torch.Tensor | None = None
    encoder_hidden_states: tuple[torch.Tensor] | None = None
    pixel_decoder_hidden_states: tuple[torch.Tensor] | None = None
    transformer_decoder_hidden_states: torch.Tensor | None = None
    attentions: tuple[torch.Tensor] | None = None

@dataclass
class SegQFormerModelOutput(ModelOutput):
    r"""
    encoder_last_hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
        Last hidden states (final feature map) of the last stage of the encoder model (backbone). Returned when
        `output_hidden_states=True` is passed.
    pixel_decoder_last_hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
        Last hidden states (final feature map) of the last stage of the pixel decoder model.
    transformer_decoder_last_hidden_state (`tuple(torch.Tensor)`):
        Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
    encoder_hidden_states (`tuple(torch.Tensor)`, *optional*):
        Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
        model at the output of each stage. Returned when `output_hidden_states=True` is passed.
    pixel_decoder_hidden_states (`tuple(torch.Tensor)`, , *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
        decoder model at the output of each stage. Returned when `output_hidden_states=True` is passed.
    transformer_decoder_hidden_states (`tuple(torch.Tensor)`, *optional*):
        Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
        transformer decoder at the output of each stage. Returned when `output_hidden_states=True` is passed.
    transformer_decoder_intermediate_states (`tuple(torch.Tensor)` of shape `(num_queries, 1, hidden_size)`):
        Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
        layernorm.
    masks_queries_logits (`tuple(torch.Tensor)` of shape `(batch_size, num_queries, height, width)`)
        Mask Predictions from each layer in the transformer decoder.
    attentions (`tuple(tuple(torch.Tensor))`, *optional*, returned when `output_attentions=True` is passed):
        Tuple of `tuple(torch.Tensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Self attentions weights from transformer decoder.
    """

    encoder_last_hidden_state: torch.Tensor | None = None
    pixel_decoder_last_hidden_state: torch.Tensor | None = None
    transformer_decoder_last_hidden_state: torch.Tensor | None = None
    encoder_hidden_states: tuple[torch.Tensor] | None = None
    pixel_decoder_hidden_states: tuple[torch.Tensor] | None = None
    transformer_decoder_hidden_states: tuple[torch.Tensor] | None = None
    transformer_decoder_intermediate_states: tuple[torch.Tensor] | None = None
    masks_queries_logits: tuple[torch.Tensor] | None = None
    attentions: tuple[torch.Tensor] | None = None

@dataclass
class SegQFormerPixelLevelModuleOutput(ModelOutput):
    r"""
    encoder_last_hidden_state (`torch.Tensor`):
        Last hidden states (final feature map of shape `(batch_size, num_channels, height, width)`) of the last
        stage of the encoder.
    encoder_hidden_states (`tuple(torch.Tensor)`, *optional*):
        Tuple of `torch.Tensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
        called feature maps) of the model at the output of each stage. Returned if output_hidden_states is set to
        True.
    decoder_last_hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, height, width)):
        1/4 scale features from the last Pixel Decoder Layer.
    decoder_hidden_states (`tuple(torch.Tensor)`):
        Tuple of `torch.Tensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
        called feature maps) of the model at the output of each stage.
    """

    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_hidden_states: tuple[torch.Tensor] | None = None
    decoder_last_hidden_state: torch.Tensor | None = None
    decoder_hidden_states: tuple[torch.Tensor] | None = None

@dataclass
class SegQFormerMaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    r"""
    hidden_states (`tuple(torch.Tensor)`, *optional*):
        Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each layer) of
        shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
        plus the initial embedding outputs. Returned when `output_hidden_states=True`.
    attentions (`tuple(torch.Tensor)`, *optional*):
        Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
        the self-attention heads. Returned when `output_attentions=True`.
    masks_queries_logits (`tuple(torch.Tensor)` of shape `(batch_size, num_queries, height, width)`):
        Tuple of mask predictions from all layers of the transformer decoder.
    intermediate_hidden_states (`tuple(torch.Tensor)` of shape `(num_queries, 1, hidden_size)`):
        Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
        layernorm.
    """

    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor] | None = None
    attentions: torch.Tensor | None = None
    masks_queries_logits: tuple[torch.Tensor] | None = None
    intermediate_hidden_states: tuple[torch.Tensor] | None = None


class SegQFormerPixelLevelModule(nn.Module):
    """
    Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2211.13227)
    """
    def __init__(self, config: SegQFormerConfig):
        super().__init__()
        self.encoder = load_backbone(config.backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)
    
    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False) -> SegQFormerPixelLevelModuleOutput:
        backbone_features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return SegQFormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


class SegQFormerMaskedAttentionDecoder(Mask2FormerMaskedAttentionDecoder):
    def __init__(self, config: SegQFormerConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [SegQFormerMaskedAttentionDecoderLayer(config) for _ in range(self.decoder_layers)]
        )
        



class SegQFormerTransformerModule(nn.Module):
    """
    Transformer Module proposed in [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2211.13227)
    """
    def __init__(self, in_features: int, config: SegQFormerConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = SegQFormerMaskedAttentionDecoder(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)


    def forward(
        self, 
        multi_scale_features: list[torch.Tensor],
        mask_features: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> SegQFormerMaskedAttentionDecoderOutput:
        
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(
                self.position_embedder(
                    multi_scale_features[i].shape, multi_scale_features[i].device, multi_scale_features[i].dtype, None
                ).flatten(2)
            )
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # Flatten (batch_size, num_channels, height, width) -> (height*width, batch_size, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        decoder_output = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_output
        



class SegQFormerModel(Mask2FormerPreTrainedModel):
    config: SegQFormerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'
    input_modalities = ('image',)

    def __init__(self, config: SegQFormerConfig):
        super().__init__(config)
        self.pixel_level_module = SegQFormerPixelLevelModule(config)
        self.transformer_module = SegQFormerTransformerModule(in_features=config.feature_size, config=config)
        
        self.post_init()


    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> SegQFormerModelOutput | tuple[torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states
        )

        transformer_module_output = self.transformer_module(
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,
            mask_features=pixel_level_module_output.decoder_last_hidden_state,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states

        output = SegQFormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output 
    

class SegQFormerLoss(nn.Module):
    def __init__(self, config: SegQFormerConfig, weight_dict: dict[str, float]):
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

         # pointwise mask loss parameters
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(logits.dtype).eps
        uncertainty_scores = -(torch.abs(logits)+eps)
        return uncertainty_scores


    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int | float,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int` | `float`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates


    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        multi_mask_labels: torch.Tensor,
        binary_masks_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_multi_mask_labels = multi_mask_labels[:, None]
        target_binary_mask_labels = binary_masks_labels
        pred_masks = masks_queries_logits

        # sample point coordinates for fixed computation budget
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_multi_labels = sample_point(target_multi_mask_labels, point_coordinates, align_corners=False, mode='nearest').squeeze(1)
            point_binary_labels = sample_point(target_binary_mask_labels, point_coordinates, align_corners=False, mode='nearest').squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False, mode='bilinear').squeeze(1)

        losses = {
            'loss_ce': sigmoid_cross_entropy_loss(point_logits, point_multi_labels),
            'loss_dice': dice_loss(point_logits, point_binary_labels),
            'loss_bce': sigmoid_bce_loss(point_logits, point_binary_labels, class_id=3),
            'loss_water_dice': dice_loss(point_logits[:, 3, ...], point_binary_labels[:, 3, ...]),
        }

        del pred_masks
        del target_multi_mask_labels
        del target_binary_mask_labels

        return losses

    def forward(
        self, 
        masks_queries_logits: torch.Tensor,
        multi_mask_labels: torch.Tensor,
        binary_masks_labels: torch.Tensor,
        auxiliary_predictions: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {
            **self.loss_masks(masks_queries_logits, multi_mask_labels, binary_masks_labels),
        }
        if  auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs['masks_queries_logits']
                loss_dict = self.forward(
                    masks_queries_logits=masks_queries_logits,
                    multi_mask_labels=multi_mask_labels,
                    binary_masks_labels=binary_masks_labels,
                )
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)
        return losses
            

class SegFormerHeteroCrossAttention(nn.Module):
    """
    Heterogeneous attention that
    query: 
    q_i = W_q_i * x_i
    key:
    k = W_k * x
    value:
    v = W_v * x
    """
    def __init__(self, query_size: int, embed_dim: int, num_heads: int, rank: int | None, dropout: float=0.0, bias: bool=True, add_bias_kv=False, add_zero_attn=False, device=None, dtype=None):  
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        if not rank:
            rank = (embed_dim * embed_dim) // (2 * embed_dim)


        self.query_proj_weight_L = nn.Parameter(
            torch.empty((embed_dim * query_size, rank), **factory_kwargs)
        )
        self.query_proj_weight_R = nn.Parameter(
            torch.empty((rank, embed_dim * query_size), **factory_kwargs)
        )
        self.in_proj_weight = nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    
    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        T, B, D = query.shape
        

        q_flattened = query.transpose(0, 1).reshape(B, T * D)
        q_intermediate = torch.nn.functional.linear(q_flattened, self.query_proj_weight_R)
        q_projected = torch.nn.functional.linear(q_intermediate, self.query_proj_weight_L)
        
        q = q_projected.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        w_k = self.in_proj_weight[self.embed_dim : 2 * self.embed_dim, :]
        w_v = self.in_proj_weight[2 * self.embed_dim : 3 * self.embed_dim, :]

        b_k = self.in_proj_bias[self.embed_dim : 2 * self.embed_dim] if self.in_proj_bias is not None else None
        b_v = self.in_proj_bias[2 * self.embed_dim : 3 * self.embed_dim] if self.in_proj_bias is not None else None
        
        S_k, B_k, _ = key.shape
        k = torch.nn.functional.linear(key, w_k, b_k).view(S_k, B_k, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = torch.nn.functional.linear(value, w_v, b_v).view(S_k, B_k, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask.view(B, -1, T, S_k) if attn_mask is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(T, B, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        attn_weight = None # SDPA 不输出显式 attention_weight 


        return attn_output, attn_weight
       



class SegQFormerMaskedAttentionDecoderLayer(GradientCheckpointingLayer):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: SegQFormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = SegFormerHeteroCrossAttention(query_size=config.num_queries, embed_dim=self.embed_dim, num_heads=config.num_attention_heads, rank=None, dropout=config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: torch.Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        hidden_states: torch.Tensor,
        level_index: int | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        query_position_embeddings: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward_pre(
        self,
        hidden_states: torch.Tensor,
        level_index: int | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        query_position_embeddings: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Self Attention Block
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        query_position_embeddings: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        if self.pre_norm:
            outputs = self.forward_pre(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
        else:
            outputs = self.forward_post(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

        return outputs





class SegQFormerForSegmentation(Mask2FormerPreTrainedModel):
    config: SegQFormerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'
    input_modalities = ('image',)

    def __init__(self, config: SegQFormerConfig):
        super().__init__(config)
        self.model = SegQFormerModel(config)
        self.weight_dict: dict[str, float] = {
            "loss_ce": config.cross_entropy_weight,
            "loss_dice": config.dice_weight,
            "loss_bce": config.bce_weight,
            "loss_water_dice": config.water_dice_weight,
        }
        
        self.criterion = SegQFormerLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()


    def get_loss_dict(
        self,
        masks_queries_logits: torch.Tensor,
        multi_mask_labels: torch.Tensor,
        binary_masks_labels: torch.Tensor,
        auxiliary_predictions: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_dict: dict[str, torch.Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            multi_mask_labels=multi_mask_labels,
            binary_masks_labels=binary_masks_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss_dict[loss_key] = loss * weight

        return loss_dict
    
    def get_loss(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(loss_dict.values())).sum()
    
    def get_auxiliary_logits(self, output_masks: torch.Tensor):
        auxiliary_logits: list[dict[str, torch.Tensor]] = []

        for aux_binary_masks in (output_masks[:-1]):
            auxiliary_logits.append({"masks_queries_logits": aux_binary_masks})

        return auxiliary_logits


    


    def forward(
        self,
        pixel_values: torch.Tensor,
        multi_mask_labels: torch.Tensor | None = None,
        binary_masks_labels: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_auxiliary_logits: bool | None = None,
        output_attentions: bool | None = None,
        output_loss_dict: bool | None = True,
        return_dict: bool | None = None,
        **kwargs,
    ) -> SegQFormerForSegmentationOutput | tuple[torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_logits = None, None, None

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(masks_queries_logits)

        loss_dict = None
        if multi_mask_labels is not None and binary_masks_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                multi_mask_labels=multi_mask_labels,
                binary_masks_labels=binary_masks_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        if output_loss_dict and loss_dict is not None:
            loss_dict = {key: value.cpu().detach() for key, value in loss_dict.items() if key in self.weight_dict}
        else:
            loss_dict = None

        output = SegQFormerForSegmentationOutput(
            loss=loss,
            loss_dict=loss_dict,
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )
        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss,) + output
        return output
        




        
 
    
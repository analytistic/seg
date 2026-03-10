# from transformers import Mask2FormerModel, AutoImageProcessor, Mask2FormerForUniversalSegmentation
# from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
# import torch
# from PIL import Image
# import requests
# # pixeldecoder = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
# #     embed_dim=256,
# #     num_heads=4,
# #     n_levels=4,
# #     n_points=4,
# # )
# # input_ids = torch.rand(4, 625, 256)
# # output = pixeldecoder.forward(
# #     hidden_states=input_ids,
# #     encoder_hidden_states=input_ids,
# #     spatial_shapes=torch.tensor([[25, 25]]),
# # )

# # processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
# # model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)
# # inputs = processor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
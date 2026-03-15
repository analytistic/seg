from src.model import SegQFormerForSegmentation, SegQFormerImageProcessor
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image


def infer(data_path, output_dir, processor: SegQFormerImageProcessor, model: SegQFormerForSegmentation):
    data_dict = []
    p_list = list(Path(data_path).rglob('*'))
    data_dir = Path(data_path)
    output_dir = Path(output_dir)

    for p in tqdm(p_list, total=len(p_list)):
        if p.is_file() and p.name.endswith(('.jpg', '.png')):

            image_path = data_dir.joinpath(p.with_suffix('.png').name)
            image = Image.open(image_path).convert('RGB')
            target_sizes = [(512, 512)]
            inputs = processor(images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.decode(outputs, target_sizes=target_sizes)
            mask = results[0]["Water"] * 255
            save_path = output_dir.joinpath(p.with_suffix('.png').name)
            Image.fromarray(mask.astype('uint8')).save(save_path) 
   
    return None

if __name__ == "__main__":
    data_path = "data/dataset-preliminary round/Test/Images"
    output_dir = "results/Predicted_masks"
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)
    processor = SegQFormerImageProcessor.from_pretrained("src/model/SegQFormer")
    model = SegQFormerForSegmentation.from_pretrained("src/model/SegQFormer", ignore_mismatched_sizes=True, device_map='auto', dtype='auto')
    infer(data_path, output_dir, processor, model)


            

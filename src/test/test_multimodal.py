from torch.utils.data import DataLoader
from PIL import Image

import sys
sys.path.extend(["../", "../../"])

from src.dataset.multimodaldataset import ImageCaptionDataset
from src.utils.data_cls import VisualConfig, LLMConfig, GenerateConfig, DiffusionConfig, multiModalReturn
from src.multi_modal.multi_modal import MultiModal
from src.train_utils.data_utils import DataCollatorForMultiModalModeling
from src.multi_modal.multi_modal import MultiModal

def main():
    model_path = "/home/liufh/project/data2/liu_project/huggingface_model/qwen/Qwen-7B-Chat"
    diffusion_config = diffusion_config = DiffusionConfig(base_model_name_or_path = "/home/liufh/project/data2/liu_project/huggingface_model/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9", extr_unet_model_name_or_path="/data2/liu_project/huggingface_model/SD/dreamshaper_8.safetensors", is_diffusers=False, config_path="/home/liufh/project/data2/liu_project/huggingface_model/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9", sampleTimeStep=30)
    LLM_config = LLMConfig(model_path=model_path)
    model = MultiModal(LLM_config, GenerateConfig(), VisualConfig(model_path="/home/liufh/project/data2/liu_project/huggingface_model/clip-vit-large-patch14"), diffusion_config, True).to("cuda")
    model.eval()

    model.load_model("/home/liufh/project/data2/liu_project/code/LLM_T2I/weights/train_V1_10/checkpoint-45000/")

    img = Image.open("../../data/img/22.png").resize((224, 224))

    test_message = [{"role": "user", "content": "描述一下这张图片<|extra_0|>"}]
    img_message = [{"role": "user", "content": [img]}]

    text, image = model.generate(test_message, img_message)
    # image = Image.fromarray(image)
    if image is not None:
        image.save("test.png")
    print(text)
        

if __name__ == "__main__":
    main()


from torch.utils.data import DataLoader

import sys
sys.path.extend(["../", "../../"])

from src.dataset.multimodaldataset import ImageCaptionDataset
from src.utils.data_cls import VisualConfig, LLMConfig, GenerateConfig, DiffusionConfig, multiModalReturn
from src.multi_modal.multi_modal import MultiModal
from src.train_utils.data_utils import DataCollatorForMultiModalModeling

def main():
    model_path = "F:/huggingface_model/qwen/Qwen-7B-chat"
    diffusion_config = DiffusionConfig(base_model_name_or_path = "F:/huggingface_model/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9")
    model = MultiModal(LLMConfig(model_path=model_path), GenerateConfig(), VisualConfig(model_path="F:/huggingface_model/clip-vit-large-patch14"), diffusion_config).to("cuda")

    dataset = ImageCaptionDataset(tokenizer=model.LLM_model.tokenizer, data_path="D:/code/LLM_T2I/data/dataset/caption_data.json", multimodal_image_get_feature=model.get_image_feature)
    # print(dataset[0])
    collate_fn = DataCollatorForMultiModalModeling(tokenizer=model.LLM_model.tokenizer).data_collator_quary
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break

if __name__ == "__main__":
    main()


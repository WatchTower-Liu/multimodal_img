from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from functools import partial

import sys
sys.path.extend(["../", "../../"])

from src.dataset.multimodaldataset import ImageCaptionDataset
from src.utils.data_cls import VisualConfig, LLMConfig, GenerateConfig, DiffusionConfig, TrainingArguments, TextFinetuneArguments, UnetFinetuneArguments
from src.multi_modal.multi_modal import MultiModal
from src.train_utils.data_utils import DataCollatorForMultiModalModeling
from src.train_utils.trainer import ModifiedTrainer

def main():
    finetune_args, unetfinetune_args, training_args = HfArgumentParser(
        (TextFinetuneArguments, UnetFinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    print(training_args)
    model_path = "/home/liufh/project/data2/liu_project/huggingface_model/qwen/Qwen-7B-Chat"
    diffusion_config = DiffusionConfig(base_model_name_or_path = "/home/liufh/project/data2/liu_project/huggingface_model/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9")
    model = MultiModal(LLMConfig(model_path=model_path), GenerateConfig(), VisualConfig(model_path="/home/liufh/project/data2/liu_project/huggingface_model/clip-vit-large-patch14"), diffusion_config, True).to("cuda")
    model.LLM_model.make_lora(finetune_args)
    print(model)

    model.train()
    model.LLM_model.model.config.use_cache = False
    dataset = ImageCaptionDataset(tokenizer=model.LLM_model.tokenizer, data_path="/home/liufh/project/data2/liu_project/code/LLM_T2I/data/dataset/caption_data_V4.json", multimodal_image_get_feature=model.get_image_feature, diffusion_config=diffusion_config)
    # print(dataset[0])
    collate_fn = DataCollatorForMultiModalModeling(tokenizer=model.LLM_model.tokenizer, diffusin_config=diffusion_config).data_collator_quary
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    trainer = ModifiedTrainer(model=model,
                                data_collator=collate_fn,
                                train_dataset=dataset,
                                args=training_args)
    trainer.train()
        

if __name__ == "__main__":
    main()


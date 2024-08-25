import torch
from typing import Optional, Union, List, Tuple, Dict
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers.generation import GenerationConfig
from transformers import AutoConfig
import sys
sys.path.extend(["../", "../../"])

from src.LLM.qwen.modeling_qwen import QWenLMHeadModel
from src.LLM.qwen.tokenization_qwen import QWenTokenizer
from src.utils.data_cls import VisualConfig, LLMConfig, GenerateConfig, TextFinetuneArguments, DiffusionConfig

from train_utils.loss_func import LLM_loss_func

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

class LLModel(nn.Module):
    def __init__(self, LLM_config:LLMConfig, gen_config: GenerateConfig, diffusion_config:DiffusionConfig, train: bool):
        super().__init__()
        self.model_name_or_path = LLM_config.model_path
        self.cache_dir = LLM_config.cache_dir
        self.diffusion_config = diffusion_config

        self.load_model()

        if train:
            self.model.gradient_checkpointing_enable() 
            self.model.enable_input_require_grads()
            # self.model.lm_head = CastOutputToFloat(self.model.lm_head)

        self.gen_config.update(**gen_config.to_dict())

    def inject_lora(self, lora_path: str):
        self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=True)

    def make_lora(self, finetune_args: TextFinetuneArguments):
        peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetune_args.Textlora_rank,
                lora_alpha=finetune_args.Textlora_alpha,
                lora_dropout=finetune_args.Textlora_dropout,
                target_modules = finetune_args.Texttarget_modules.split('|')
            )

        self.model = get_peft_model(self.model, peft_config)



    def special_tag_tokenizer(self, text: str):
        ids = self.tokenizer.encode(text)[0]
        return ids

    def load_model(self):
        self.tokenizer = QWenTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        img_tags_start_ids = self.tokenizer.encode(self.diffusion_config.img_tags[0])[0]
        other_generate_config={"gen_img_tag_ids": [img_tags_start_ids]}

        self.model = QWenLMHeadModel.from_pretrained(self.model_name_or_path, other_generate_config, cache_dir=self.cache_dir)
        self.gen_config = GenerationConfig.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)

    def generate(self, input_feature: torch.Tensor, **kwargs):
        output = self.model.generate(
            inputs_embeds=input_feature,
            generation_config=self.gen_config)

        logits = output.hidden_states[0]
        # logits = torch.cat([output.hidden_states[0][:, -1:], *logits], dim=1)
        # print(output.sequences[0])
        # response = self.tokenizer.decode(output.sequences[0], skip_special_tokens=False)
        response = output.sequences

        return response, logits
    
    def forward(self, input_feature: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        if input_feature.device != self.model.device:
            input_feature = input_feature.to(self.model.device)
        output = self.model(inputs_embeds=input_feature, labels = labels, **kwargs)
        
        return output

def main():
    model_path = "F:/huggingface_model/qwen/Qwen-7B-chat"
    model = LLModel(LLMConfig(model_path=model_path), GenerateConfig()).to("cuda")
    messages = [{"role": "user", "content": "你好"}]
    # response = model.generate(messages, max_length=100)
    # print(response)
    test_input_ids = torch.rand(1, 20, 4096).to(torch.bfloat16)
    model(test_input_ids)

if __name__ == "__main__":
    main()


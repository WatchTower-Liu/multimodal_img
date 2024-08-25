from dataclasses import dataclass, field
from dataclasses import asdict
from typing import List, Optional
import torch
from transformers.modeling_outputs import (
    CausalLMOutputWithPast
)
import transformers

@dataclass
class TextFinetuneArguments:
    Textmodel_path: str = field(default=" ")
    Textlora_rank: int = field(default=8)
    Textlora_alpha: int = field(default=32)
    Textlora_dropout: float = field(default=0.1)
    Textprevious_lora_weights: str = field(default=None) 
    Texttarget_modules: str = field(default="W_pack") 

@dataclass
class UnetFinetuneArguments:
    Unetmodel_path: str = field(default=" ")
    Unetlora_rank: int = field(default=8)
    Unetlora_alpha: int = field(default=32)
    Unetlora_dropout: float = field(default=0.1)
    Unetprevious_lora_weights: str = field(default=None) 
    Unettarget_modules: str = field(default="to_k|to_q|to_v|to_out.0") 

@dataclass
class VisualConfig:
    visual_hidden_project_dim: int = field(default=4096)
    img_feature_dim: int = field(default=1024)
    torch_dtype: torch.dtype = field(default=torch.bfloat16)
    model_path: str = field(default="")
    cache_dir: str = field(default=None)

@dataclass
class DiffusionConfig:
    img_tags: List[str] = field(default_factory=lambda: ["<|extra_2|>"]) 
    base_model_name_or_path: str = field(default="runwayml/stable-diffusion-v1-5")
    extr_unet_model_name_or_path: str = field(default="runwayml/stable-diffusion-v1-5")
    sampleTimeStep: int = field(default=1000)
    trainTimeStep: int = field(default=1000)
    is_diffusers: bool = field(default=True)
    only_local_files: bool = field(default=True)
    visual_hidden_project_dim: int = field(default=4096)
    text_hidden_length: int = field(default=77)
    cond_feature_dim: int = field(default=768)
    cache_dir:str = field(default=None)
    DEFAULT_MODEL: str = field(default="runwayml/stable-diffusion-v1-5")
    gen_size:List[int] = field(default_factory=lambda: [512, 512])
    config_path: str = field(default="")

@dataclass
class LLMConfig:
    model_path: str = field(default="")
    special_token: str = field(default="<|extra_0|>")
    cache_dir: str = field(default=None)

@dataclass
class GenerateConfig: 
    return_dict_in_generate: bool = field(default=True)
    output_logits: bool = field(default=True)
    output_hidden_states: bool = field(default=True)
    max_length: int = field(default=1024)

    def to_dict(self):
        return asdict(self)

@dataclass
class DiffusionReturn:
    loss: torch.Tensor = field(default=None)
    noise_latten: torch.Tensor = field(default=None)
    noise_pred: torch.Tensor = field(default=None)

@dataclass
class multiModalReturn:
    loss: torch.Tensor = field(default=None)
    loss1: torch.Tensor = field(default=None)
    loss2: torch.Tensor = field(default=None)
    clip_loss: torch.Tensor = field(default=None)
    LLM_output: CausalLMOutputWithPast = field(default=None)
    diffusion_output: DiffusionReturn = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    feature_proj_lr: Optional[float] = None
    train_diffusion: bool = field(default=False)

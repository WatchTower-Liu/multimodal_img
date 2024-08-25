import transformers
from transformers import Trainer
import os
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.loaders import LoraLoaderMixin
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
import torch
from torch import nn


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
    

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            images=inputs["images"],
            text_label=inputs["text_label"],
            image_label=inputs["image_label"],
            input_text=inputs["input_text"],
            image_label_mask=inputs["image_label_mask"]
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # vision_params = self.model.visualModel.state_dict()
        # torch.save(vision_params, os.path.join(output_dir, "vision_model.bin"))

        saved_params_input_feature_proj = self.model.input_feature_proj.state_dict()     # input feature projection weight
        torch.save(saved_params_input_feature_proj, os.path.join(output_dir, "input_feature_proj.bin"))

        saved_params_LLM = get_peft_model_state_dict(self.model.LLM_model.model)    # LLM lora weight
        torch.save(saved_params_LLM, os.path.join(output_dir, "adapter_model.bin"))

        saved_params_output_feature_proj = self.model.output_feature_proj.state_dict()     # output feature projection weight
        torch.save(saved_params_output_feature_proj, os.path.join(output_dir, "output_feature_proj.bin"))

        saved_params_attention_pooler = self.model.attention_pooler.state_dict()     # attention pooler weight
        torch.save(saved_params_attention_pooler, os.path.join(output_dir, "attention_pooler.bin"))

        config = self.model.LLM_model.model.peft_config
        selected_adapters = list(config.keys())
        print(selected_adapters)
        config[selected_adapters[0]].save_pretrained(output_dir, auto_mapping_dict=None)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        def remove_module(name):
            if "vae" in name:
                return False
            if "text_encoder" in name:
                return False
            if "diffusion_model" in name:
                return False
            return True

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            print([n for n, p in opt_model.named_parameters() if (p.requires_grad and remove_module(n))])
            if self.args.feature_proj_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "input_feature_proj" in name 
                                        or "output_feature_proj" in name 
                                        or "clip_feature_proj" in name 
                                        or "attention_pooler" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad and remove_module(n))
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad and remove_module(n))
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad and remove_module(n))
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.feature_proj_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad and remove_module(n))
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.feature_proj_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and remove_module(n))
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and remove_module(n))
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            # print(optimizer_grouped_parameters)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


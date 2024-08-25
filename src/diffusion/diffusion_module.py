import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer, CLIPTextModel
import accelerate
import os
from safetensors.torch import load_file
from peft import LoraConfig
from diffusers.loaders import LoraLoaderMixin
import sys
sys.path.extend(["../", "../../"])

from src.utils.data_cls import DiffusionReturn, DiffusionConfig, UnetFinetuneArguments
from src.utils.utils import convert_ldm_unet_checkpoint

class DiffusionModel(torch.nn.Module):
    def __init__(self,
                 diffusion_config: DiffusionConfig):
        super().__init__()
        self.base_model_name_or_path = diffusion_config.base_model_name_or_path
        self.sampleTimeStep = diffusion_config.sampleTimeStep
        self.trainTimeStep = diffusion_config.trainTimeStep
        self.is_diffusers = diffusion_config.is_diffusers
        self.only_local_files = diffusion_config.only_local_files
        self.cache_dir = diffusion_config.cache_dir
        self.DEFAULT_MODEL = diffusion_config.DEFAULT_MODEL
        self.config_path = diffusion_config.config_path
        self.extr_unet_model_name_or_path = diffusion_config.extr_unet_model_name_or_path

        self.load_model()
        self.getLatent_model()
        self.load_scheduler()
        self.load_text_encoder()

        self.ddim.set_timesteps(self.sampleTimeStep)

        self.image_processor = VaeImageProcessor()

        self.allTimestep = self.ddim.timesteps

    def inject_lora(self, lora_path: str):
        if os.path.exists(os.path.join(lora_path, "unet_lora_weights.safetensors")):
            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(os.path.join(lora_path, "unet_lora_weights.safetensors"))
            LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=self.unet)

    def make_lora(self, finetune_args: UnetFinetuneArguments):
        unet_lora_config = LoraConfig(
                r=finetune_args.Unetlora_rank,
                init_lora_weights="gaussian",
                lora_alpha=finetune_args.Unetlora_alpha,
                lora_dropout=finetune_args.Unetlora_dropout,
                target_modules = finetune_args.Unettarget_modules.split('|')
            )

        self.unet.add_adapter(unet_lora_config)

    
    def forward(self, image: torch.Tensor, image_label_mask:torch.Tensor, time_step: int, prompt_embeds:torch.Tensor, prompt_test:str):

        sample_image = self.getImgLatent(image)
        noise = torch.randn(sample_image.shape, dtype = prompt_embeds.dtype).to(self.unet.device)
        noisy_image = self.addNoise(sample_image, noise, time_step)
        # latent_model_input = self.ddim.scale_model_input(noisy_image, time_step)

        noise_pred = self.unet(noisy_image, time_step, encoder_hidden_states = prompt_embeds).sample
        #CLIP_embeds = self.encode_prompt(prompt_test)

        loss = F.mse_loss(noise_pred, noise, reduce=False) * image_label_mask# + 0.3*F.mse_loss(prompt_embeds, CLIP_embeds)
        loss = torch.mean(loss)

        return DiffusionReturn(loss=loss, noise_latten=noisy_image, noise_pred=noise_pred)

    def get_train_time_step(self):
        return self.ddim.config.num_train_timesteps

    @torch.no_grad()
    def sample(self, latent:torch.Tensor, prompt_embeds:torch.Tensor, guidance_scale:int=6):
        # print(prompt_embeds.shape)
        # print(latent.shape)
        for Tin in tqdm(range(len(self.allTimestep))):
            Ti = self.allTimestep[Tin]
            latent_model_input = torch.cat([latent] * 2)
            # latent_model_input = latent
            # print(latent)
            latent_model_input = self.ddim.scale_model_input(latent_model_input, Ti)
            noise_pred = self.unet(latent_model_input, Ti, encoder_hidden_states = prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # # print(noise)
            latent = self.sample_step(latent, noise_pred, Ti)
        return latent
    
    def sample_step(self, latent: torch.Tensor, niose: torch.Tensor, timestep: torch.Tensor):
        return self.ddim.step(niose, timestep, latent)['prev_sample']
    
    def addNoise(self, latent:torch.Tensor, noise: torch.Tensor, timestep:torch.Tensor):
        return self.ddim.add_noise(latent, noise, timestep)
    
    def getImgLatent(self, img:torch.Tensor):
        # img = self.image_processor.preprocess(img)
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
    
    def getLatent_model(self):
        MN = self.base_model_name_or_path
        self.vae = AutoencoderKL.from_pretrained(MN, 
                                                 local_files_only = self.only_local_files,
                                                 torch_dtype=torch.bfloat16,
                                                #  use_safetensors=True,
                                                 subfolder = "vae",
                                                 cache_dir = self.cache_dir)

    def load_scheduler(self):
        MN = self.base_model_name_or_path
        self.ddim = DDPMScheduler.from_pretrained(MN, 
                                                  subfolder="scheduler", 
                                                  local_files_only=self.only_local_files, 
                                                  torch_dtype=torch.bfloat16, 
                                                #   use_safetensors=True, 
                                                  cache_dir = self.cache_dir)
        

        
    
    def load_model(self):
        if self.is_diffusers:
            self.unet = UNet2DConditionModel.from_pretrained(self.base_model_name_or_path, 
                                                             local_files_only = self.only_local_files, 
                                                             torch_dtype=torch.bfloat16, 
                                                            #  use_safetensors=True, 
                                                             subfolder = "unet",
                                                             cache_dir = self.cache_dir)
        else:
            state_dict = load_file(self.extr_unet_model_name_or_path)  # 加载ldm参数

            unet_config = UNet2DConditionModel.load_config(self.config_path, 
                                                local_files_only = self.only_local_files, 
                                                torch_dtype=torch.bfloat16, 
                                                # use_safetensors=True, 
                                                subfolder = "unet",
                                                cache_dir = self.cache_dir)
            converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)  # 转换ldm参数到diffusers

            self.unet = UNet2DConditionModel(**unet_config).to(torch.bfloat16)
            self.unet.load_state_dict(converted_unet_checkpoint)

        
        self.unet.enable_xformers_memory_efficient_attention()

    def load_text_encoder(self):
        MN = self.base_model_name_or_path
        self.text_encoder = CLIPTextModel.from_pretrained(MN, 
                                                          local_files_only = self.only_local_files,
                                                          torch_dtype=torch.bfloat16,
                                                        #   use_safetensors=True,
                                                          subfolder = "text_encoder",
                                                          cache_dir = self.cache_dir).cuda()
    
        self.tokenizer = CLIPTokenizer.from_pretrained(MN,
                                                         local_files_only = self.only_local_files,
                                                         subfolder = "tokenizer",
                                                         cache_dir = self.cache_dir)
        
        
    @staticmethod
    def tokenize_prompt(tokenizer, prompt: str):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids 

    
    def encode_prompt(self, prompt:str):
        text_input_ids = self.tokenize_prompt(self.tokenizer, prompt)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )

        encoder_hidden_states = prompt_embeds.hidden_states[-2]
        prompt_embeds = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)

        return prompt_embeds

    # def encode_prompt(self, prompt:str, neg_prompt:str = None):
    #     text_input_ids = self.tokenize_prompt(self.tokenizer, prompt)

    #     prompt_embeds = self.text_encoder(
    #         text_input_ids.to(self.text_encoder.device),
    #         output_hidden_states=True,
    #     )
    #     print(prompt_embeds)
    #     encoder_hidden_states = prompt_embeds.hidden_states[-2]
    #     prompt_embeds = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
    #     # prompt_embeds = prompt_embeds[0]

    #     if neg_prompt is None:
    #         neg_prompt = ""
    #     negative_text_input_ids = self.tokenize_prompt(self.tokenizer, neg_prompt)
    #     negative_prompt_embeds = self.text_encoder(
    #         negative_text_input_ids.to(self.text_encoder.device),
    #         output_hidden_states=True,
    #     )
    #     negative_prompt_embeds = negative_prompt_embeds[0]
    
    #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    #     return prompt_embeds


    def getImg(self, latent:torch.Tensor):
        image = self.vae.decode(latent / self.vae.config.scaling_factor)[0]
        image = image.detach()
        image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])
        return image
    
    
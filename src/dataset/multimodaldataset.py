from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPProcessor
from typing import Callable
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.extend(["../", "../../"])

from src.utils.utils import readJson
from src.utils.data_cls import DiffusionConfig

class ImageCaptionDataset(Dataset):
    end_token="<|endoftext|>"
    def __init__(self, tokenizer: AutoTokenizer, data_path:str, multimodal_image_get_feature: Callable, diffusion_config:DiffusionConfig) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_ids = tokenizer.encode(self.end_token, truncation=True)[0]
        self.data = readJson(data_path)
        
        self.multimodal_image_get_feature = multimodal_image_get_feature


    def __getitem__(self, index):  
        # image, ID = self.data_list[index]
        sample = self.data[index]
        return_dict = {}
        if "img_path" in sample:
            img_path = sample["img_path"]
            image = Image.open(img_path).convert("RGB")       
            image_feature = self.multimodal_image_get_feature(image)
            feature_list = [image_feature]
        else:
            feature_list = []

        if "gen_image_path" in sample:
            img_path = sample["gen_image_path"]
            image = Image.open(img_path).convert("RGB")
            image_size = image.size
            image_numpy = np.array(image)

            base_image = np.zeros((512, 512, 3), dtype=np.uint8)
            center_x = image_size[0] // 2
            if image_size[0]%2==1:
                offset_x = 1
            else:
                offset_x = 0
            center_y = image_size[1] // 2
            if image_size[1]%2==1:
                offset_y = 1
            else:
                offset_y = 0
            base_image[256-center_y: center_y+256+offset_y, 256-center_x:center_x+256+offset_x, :] = image_numpy
            image = Image.fromarray(base_image)
            mask = torch.zeros((4, 64, 64), dtype=torch.bfloat16)
            mask[:, 32-center_y//8: center_y//8+32+offset_y, 32-center_x//8:center_x//8+32+offset_x] = 1
            # gen_image = self.diffusion_img_preprocess(image)
            
            return_dict.update({"image_label" : image, "image_label_mask": mask})

        captions = sample["caption"]
        prompt = sample["query"]
        messages = [{"role": "system", "content": "you are a helpful assistant"}, {"role": "user", "content": prompt}]

        prompt_raw = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
 
        choice_captions = self.tokenizer(prompt_raw)["input_ids"]
        seq_len = len(choice_captions)
        answer = self.tokenizer(captions)["input_ids"]
        choice_captions = choice_captions + answer + [self.pad_token_ids]
        return_dict.update({"images":feature_list, "input_ids": choice_captions, "seq_len": seq_len, "text": captions})
        return return_dict

    def __len__(self):
        return len(self.data)
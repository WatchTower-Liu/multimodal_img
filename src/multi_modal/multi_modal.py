import torch
import transformers
from torch import nn
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
import sys
sys.path.extend(["../", "../../"])

from src.LLM.LLM_module import LLModel
from src.utils.data_cls import VisualConfig, LLMConfig, GenerateConfig, DiffusionConfig, multiModalReturn
from src.diffusion.diffusion_module import DiffusionModel
from src.visual.CLIP_VIT import visualModel
from src.multi_modal.clip_loss import ClipLoss
from src.multi_modal.pool import AttentionalPooler

class MultiModal(nn.Module):
    def __init__(self, LLM_config:LLMConfig, gen_config: GenerateConfig, visual_config: VisualConfig, diffusion_config: DiffusionConfig, train: bool) -> None:
        super().__init__()
        self.LLM_config = LLM_config
        self.visual_config = visual_config
        self.diffusion_config = diffusion_config
        self.make_input_img_feature_project()
        self.make_output_img_feature_project()

        self.LLM_model = LLModel(LLM_config, gen_config, diffusion_config, train)
        self.diffusion_model = DiffusionModel(diffusion_config)
        
        self.visual_model = visualModel(visual_config)

        self.attention_pooler = AttentionalPooler(self.diffusion_config.cond_feature_dim, self.diffusion_config.visual_hidden_project_dim, n_queries=diffusion_config.text_hidden_length).to(self.visual_config.torch_dtype)

        self.clip_loss_func = ClipLoss()

        self.img_tags_start_ids = self.LLM_model.special_tag_tokenizer(diffusion_config.img_tags[0])
        self.LLM_img_token = self.LLM_model.special_tag_tokenizer(LLM_config.special_token)



    def get_image_feature(self, image: Image, return_pooled: bool = False):
        with torch.no_grad():
            image_feature, pooled_feature=self.visual_model.get_image_features(image)
            image_feature = image_feature[0, 1:, :].detach()
        if return_pooled:
            return image_feature, pooled_feature
        return image_feature

    def extract_img_logits(self, gen_ids:torch.Tensor, logits:torch.Tensor, input_text: Optional[list[str]]=None, train:bool = False):
        # assert list(gen_ids.shape) == list(logits.shape)[:2]
        start_ids_index = torch.where(gen_ids == self.img_tags_start_ids)
        # end_ids_index = torch.where(gen_ids == self.img_tags_end_ids)
        batch_indexs = start_ids_index[0]
        if len(batch_indexs) == 0:
            return None, None
        
        # print(gen_ids[batch_indexs, start_ids_index[1]])
        img_conditions = []

        img_conditions = logits[batch_indexs, :]  

        img_conditions = self.attention_pooler(img_conditions)

        img_conditions_proj = self.output_feature_proj(img_conditions)

        return img_conditions_proj, [input_text[idx] for idx in batch_indexs] if input_text is not None else None


    def make_output_img_feature_project(self):
        self.output_feature_proj = nn.Sequential(
            nn.Linear(self.diffusion_config.cond_feature_dim, self.diffusion_config.cond_feature_dim, dtype=self.visual_config.torch_dtype),
            nn.GELU(),
            nn.Linear(self.diffusion_config.cond_feature_dim, self.diffusion_config.cond_feature_dim, dtype=self.visual_config.torch_dtype),
            # nn.LayerNorm(self.diffusion_config.cond_feature_dim, eps=1e-6, dtype=self.visual_config.torch_dtype)
        )


        for name, module in self.output_feature_proj.named_children():
            if "Linear" in module._get_name(): 
                module.weight.data.normal_(mean=0.0, std = 0.01)
                module.bias.data.zero_()


    def make_input_img_feature_project(self):
        self.input_feature_proj = nn.Sequential(
            nn.Linear(self.visual_config.img_feature_dim, self.visual_config.visual_hidden_project_dim, dtype=self.visual_config.torch_dtype),
            nn.GELU(),
            nn.Linear(self.visual_config.visual_hidden_project_dim, self.visual_config.visual_hidden_project_dim, dtype=self.visual_config.torch_dtype)
        )
    
        for name, module in self.input_feature_proj.named_children():
            if "Linear" in module._get_name(): 
                module.weight.data.normal_(mean=0.0, std = 0.01)
                module.bias.data.zero_()

    def pre_make_embedding(self, index: List[int], input_ids: torch.Tensor, images: List[torch.Tensor]):
        embedding = self.LLM_model.model.transformer.wte(input_ids.unsqueeze(0).to(self.LLM_model.model.device)).squeeze(0)

        offset = 0
        split_embedding = []
        for idx in index:
            split_embedding.append(embedding[:idx-offset])
            embedding = embedding[idx+1-offset:]
            offset=idx+1

        split_embedding.append(embedding)

        cat_embedding = split_embedding[0]

        for idx in range(len(split_embedding)-1):
            images_proj = self.input_feature_proj(images[idx].to(self.LLM_model.model.device))
            cat_embedding = torch.cat([cat_embedding, images_proj, split_embedding[idx+1]], dim=0)

        return cat_embedding
    
    def process_generate_message(self, messages: List[Dict], image_message: List[Dict]):
        assert len(messages) == len(image_message), "messages and image_message should have same length"
        message_feature = []
        for idx in range(len(messages)):
            if messages[idx]["role"] == "system":
                text = self.LLM_model.tokenizer.apply_chat_template([messages[idx]], tokenize=False, add_generation_prompt=False)
                input_ids = torch.tensor(self.LLM_model.tokenizer.encode(text))
                processed_feature = self.LLM_model.model.transformer.wte(input_ids.to(self.LLM_model.model.device))
            elif messages[idx]["role"] == "user":
                text = self.LLM_model.tokenizer.apply_chat_template([messages[idx]], tokenize=False, add_generation_prompt=True)
                print(text)
                input_ids = torch.tensor(self.LLM_model.tokenizer.encode(text))
                print(input_ids)
                img_tag_index = torch.where(input_ids == self.LLM_img_token)[0]
                if len(img_tag_index) != 0:
                    images = image_message[idx]["content"]
                    image_features = [self.get_image_feature(image) for image in images]
                    processed_feature = self.pre_make_embedding(img_tag_index, input_ids, image_features)
                    processed_feature = processed_feature
                else:
                    processed_feature = self.LLM_model.model.transformer.wte(input_ids.to(self.LLM_model.model.device))
            else:
                if self.diffusion_config.img_tags[0] in messages[idx]["content"]:
                    messages[idx]["content"] = messages[idx]["content"] + "这里是生成的图像：" + self.LLM_config.special_token
                    text = self.LLM_model.tokenizer.apply_chat_template([messages[idx]], tokenize=False, add_generation_prompt=False)
                    input_ids = torch.tensor(self.LLM_model.tokenizer.encode(text))
                    fake_gen_img_token_index = torch.where(input_ids == self.LLM_img_token)[0]
                    images = image_message[idx]["content"]
                    image_features = [self.get_image_feature(image) for image in images]
                    processed_feature = self.pre_make_embedding(fake_gen_img_token_index, input_ids, image_features)
                    processed_feature = processed_feature
                else:
                    text = self.LLM_model.tokenizer.apply_chat_template([messages[idx]], tokenize=False, add_generation_prompt=False)
                    input_ids = torch.tensor(self.LLM_model.tokenizer.encode(text))
                    processed_feature = self.LLM_model.model.transformer.wte(input_ids.to(self.LLM_model.model.device))
            message_feature.append(processed_feature)

        message_feature = torch.cat(message_feature, dim=0).unsqueeze(0)
        print(message_feature.shape)

        return message_feature
    
    def generate(self, messages: List[Dict], image_message: List[Dict], **kwargs):
        
        message_feature = self.process_generate_message(messages, image_message)
        response, LLM_logits = self.LLM_model.generate(input_feature=message_feature, **kwargs)
        text = self.LLM_model.tokenizer.decode(response[0], skip_special_tokens=False)

        prompt_latten, _ = self.extract_img_logits(response, LLM_logits)
        image = None
        if prompt_latten is not None:
            img_latten = torch.randn(1, 4, self.diffusion_config.gen_size[0]//8, self.diffusion_config.gen_size[1]//8).to(prompt_latten)
            # prompt_latten = self.diffusion_model.encode_prompt("根据描述绘制图像，这是描述：展示城市广场的现实主义风格照片。在画面前方，广场铺有黄色和棕色相间的石砖，形成了引人注目的图案。广场上散布着几位行人，他们似乎正在享受晴朗天气下的休闲时光。背景中，可以看到几座具有历史特色的建筑，包括一座装饰华丽的教堂和一些现代风格的高楼大厦。天空是清澈的蓝色，几朵白云点缀其间。")
            neg_prompt_latten = self.diffusion_model.encode_prompt("Low quality, distorted, incongruous, wrong faces")
            prompt_latten = torch.cat([neg_prompt_latten, prompt_latten], dim=0)
            img_latten = self.diffusion_model.sample(img_latten, prompt_latten)
            image = self.diffusion_model.getImg(img_latten)[0]

        return text, image
    
    def load_model(self, checkpoint_path: str):
        self.LLM_model.inject_lora(checkpoint_path)
        # self.diffusion_model.inject_lora(checkpoint_path)
        input_project_weight = torch.load(checkpoint_path + "input_feature_proj.bin")
        self.input_feature_proj.load_state_dict(input_project_weight)
        output_project_weight = torch.load(checkpoint_path + "output_feature_proj.bin")
        self.output_feature_proj.load_state_dict(output_project_weight)
        attention_pooler_weight = torch.load(checkpoint_path + "attention_pooler.bin")
        self.attention_pooler.load_state_dict(attention_pooler_weight)
        # embedding_weight = torch.load(checkpoint_path + "embedding_model.bin")
        # self.LLM_model.model.transformer.wte.load_state_dict(embedding_weight)

    def clip_loss(self, image_label: List, conditions: torch.Tensor, **kwargs):
        image_label_feature = []
        for idx in range(len(image_label)):
            img_feature, pooled_feature = self.get_image_feature(image_label[idx], return_pooled=True)
            image_label_feature.append(img_feature)
        image_label_feature = torch.stack(image_label_feature, dim=0).to(conditions)
        # print(image_label_feature.shape)
        image_label_feature = torch.mean(image_label_feature, dim=1)
        conditions_feature = torch.mean(conditions, dim=1)
        # print(conditions_feature.shape)

        loss = self.clip_loss_func(image_label_feature, conditions_feature)
        return loss
    
    def diffusion_img_preprocess(self, image: List):
        feature = []
        for img in image:
            image_ndarray = np.array(img)
            image_ndarray = torch.from_numpy(image_ndarray).permute(2, 0, 1) / 127.5 - 1.0
            feature.append(image_ndarray)
    
        feature = torch.stack(feature, dim=0)

        return feature

    def forward(self, input_ids: torch.Tensor, images: Optional[List[List[torch.Tensor]]] = None, 
                text_label: Optional[torch.Tensor]=None, image_label: Optional[List]=None, input_text:List[str] = None, 
                image_label_mask:Optional[torch.Tensor] = None, **kwargs):
        if images is not None:
            LB, _ = input_ids.shape
            IB = len(images)

            assert LB == IB, "input_ids and images should have same batch size"

            img_tag_index = torch.where(input_ids == self.LLM_img_token)

            batch_index = {idx:{"type": "norm", "index": []} for idx in range(LB)}
            for batch_idx, feature_idx in zip(img_tag_index[0].cpu().numpy(), img_tag_index[1].cpu().numpy()):
                batch_index[batch_idx]["type"] = "img"
                batch_index[batch_idx]["index"].append(feature_idx)
            # print(self.LLM_model.model.transformer.wte)
            processed_feature = []
            # print(batch_index)
            if len(batch_index) != 0:
                for batch_idx in batch_index.keys():
                    if batch_index[batch_idx]["type"] == "img":
                        processed_feature.append(self.pre_make_embedding(batch_index[batch_idx]["index"], input_ids[batch_idx], images[batch_idx])[:len(text_label[batch_idx])])
                    else:
                        processed_feature.append(self.LLM_model.model.transformer.wte(input_ids[batch_idx].to(self.LLM_model.model.device)))
                
                processed_feature = torch.stack(processed_feature, dim=0)
                
            else:
                processed_feature = self.LLM_model.model.transformer.wte(input_ids.to(self.LLM_model.model.device))
        else:
            processed_feature = self.LLM_model.model.transformer.wte(input_ids.to(self.LLM_model.model.device))

        LLM_output = self.LLM_model(input_feature=processed_feature, labels = text_label, **kwargs)

        return_data = multiModalReturn(LLM_output=LLM_output)

        loss = None
        if text_label is not None:
            loss1 = LLM_output.loss
            loss=loss1
            return_data.loss1 = loss1
        

        if image_label is not None and text_label is not None:
            image_label_feature = self.diffusion_img_preprocess(image_label)
            random_time_step = torch.randint(0, self.diffusion_model.get_train_time_step(), (len(image_label),)).to(torch.long).to(self.LLM_model.model.device)
            condition_hidden, prompt_text = self.extract_img_logits(text_label, LLM_output.hidden_states, input_text, train=True)  
            diffusion_output = self.diffusion_model(image_label_feature.to(condition_hidden), image_label_mask.to(condition_hidden), 
                                                    random_time_step, condition_hidden, prompt_text)
            loss2 = diffusion_output.loss
            return_data.diffusion_output = diffusion_output
            return_data.loss2 = loss2
            # print(loss2)
            loss += loss2

            # clip_loss = self.clip_loss(image_label, clip_feature)
            # return_data.clip_loss = clip_loss

            # loss += 0.5*clip_loss

        return_data.loss = loss

        return return_data

def main():
    model_path = "F:/huggingface_model/qwen/Qwen-7B-chat"
    diffusion_config = DiffusionConfig(base_model_name_or_path = "F:/huggingface_model/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9")
    model = MultiModal(LLMConfig(model_path=model_path), GenerateConfig(), VisualConfig(model_path="F:/huggingface_model/clip-vit-large-patch14"), diffusion_config).to("cuda")
    messages = [{"role": "user", "content": "你好"}]
    # response = model.generate(messages, max_length=100)
    # print(response)
    test_input_ids = torch.tensor([[121,12314,35,25,34,52,35,34245,63,63,151646,6,36,346,456,151646,352,6,7,8,7,5,87,46,34635,35,3,151646]])
    # print(test_input_ids.shape)
    test_images = torch.rand(3, 1024).to(torch.bfloat16)
    test_label_ids = torch.tensor([[-100,-100,-100,-100,-100,-100,-100,-100,35,25,34,52,35,34245,63,63,151646,6,36,346,456,151648,-100,151649,7,8,7,5,87,46,34635,35,3,7]]).to("cuda")

    test_image_label = torch.rand(1, 3, 512, 512).to(torch.bfloat16).to("cuda")
    model(test_input_ids, [[test_images, test_images, test_images]], test_label_ids, test_image_label)

if __name__ == "__main__":
    main()

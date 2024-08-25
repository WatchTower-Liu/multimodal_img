import transformers
import torch
from typing import Optional
from torch import nn
from PIL import Image
from transformers import CLIPModel, CLIPConfig, CLIPProcessor
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.clip.modeling_clip import CLIP_VISION_INPUTS_DOCSTRING
import sys
sys.path.extend(["../", "../../"])

from src.utils.data_cls import VisualConfig

class VitCLIPModel(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden = vision_outputs.last_hidden_state  
        pooled_output = vision_outputs.pooler_output  # pooled_output
        pooled_image_features = self.visual_projection(pooled_output)
        # print(pooled_output.shape)
        return hidden, pooled_image_features

class visualModel:
    def __init__(self, visual_config: VisualConfig) -> None:
        self.visual_config = visual_config
        self.CLIP_model = VitCLIPModel.from_pretrained(visual_config.model_path, cache_dir=visual_config.cache_dir)
        self.processor = CLIPProcessor.from_pretrained(visual_config.model_path, cache_dir=visual_config.cache_dir)

    def get_image_features(self, image: Image):
        P_input = self.processor(images=image, return_tensors="pt")
        hidden, pooled_image_features = self.CLIP_model.get_image_features(**P_input)
        return hidden.to(self.visual_config.torch_dtype), pooled_image_features.to(self.visual_config.torch_dtype)

def main():
    model_path = "F:/huggingface_model/clip-vit-large-patch14"
    visual_config = VisualConfig(model_path=model_path)
    model = visualModel(visual_config)
    test_img = Image.open("D:/code/LLM_T2I/data/img/184.png")
    print(model.get_image_features(test_img).shape)

if __name__ == "__main__":
    main()

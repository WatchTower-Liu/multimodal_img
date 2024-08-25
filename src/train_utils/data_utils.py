import numpy as np
import torch
from copy import deepcopy
import sys
sys.path.extend(["../", "../../"])

from src.utils.data_cls import DiffusionConfig

class DataCollatorForMultiModalModeling():
    pad_token="<|endoftext|>"
    def __init__(self, tokenizer, diffusin_config:DiffusionConfig) -> None:
        self.diffusin_config = diffusin_config
        self.pad_token_ids = tokenizer.encode(self.pad_token, truncation=True)[0]
        self.gen_special_token = tokenizer.encode(diffusin_config.img_tags[0], truncation=True)[0]

    def data_collator_quary(self, features: list) -> dict:
        len_ids = [len(feature["input_ids"]) for feature in features]
        image_number = [sum([len(feature["images"][i]) for i in range(len(feature["images"]))]) for feature in features]
        longest_ids = max(len_ids)
        largest_image_number = max(image_number)
        total_max_length = longest_ids + largest_image_number  # Maximum length with redundancy
        input_ids = []
        input_images = []
        text_label = []
        image_label = []
        image_label_mask = []
        input_text = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            images = feature["images"]
            input_text.append(feature["text"])
            masked_ids = deepcopy(ids)
            input_images.append(images)
            if "image_label" in feature:
                image_label.append(feature["image_label"])
                image_label_mask.append(feature["image_label_mask"])
            seq_len = feature["seq_len"] # prompt length
            prefill_len = seq_len - len(images) + sum([len(i) for i in images])

            labels = (
                [-100] * (prefill_len) + masked_ids[seq_len :] + [-100] * (total_max_length - (prefill_len + len(ids) - seq_len))
            )

            ids = ids + [self.pad_token_ids] * (total_max_length - len(ids))
            
            _ids = torch.LongTensor(ids)
            text_label.append(torch.LongTensor(labels))
            input_ids.append(_ids)
        input_ids = torch.stack(input_ids)
        text_label = torch.stack(text_label)
        image_label = image_label if len(image_label) > 0 else None
        image_label_mask = torch.stack(image_label_mask) if len(image_label_mask) > 0 else None
        return {
            "input_ids": input_ids,
            "text_label": text_label,
            "images": input_images,
            "image_label": image_label,
            "image_label_mask": image_label_mask,
            "input_text": input_text
        }
    
def main():
    prompt_part_ids = []
    np.random.choice(len(prompt_part_ids), max(min(2, len(prompt_part_ids)-6), 1), replace=False)

if __name__ == "__main__":
    main()


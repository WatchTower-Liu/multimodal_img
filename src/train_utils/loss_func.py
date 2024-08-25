import torch
from typing import List
from torch.nn import CrossEntropyLoss, MSELoss

def LLM_loss_func(shift_logits: torch.Tensor, shift_labels: torch.Tensor, img_hidden_feature: torch.Tensor, hideen_pos:List[List[int]]):

    assert shift_logits.size() == shift_labels.size(), "shift_logits and shift_labels should have same size"
    assert len(hideen_pos) == shift_logits.size(0), "hideen_pos and shift_logits should have same batch size"

    loss_fct_text = CrossEntropyLoss()
    loss_text = loss_fct_text(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    
    loss_fct_img = MSELoss()
    loss_img = 0
    for i in range(len(hideen_pos)):
        range_sert = torch.tensor(hideen_pos[i][0]).to(shift_logits.device)
        range_end = torch.tensor(hideen_pos[i][1]).to(shift_logits.device)
        loss_img += loss_fct_img(img_hidden_feature[i], shift_logits[i][range_sert:range_end])
        
    loss_img = loss_img / len(hideen_pos)

    return loss_text + loss_img
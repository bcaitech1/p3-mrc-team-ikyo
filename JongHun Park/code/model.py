#!/usr/bin/env python
# coding: utf-8


from torch import nn
from transformers import AutoModel, AutoConfig

class Encoder(nn.Module):
    def __init__(self, model_checkpoint):
        super(Encoder, self).__init__()

        config = AutoConfig.from_pretrained(model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint, config=config)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output
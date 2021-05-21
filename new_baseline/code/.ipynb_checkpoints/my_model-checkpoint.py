import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

class Mymodel(nn.Module):
    def __init__(self, model_name, model_config):
        super().__init__() 
        
        self.model_name = model_name
        self.backborn_model = AutoModel.from_pretrained(model_name)
        self.conv1d_layer = nn.Conv1d(model_config.hidden_size, 256, kernel_size=3, padding=1)
        # self.lstm = nn.LSTM(input_size = model_config.hidden_size, hidden_size = model_config.hidden_size, dropout=0.3, bidirectional = True, batch_first = True)
        self.dense_layer = nn.Linear(256, 2, bias=True)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if "xlm" in self.model_name:
            outputs = self.backborn_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
            outputs = self.backborn_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        sequence_output = outputs[0]
        # lstm 
        #print(sequence_output.size())
        conv_input = sequence_output.transpose(1, 2)
        #print(conv_input.size())
        conv_output = self.conv1d_layer(conv_input)
        #print(conv_output.size())
        dense_input = conv_output.transpose(1, 2)
        #print(dense_input.size())
        logits = self.dense_layer(dense_input)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        return {"start_logits" : start_logits, "end_logits" : end_logits, "hidden_states" :  outputs.hidden_states, "attentions" : outputs.attentions} 
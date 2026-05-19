import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class BertBiLSTMCRF(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim=256, dropout=0.1):
        super(BertBiLSTMCRF, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Linear layer to project LSTM output to the number of labels
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # shape: [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # BiLSTM layer
        lstm_out, _ = self.lstm(sequence_output)
        
        # Project LSTM output to tag space
        emissions = self.hidden2tag(lstm_out)  # shape: [batch_size, seq_len, num_labels]

        # If labels are provided, compute the loss
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte())
            return loss, emissions  # 返回loss和emissions

        # Otherwise, return only the emissions (to match the Trainer's expectations)
        else:
            return emissions, None  # 返回emissions和None
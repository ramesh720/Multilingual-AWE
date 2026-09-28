import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
from torch.nn.utils.rnn import pack_padded_sequence

class NAW_LSTM_acoustic_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=2, bidirectional=True, dropout=0.2):
        super(NAW_LSTM_acoustic_model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, embedding_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed)
        # Use the top‑layer forward/backward hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        embedding = self.fc(h_cat)
        return F.normalize(embedding, dim=-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.pe = nn.Parameter(self._generate_positional_encodings(d_model, max_len), requires_grad=False)

    def _generate_positional_encodings(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].to(x.device)
        return x


class NAW_LSTM_text_model(nn.Module):
    def __init__(self, input_dim_text, no_of_tokens, hidden_dim, embedding_dim, num_layers, bidirectional=True, dropout=0.2):
        super(NAW_LSTM_text_model, self).__init__()
        self.no_of_tokens = no_of_tokens
        self.embedding = nn.Embedding(no_of_tokens, input_dim_text, padding_idx=0)
        std = sqrt(2.0 / (no_of_tokens + input_dim_text))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.positional_embedding = PositionalEmbedding(input_dim_text)
        self.lstm = nn.LSTM(input_dim_text, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, embedding_dim)

    def forward(self, inputs, lengths):
        inputs = self.embedding(inputs.to(torch.int64))
        inputs = self.positional_embedding(inputs)
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        embedding = self.fc(h_cat)
        return F.normalize(embedding, dim=-1)


class NAW_LSTM_multi_view_model(nn.Module):
    def __init__(self, input_dim_acoustic, input_dim_text, no_of_tokens, hidden_dim, embedding_dim, num_layers=2, bidirectional=True):
        super(NAW_LSTM_multi_view_model, self).__init__()
        self.lstm_acoustic = NAW_LSTM_acoustic_model(input_dim_acoustic, hidden_dim, embedding_dim,
                                                      num_layers=num_layers, bidirectional=bidirectional)
        self.lstm_text = NAW_LSTM_text_model(input_dim_text, no_of_tokens, hidden_dim, embedding_dim,
                                             num_layers=num_layers, bidirectional=bidirectional)
        # ✅ FIX: Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, c, x_lengths, c_lengths):
        acoustic_word_embedding = self.lstm_acoustic(x, x_lengths)
        text_word_embedding = self.lstm_text(c, c_lengths)
        logits = self.logit_scale.exp() * (text_word_embedding @ acoustic_word_embedding.T)
        return acoustic_word_embedding, text_word_embedding, logits

    def forward_infer(self, x, c, x_lengths, c_lengths):
        acoustic_word_embedding = self.lstm_acoustic(x, x_lengths)
        text_word_embedding = self.lstm_text(c, c_lengths)
        return acoustic_word_embedding, text_word_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time


# Define a function to generate positional encodings
def get_positional_encoding(max_seq_len: int, num_embeddings: int):
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, num_embeddings, 2) * -(np.log(10000.0) / num_embeddings))
    positional_encoding = np.zeros((max_seq_len, num_embeddings))
    positional_encoding[:, 0::2] = np.sin(position * div_term)
    positional_encoding[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(positional_encoding, dtype=torch.float)


def generate_square_subsequent_mask(sz: int):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class DecoderBlock(nn.Module):
    def __init__(self, num_embeddings: int, num_attention_heads: int, hidden_layer_dim: int, dropout_rate: float):
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(num_embeddings, num_attention_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(num_embeddings)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(num_embeddings, hidden_layer_dim)
        self.linear2 = nn.Linear(hidden_layer_dim, num_embeddings)
        self.norm2 = nn.LayerNorm(num_embeddings)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, tgt_mask):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_embeddings, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        pos_enc = torch.zeros(max_len, num_embeddings)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_embeddings, 2).float() * (-math.log(10000.0) / num_embeddings))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pos_enc)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# wrap the above into a decoder-only model
class MultiLayerTransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, num_embeddings: int, num_attention_heads: int, hidden_layer_dim: int,
                dropout_rate: float, num_decoders: int):
        super(MultiLayerTransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, num_embeddings)
        self.pos_encoder = PositionalEncoding(num_embeddings, dropout_rate)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(num_embeddings, num_attention_heads, hidden_layer_dim, dropout_rate)
            for _ in range(num_decoders)
        ])
        self.linear = nn.Linear(num_embeddings, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for transformer_block in self.transformer_blocks:
            tgt_mask = generate_square_subsequent_mask(x.size(0))
            x = transformer_block(x, tgt_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output


vocab_size = 10000
num_embeddings = 2048
num_attention_heads = 1
hidden_layer_dim = 4 * num_embeddings
dropout_rate = 0.1
num_decoders = 10

# Define the vocabulary
vocab = ["of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among"]
vocab_size = len(vocab)
# word-to-index and index-to-word mapping
word2id = {word: id for id, word in enumerate(vocab)}
id2word = {id: word for id, word in enumerate(vocab)}

model = MultiLayerTransformerDecoder(vocab_size, num_embeddings, num_attention_heads, hidden_layer_dim, dropout_rate, num_decoders)
print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# create input data
context_length = 100
batch_size = 1
# input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))
sequence = ["of", "in", "to", "for", "with", "on", "at"][:context_length]
input_tensor = torch.tensor([[word2id[word] for word in sequence]])

# generate sequence of words
n_words = 10
generated_words = []
for i in range(n_words):
    output = model(input_tensor)
     # take the last word in the sequence
    predicted_index = output.argmax(dim=-1)[0, -1]
    predicted_word = id2word[predicted_index.item()]
    print(predicted_word, end=" ")
    generated_words.append(predicted_word)
    # append the predicted word to the input
    input_tensor = torch.cat([input_tensor, predicted_index.unsqueeze(0).unsqueeze(0)], dim=-1)
    time.sleep(0.75)

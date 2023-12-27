import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# define feed forward network
class FeedForward(nn.Module):
    def __init__(self, num_embeddings: int, hidden_layer_dim: int, dropout_rate: float = 0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(num_embeddings, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, num_embeddings)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# define encoder block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_embeddings: int, num_attention_heads: int, hidden_layer_dim: int, dropout_rate: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(num_embeddings, num_attention_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(num_embeddings)
        self.norm2 = nn.LayerNorm(num_embeddings)
        self.feed_forward = FeedForward(num_embeddings, hidden_layer_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Multi-Head Attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed Forward Network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


# wrap into an encoder-only model
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_embeddings: int, max_seq_len: int, num_attention_heads: int, hidden_layer_dim: int, num_encoder_blocks: int, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding = nn.Embedding(max_seq_len, num_embeddings)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(num_embeddings, num_attention_heads, hidden_layer_dim, dropout_rate)
                for _ in range(num_encoder_blocks)
            ]
        )

    def forward(self, x, mask=None):
        seq_length = x.shape[1]
        positions = torch.arange(0, seq_length).expand(x.shape[0], seq_length).to(x.device)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, mask)

        return out


# Assume the following hyperparameters
vocab_size = 10000
num_embeddings = 1024
max_seq_len = 1000
num_attention_heads = 8
hidden_layer_dim = num_embeddings * 4
num_encoder_blocks = 12
dropout = 0.1

# Instantiate the model
model = TransformerEncoder(vocab_size, num_embeddings, max_seq_len, num_attention_heads, hidden_layer_dim, num_encoder_blocks, dropout)

# Generate some example input
input_tensor = torch.randint(0, vocab_size, (1, 20))  # batch size of 1 and sequence length of 20

# Forward pass through the model
output = model(input_tensor, mask=None)

print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

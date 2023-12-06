import torch
import torch.nn as nn
import re
from transformers import BertConfig, BertModel

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class ImageEmbeddingPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 1024

        # Configure a new BERT model with 2 hidden layers and without positional embeddings
        config = BertConfig(
            hidden_size=self.embedding_dim,
            num_hidden_layers=2,  # Set the number of hidden layers to 2
            num_attention_heads=8,
            intermediate_size=self.embedding_dim*2,
            use_position_embeddings=True,
            max_position_embeddings=2304,
            use_bfloat16=True,
            vocab_size=1,
        )
        self.bert = BertModel(config)

    def forward(self, embeddings):
        # embeddings shape: (batch_size, num_images, embedding_dim)
        batch_size, num_tokens, _ = embeddings.shape
        num_views = num_tokens // 576

        # Process embeddings through BERT without positional IDs
        outputs = self.bert(inputs_embeds=embeddings)

        last_hidden_states = outputs['last_hidden_state'].to(embeddings.dtype)
        # split mid dimension into num_views
        last_hidden_states = last_hidden_states.view(batch_size, num_views, 576, self.embedding_dim)

        pooled_output = torch.mean(last_hidden_states, dim=1)  # For mean pooling

        return pooled_output

def build_image_pooler(config):
    return ImageEmbeddingPooler()

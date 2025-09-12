import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.image_processing import position_encoder

class ViT(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, latent_vector_size: int, patch_size: int, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.latent_vector_size = latent_vector_size
        self.patch_size = patch_size

        # The paper does not use a bias term to create the patch embeddings so perhaps set bias to false
        self.patch_linear_proj = nn.Linear((self.patch_size**2) * self.in_channels, latent_vector_size)

        # The original transformer paper uses 6 encoder layers, but the ViT paper does not mention number of encoders
        self.enc = TransformerEncoder(latent_vector_size, int(latent_vector_size / num_heads), int(latent_vector_size / num_heads), num_heads)

        # MLP Head
        self.input = nn.Linear(latent_vector_size, latent_vector_size*2)
        self.hidden1 = nn.Linear(latent_vector_size*2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    # Takes in tensors of shape (B,N,C*P^2) representing batches of flattened patches of an image
    def forward(self, x):
        # TODO: perhaps we take even this out?
        patch_embeddings = self.patch_linear_proj(x) # Project the patches to (B,N,D) where D is latent vector size
        class_token = torch.normal(0., 1., (x.shape[-3], self.latent_vector_size)).to(x.device)
        position_embeddings = torch.stack([position_encoder(patch_embeddings.shape[-2] + 1, self.latent_vector_size)]*x.shape[-3]).to(x.device)

        vit_input = torch.cat((class_token.unsqueeze(dim=1), patch_embeddings), dim=-2) + position_embeddings

        encoded_output = self.enc(vit_input)

        classification_output = self.softmax(self.hidden1(self.input(encoded_output))) # Do we just pass in the class vector (i.e. encoded_output[0])

        return classification_output[:,0]
        

class MultiHeadAttention(nn.Module):
    def __init__(self, latent_vector_size: int, key_dim: int, val_dim: int, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.key_dim = torch.tensor(key_dim)

        # In practice, key_dim = val_dim = latent_vec_size / num_heads
        self.qkv_proj = nn.Linear(latent_vector_size, num_heads * (2*key_dim + val_dim))
        self.msa_proj = nn.Linear(num_heads*val_dim, latent_vector_size)
    
    def forward(self, x):
        qkv = self.qkv_proj(x) # dim (N, 3*num_heads*key_dim)

        mh_key_dim = self.num_heads * self.key_dim
        mh_query = qkv[:, :, :mh_key_dim]
        mh_key = qkv[:, :, mh_key_dim:2*mh_key_dim]
        mh_value = qkv[:, :, 2*mh_key_dim:]

        mh_query_key = torch.matmul(mh_query, torch.transpose(mh_key, 1, 2))
        scaled_mh_query_key = mh_query_key / torch.sqrt(self.key_dim)
        softmax_scaled_mh_query_key = F.softmax(scaled_mh_query_key, dim=-1)
        mh_attention = torch.matmul(softmax_scaled_mh_query_key, mh_value)
        return self.msa_proj(mh_attention)

class TransformerEncoder(nn.Module):
    def __init__(self, latent_vector_size: int, key_dim: int, val_dim: int, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer_norm = nn.LayerNorm(latent_vector_size) # FIXME: need to figure out what dim to use for layer norm
        self.msa = MultiHeadAttention(latent_vector_size, key_dim, val_dim, num_heads)

        # MLP with 2 layers and gelu
        self.hidden1 = nn.Linear(latent_vector_size, int(latent_vector_size / 2)) # TODO: find a better hidden dimenesion
        self.hidden2 = nn.Linear(int(latent_vector_size / 2), latent_vector_size)
        self.gelu = nn.GELU()
        self.output = nn.Softmax(dim=-1)

    # Input here should be dim (num_patches, latent_vector_size)
    def forward(self, x):
        ln1 = self.layer_norm(x)
        mha = self.msa(ln1)
        enc1 = x + mha

        ln2 = self.layer_norm(enc1)
        h1 = self.gelu(self.hidden1(ln2))
        h2 = self.gelu(self.hidden2(h1))
        out = self.output(h2)
        enc2 = enc1 + out

        return enc2
    
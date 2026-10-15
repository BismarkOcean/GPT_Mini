from torch import nn
from .attention import MultiHeadCausalAttention
import torch

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attenion=MultiHeadCausalAttention(embed_dim,num_heads)
        self.mlp=nn.Sequential(
            nn.Linear(embed_dim,4*embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim*4,embed_dim)
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        y= x+self.attenion(self.ln1(x))
        out=y+self.mlp(self.ln2(y))
        return out

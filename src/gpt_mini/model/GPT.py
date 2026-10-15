from torch import nn
import torch
from gpt_mini.model.block import TransformerBlock
class GPT(nn.Module):
    def __init__(self,
                  vocab_size:int,
                  max_seq_len:int,
                  embed_dim:int,
                  num_heads:int,
                  num_layers:int):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.token_embedding=nn.Embedding(vocab_size,embed_dim)
        self.position_embedding=nn.Embedding(max_seq_len,embed_dim)
        self.blocks=nn.ModuleList([
                TransformerBlock(embed_dim,num_heads)
                for _ in range(num_layers)
        ])
        self.fin_ln=nn.LayerNorm(embed_dim)
        self.lm_head=nn.Linear(
            embed_dim,
            vocab_size,
            bias=False)
    def forward(self,ids:torch.Tensor) ->torch.Tensor:
        B,T=ids.shape
        if(T>self.max_seq_len):
            raise ValueError("Sequence length exceeds max_seq_len")
        token_emb=self.token_embedding(ids)
        positons = torch.arange(T,device=ids.device)
        positions_emb = self.position_embedding(positons)
        emb=token_emb+positions_emb
        for block in self.blocks:
            emb=block(emb)
        emb=self.fin_ln(emb)
        logits=self.lm_head(emb)
        return logits

        

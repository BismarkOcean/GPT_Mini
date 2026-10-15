import torch
from torch import nn


class SingleHeadCausalAttention(nn.Module):
    def __init__(self,embed_dim:int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim,embed_dim,bias=False)
        self.k_proj = nn.Linear(embed_dim,embed_dim,bias=False)
        self.v_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        score=(q@k.transpose(-1,-2))/q.size(-1)**0.5
        T = score.size(1)
        mask = torch.triu(torch.ones(T,T,
                                     dtype=torch.bool,
                                     device=x.device),
                         diagonal=1)
        masked_score = score.masked_fill(mask,float("-inf"))
        weight = torch.softmax(masked_score,dim=-1)
        out=weight@v
        return out

class MultiHeadCausalAttention(nn.Module):
    def __init__(self,embed_dim:int,num_heads:int):
        if(embed_dim%num_heads!=0):
            raise ValueError
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        self.q_proj = nn.Linear(embed_dim,embed_dim,bias=False)
        self.k_proj = nn.Linear(embed_dim,embed_dim,bias=False)
        self.v_proj = nn.Linear(embed_dim,embed_dim,bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        q=q.reshape(x.shape[0],x.shape[1],self.num_heads,self.head_dim)
        k=k.reshape(x.shape[0],x.shape[1],self.num_heads,self.head_dim)
        v=v.reshape(x.shape[0],x.shape[1],self.num_heads,self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        score = (q@k.transpose(-1,-2))/self.head_dim**0.5
        T= score.shape[2]
        mask = torch.triu(torch.ones(T,T,
                                     dtype=torch.bool,
                                     device=x.device),
                                     diagonal=1)
        masked_score=score.masked_fill(mask,float("-inf"))
        weight=torch.softmax(masked_score,dim=-1)
        out = weight @ v
        out = out.transpose(1, 2)
        out = out.reshape(x.shape[0], x.shape[1], x.shape[2])
        out = self.out_proj(out)
        return out
from typing import Any

import torch

def make_window(
    token_ids: list[int],
    start: int,
    seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len<=0:
        raise ValueError("seq_len非正")
    if start < 0 or start >= len(token_ids) - seq_len:
        raise IndexError("illegal start")
    x_list=token_ids[start:start+seq_len]
    y_list=token_ids[start+1:start+seq_len+1]
    x=torch.tensor(x_list,dtype=torch.long,device="cpu")
    y=torch.tensor(y_list,dtype=torch.long,device="cpu")
    return x,y

class TokenWindowDataset(torch.utils.data.Dataset):
    def __init__(self,token_ids: list[int],seq_len :int) ->None:
        self.token_ids=token_ids
        self.T = seq_len
    def __len__(self) ->int:
        return max(0,len(self.token_ids)-self.T)
    def __getitem__(self,index:int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return make_window(self.token_ids,index,self.T)
        




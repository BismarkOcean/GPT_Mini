from gpt_mini.data.dataset import make_window,TokenWindowDataset
import pytest
import torch
from torch.utils.data import DataLoader
def test_make_window():
    token_ids=[1,3,2,4,5]
    x,y=make_window(token_ids,1,3)
    assert x.tolist()==[3,2,4]
    assert y.tolist()==[2,4,5]

def test_last_window():
    token_ids=[1,2,3,4]
    make_window(token_ids,1,2)
    with pytest.raises(IndexError):
        make_window(token_ids,2,2)

def test_raises_error():
    token_ids=[1,3,2,4,5]
    with pytest.raises(ValueError):
        make_window(token_ids,0,-1)
    with pytest.raises(ValueError):
        make_window(token_ids,0,0)
    with pytest.raises(IndexError):
        make_window(token_ids,-1,1)
    with pytest.raises(IndexError):
        make_window(token_ids,99,1)

def test_format():
    token_ids=[1,3,2,4,5]
    back=token_ids[:]
    x,y = make_window(token_ids,1,3)
    assert back == token_ids
    assert x.dtype==torch.long and y.dtype==torch.long
    assert x.device.type=="cpu" and y.device.type=="cpu"
    assert x.requires_grad==False and y.requires_grad==False
    assert x.shape == (3,)
    assert y.shape == (3,)

def test_token_window_dataset():
    token_ids=[2,5,1,4,0,3]
    T=3
    testobject=TokenWindowDataset(token_ids,T)
    assert len(testobject)==3
    x,y = testobject[2]
    assert x.tolist()==[1,4,0] and y.tolist()==[4,0,3]

def test_dataloader():
    token_ids = [2, 5, 1, 4, 0, 3]
    T = 3
    loader = DataLoader(
        TokenWindowDataset(token_ids,T),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    for i,(x_batch,y_batch) in enumerate(loader):
        if(i==0):
            assert x_batch.tolist() == [[2, 5, 1], [5, 1, 4]]
            assert y_batch.tolist() == [[5,1,4],[1,4,0]]
        if(i==1):
            assert x_batch.shape==(1,3)
            assert y_batch.shape==(1,3)



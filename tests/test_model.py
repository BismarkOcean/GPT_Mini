
import torch
from torch import nn
from gpt_mini.model.GPT import GPT


def test_gpt():
    torch.manual_seed(42)

    model = GPT(
        vocab_size=7,
        max_seq_len=8,
        embed_dim=8,
        num_heads=2,
        num_layers=2
    )

    # 1. 输出形状测试
    ids = torch.tensor([
        [1, 2, 3, 4],
        [0, 1, 6, 5]
    ], dtype=torch.long)

    logits = model(ids)
    assert logits.shape == (2, 4, 7)

    # 2. 因果性测试
    # 只改变最后一个 Token
    changed_ids = ids.clone()
    changed_ids[:, -1] = torch.tensor([5, 3])

    changed_logits = model(changed_ids)

    # 前三个位置的输出不应该受到影响
    assert torch.allclose(
        logits[:, :3, :],
        changed_logits[:, :3, :],
        atol=1e-6,
        rtol=1e-5
    )

    # 3. 梯度连通性测试
    loss = logits.square().mean()
    loss.backward()

    # 确认 Transformer Blocks 正确登记
    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == 2

    # 确认所有已登记参数都获得有限梯度
    for name, param in model.named_parameters():
        assert param.grad is not None, (
            f"{name} has no gradient"
        )
        assert torch.isfinite(param.grad).all(), (
            f"{name} has non-finite gradient"
        )

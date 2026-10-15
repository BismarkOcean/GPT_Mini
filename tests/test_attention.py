import torch
from gpt_mini.model.attention import SingleHeadCausalAttention,MultiHeadCausalAttention
import math
def test_singleheadcausalattention():
    # B=1, T=3, C=2
    x = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ]])

    model = SingleHeadCausalAttention(2)

    # 将 Q/K/V 权重设置为单位矩阵
    with torch.no_grad():
        W = torch.eye(2)
        model.q_proj.weight.copy_(W)
        model.k_proj.weight.copy_(W)
        model.v_proj.weight.copy_(W)

    out = model(x)

    # 1. 验证输出形状
    assert out.shape == (1, 3, 2)

    # 2. 验证数值
    expected = torch.tensor([[
        [1.0, 0.0],
        [0.3302, 0.6698],
        [0.7517, 0.7517]
    ]])

    torch.testing.assert_close(
        out, expected, atol=1e-4, rtol=0
    )

    # 3. 验证因果性
    x_modified = x.clone()
    x_modified[0, 2] = torch.tensor([8.0, -3.0])

    out_modified = model(x_modified)

    # 修改 Token 2，不应影响 Token 0 和 Token 1
    torch.testing.assert_close(
        out[:, :2, :],
        out_modified[:, :2, :]
    )

    # 4. 验证梯度可以正常反向传播
    loss = out.sum()
    loss.backward()

    for proj in [model.q_proj, model.k_proj, model.v_proj]:
        assert proj.weight.grad is not None
        assert torch.isfinite(proj.weight.grad).all()


def test_multi_head_causal_attention():
    B, T, C, H = 2, 3, 4, 2

    x = torch.tensor([
        [
            [1., 0., 0., 1.],
            [0., 1., 2., 0.],
            [1., 1., 0., 2.]
        ],
        [
            [2., 1., 1., 0.],
            [0., 2., 1., 3.],
            [3., 1., 0., 1.]
        ]
    ])

    model = MultiHeadCausalAttention(C, H)

    # 四个线性层全部设置为单位矩阵
    with torch.no_grad():
        for proj in (
            model.q_proj,
            model.k_proj,
            model.v_proj,
            model.out_proj
        ):
            proj.weight.copy_(torch.eye(C))

    out = model(x)

    # 1. 检查输出形状
    assert out.shape == (B, T, C)

    # 2. 手算样本0的前两个token
    # token0只能关注自身
    # token1每个头分别计算softmax

    # Head 0: scores = [0, 1/sqrt(2)]
    w0 = 1 / (1 + math.exp(1 / math.sqrt(2)))

    # Head 1: scores = [0, 4/sqrt(2)]
    w1 = 1 / (1 + math.exp(4 / math.sqrt(2)))

    expected = torch.tensor([
        [1., 0., 0., 1.],
        [w0, 1 - w0, 2 * (1 - w1), w1]
    ])

    torch.testing.assert_close(
        out[0, :2],
        expected,
        rtol=1e-5,
        atol=1e-6
    )

    # 3. 修改未来token
    x_modified = x.clone()
    x_modified[0, 2] = torch.tensor([-3., 4., 7., 2.])

    out_modified = model(x_modified)

    # 样本0的前两个token不应发生变化
    torch.testing.assert_close(
        out[0, :2],
        out_modified[0, :2]
    )

    # 样本1完全没有修改，因此输出不应变化
    torch.testing.assert_close(
        out[1],
        out_modified[1]
    )
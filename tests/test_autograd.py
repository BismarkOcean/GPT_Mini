import torch
def test_scalar_autograd():

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-1.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(4.0)
    y_hat = a * x + b
    loss = (y_hat - y) ** 2
    
    loss.backward()
    
    assert loss.item() == 1.0
    assert y_hat.item() == 5.0
    assert a.grad.item() == 6.0
    assert b.grad.item() == 2.0

    old_a_grad =a.grad.item()
    old_b_grad =b.grad.item()

    y_hat = a * x + b
    loss = (y_hat - y) ** 2

    loss.backward()

    assert old_a_grad*2 == a.grad.item()
    assert old_b_grad*2 == b.grad.item()

    a.grad=None
    b.grad=None

    y_hat = a * x + b
    loss = (y_hat - y) ** 2
    
    loss.backward()
    old_loss=loss
    assert a.grad.item() == 6.0
    assert b.grad.item() == 2.0

    with torch.no_grad():
        a-=0.01*a.grad
        b-=0.01*b.grad
    a.grad=None
    b.grad=None

    y_hat = a * x + b
    loss = (y_hat - y) ** 2

    loss.backward()

    assert loss<old_loss
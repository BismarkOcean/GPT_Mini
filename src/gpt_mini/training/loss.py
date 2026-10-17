import torch
import torch.nn.functional as F

def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    B, T, V = logits.shape
    logits=logits.reshape(B*T,V)
    targets=targets.reshape(B*T)
    loss= F.cross_entropy(logits,targets)
    return loss
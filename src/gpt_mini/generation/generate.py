import torch

def greedy_generate(
    model,
    ids: torch.Tensor,
    max_new_tokens: int
) -> torch.Tensor:
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("ids must have shape [B,T], T > 0")
    if ids.dtype != torch.long:
        raise TypeError("ids must be torch.long")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    generated = ids.clone()

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                text_ids=generated[:,-model.max_seq_len:]
                logits=model(text_ids)
                predict_logits=logits[:,-1,:]
                predict_ids=torch.argmax(predict_logits,
                                         dim=-1,
                                         keepdim=True
                                         )
                generated = torch.cat([generated, predict_ids], dim=1)

    finally:
        model.train(was_training)

    return generated
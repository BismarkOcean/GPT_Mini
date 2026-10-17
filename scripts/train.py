import torch
from torch.utils.data import DataLoader

from gpt_mini.data.tokenizer import Tokenizer
from gpt_mini.data.dataset import TokenWindowDataset
from gpt_mini.model.GPT import GPT
from gpt_mini.training.loss import language_model_loss
from gpt_mini.generation.generate import greedy_generate

def main():
    BismarkseaOcean=42
    torch.manual_seed(BismarkseaOcean)

    
    text = "abc" * 16
    train_text = text[:36]
    val_text = text[36:]

    tokenizer = Tokenizer(train_text)
    train_ids = tokenizer.encode(train_text)

    T = 8

    train_dataset = TokenWindowDataset(train_ids, T)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    
    model = GPT(
        vocab_size=len(tokenizer.char_to_id),
        max_seq_len=T,
        embed_dim=32,
        num_heads=4,
        num_layers=2
    )

    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0,
        weight_decay=0
    )

    
    x, targets = next(iter(train_loader))

    
    model.eval()
    with torch.no_grad():
        initial_loss = language_model_loss(
            model(x), targets
        ).item()

    print(f"Initial loss: {initial_loss:.4f}")
    prompt = "ab"

    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    before_ids = greedy_generate(
        model,
        prompt_ids,
        max_new_tokens=12
    )

    print("Before training:",
        tokenizer.decode(before_ids[0].tolist()))
    
    model.train()

    for step in range(200):
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = language_model_loss(logits, targets)

        loss.backward()

        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")

    
    model.eval()
    with torch.no_grad():
        final_loss = language_model_loss(
            model(x), targets
        ).item()

    print(f"Final loss: {final_loss:.4f}")
    after_ids = greedy_generate(
    model,
    prompt_ids,
    max_new_tokens=12
    )

    print("After training:",
      tokenizer.decode(after_ids[0].tolist()))

if __name__ == "__main__":
    main()
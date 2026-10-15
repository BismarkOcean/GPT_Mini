from gpt_mini.data.tokenizer import Tokenizer
from torch.utils.data import DataLoader
from gpt_mini.data.dataset import TokenWindowDataset

def main():
    text = "abcabcabcabcabcabc"
    T=3
    train=text[:12]
    val=text[12:]
    tokenizer = Tokenizer(train)
    train_ids=tokenizer.encode(train)
    val_ids=tokenizer.encode(val)
    train_dataset=TokenWindowDataset(train_ids,T)
    val_dataset=TokenWindowDataset(val_ids,T)
    train_dataloader=DataLoader(train_dataset,batch_size=2,shuffle=False,num_workers=0,drop_last=False)
    val_dataloader=DataLoader(val_dataset,batch_size=2,shuffle=False,num_workers=0,drop_last=False)
    print(len(tokenizer.char_to_id))
    print(len(train_dataset))
    print(len(val_dataset))
    x_batch, y_batch = next(iter(train_dataloader))
    print(x_batch.tolist())
    print(y_batch.tolist())
    print(x_batch.shape)
    print(y_batch.shape)

if __name__ == "__main__":
    main()
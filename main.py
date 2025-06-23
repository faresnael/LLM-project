import imp
from multiprocessing import context
import tiktoken
import torch
from GPTModel import GPTModel
from GPTDataset import GPTDatasetV1
from torch.utils.data import Dataset, Dataloader


def generate_text_simple(model,idx,max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:] #last 1024 items in the idx 
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        probas  = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx

def calc_loss_loader(data_loader, model, device, num_batches=None):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
model = GPTModel(GPT_CONFIG_124M)
start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
# print ("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print("encoded_tensor shape: ", encoded_tensor)

model.eval()
out = generate_text_simple(model = model, idx = encoded_tensor, max_new_tokens=6, context_size = GPT_CONFIG_124M["context_length"])
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print (decoded_text) 
with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text  = f.read()
def create_dataloader_v1 (txt, max_length=256, batch_Size=4, stride=128, shuffle=True, drop_last=True, num_workers = 0 ):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1 (txt, tokenizer, max_length, stride)
    dataloader = Dataloader(
        dataset,
        batch_size= batch_Size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )
    #data = tokenizer.encode(raw_text)
    return dataloader

train_loader = create_dataloader_v1()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader,model, device)

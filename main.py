import tiktoken
import torch
from GPTModel import GPTModel
from GPTDataset import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader

def generate(model, idx, max_new_tokens, context_size, temperature = 0.0, top_k = None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  #last 1024 items in the idx 
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:] #logits.shape() = (batch, context_length, vocab size) -> (batch,vocab size) for all batches and vocab posiblities you take the last index
        if top_k is not None:
            top_logits, _ = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits= torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
        else:
            idx_next = torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx,idx_next),dim=1)
    return idx

def calc_loss_batch(input_batch, target_batch,model,device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    # input_batch = input_batch.to(device)
    # target_batch = target_batch.to(device)
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i <num_batches:
            loss = calc_loss_batch(input_batch,target_batch, model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss/ num_batches
            

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
model = GPTModel(GPT_CONFIG_124M)
with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text  = f.read()
def create_dataloader_v1 (txt, max_length=256, batch_Size=4, stride=128, shuffle=True, drop_last=True, num_workers = 0 ):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1 (txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size= batch_Size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )
    #data = tokenizer.encode(raw_text)
    return dataloader

train_ratio = 0.9
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]
#print (val_data)
torch.manual_seed(123)
train_loader = create_dataloader_v1(train_data, max_length= GPT_CONFIG_124M["context_length"], batch_Size=2, stride=GPT_CONFIG_124M["context_length"], shuffle=True, drop_last=True,num_workers=0)
val_loader = create_dataloader_v1(val_data, max_length= GPT_CONFIG_124M["context_length"], batch_Size=2, stride=GPT_CONFIG_124M["context_length"], shuffle=True, drop_last=True,num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def text_to_tokens_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
def generate_and_print_sample(model, tokenizer,device,start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0] 
    encoded = text_to_tokens_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model=model, idx = encoded, max_new_tokens= 15, context_size=context_size, top_k =25, temperature=1.4)
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step+=1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1}(Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Evaluation loss {val_loss:.3f}")
            
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.1)
num_epochs = 10
tokenizer = tiktoken.get_encoding("gpt2")
train_losses, val_losses, tokens_seen = train_model_simple(model,train_loader, val_loader,optimizer,device,num_epochs = num_epochs,eval_freq=5,eval_iter=5,start_context="Every effort moves you", tokenizer = tokenizer)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict" : optimizer.state_dict(),
},
"model_and_optimizer.pth")

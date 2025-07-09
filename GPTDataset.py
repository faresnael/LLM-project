import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# weirdword = tokenizer.encode("akwirw ier")
# print(weirdword)
# print(tokenizer.decode([220]))

import torch 
from torch.utils.data import Dataset, DataLoader
class  GPTDatasetV1 (Dataset):
    def __init__ (self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length , stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def  __len__ (self):
        return len(self.input_ids)
    
    def __getitem__ (self,idx):
        return self.input_ids[idx], self.target_ids[idx]
max_length = 4
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

#dataloader = create_dataloader_v1 (raw_text, max_length=max_length, batch_Size = 8, stride = max_length, shuffle = False )
#data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# out_dim = 256
# token_embedding_layer = torch.nn.embedding(max_length, out_dim)
# token_embedding = token_embedding_layer(inputs)

# context_length = max_length
# pos_embedding_layer = torch.nn.embedding(context_length, out_dim)
# pos_embedding = pos_embedding_layer(torch.arange(context_length)) #torch arrange is [0,1,...,context_len-1]

# input_embeddings = token_embedding + pos_embedding

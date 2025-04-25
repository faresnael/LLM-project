import torch
import torch.nn as nn
# inputs = torch.tensor([
#     [0.43, 0.15, 0.89],
#     [0.55, 0.87, 0.66],
#     [0.57, 0.85, 0.64],
#     [0.22, 0.58, 0.33],
#     [0.77, 0.25, 0.10],
#     [0.05, 0.80, 0.55] 
# ],
#[
#     [0.43, 0.15, 0.89],
#     [0.55, 0.87, 0.66],
#     [0.57, 0.85, 0.64],
#     [0.22, 0.58, 0.33],
#     [0.77, 0.25, 0.10],
#     [0.05, 0.80, 0.55] 
# ]
#)


# x_2 = inputs[1]
# d_in = inputs.shape[1]
# d_out = 2

# torch.manual_seed(123)
# W_query = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad = False)
# W_key = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad = False)
# print ("w_query: ", W_query)
# print ("w_key: ",W_key)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert (d_out%num_heads==0), \
            "d_out must be divisble by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in,d_out, bias =qkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_vlaue = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout (dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length,context_length),diagonal = 1))

    def forward(self, x):
        b , num_tokens,d_in = x.shape
        keys = self.w_key(x) #linear function takes d_in and d_out and x (matrix), makes a matrix of shape (b,num_tokens,d_out)  (2,6,2)
        queries = self.w_query(x)
        values = self.w_vlaue(x)
        keys = keys.view(b,num_tokens,self.num_heads, self.head_dim) # changed the dimensions to (2,6,2,1)
        values = values.view(b,num_tokens,self.num_heads, self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads, self.head_dim)
        
        keys = keys.transpose(1,2) #changed again to (2,2,6,1)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3) #(2,2,6,6)
        print (attn_scores)

inputs = torch.tensor([[
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55] 
],
[
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55] 
]]
)
torch.manual_seed(123)
mha = MultiHeadAttention(3,2,6,0.0,2)
mha(inputs)
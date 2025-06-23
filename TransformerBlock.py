import torch 
import torch.nn as nn
from attention import MultiHeadAttention
from FeedForward import FeedForward
from LayerNorm import LayerNorm
class TransformerBlock (nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in = cfg["emb_dim"],d_out= cfg["emb_dim"],context_length=cfg["context_length"],num_heads=cfg["n_heads"],dropout = cfg["drop_rate"],qkv_bias=cfg["qkv_bias"])
        self.ff =  FeedForward(cfg)
        self.norm1 = LayerNorm (cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x= self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x= x+shortcut
        return x
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
torch.manual_seed(123)
x = torch.rand(2,4,768)
block  = TransformerBlock(GPT_CONFIG_124M)
output= block(x)
# import urllib.request
# url = (
#     "https://raw.githubusercontent.com/rasbt/"
#     "LLMs-from-scratch/main/ch05/"
#     "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url,filename)

from gpt_download import download_and_load_gpt2
from GPTModel import GPTModel
import torch
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

# print("settings:", settings)
# print("parameter dictionary key:" , params.keys())

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         f"Right: {right.shape} "
                         )
    return torch.nn.Parameter(torch.tensor(right))

# print("blocks:",params['blocks'][0]['attn'])

import numpy as np

def load_weights_into_gpt (gpt, params) :
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.w_query.weight = assign(
            gpt.trf_blocks[b].att.w_query.weight, q_w.T)
        gpt.trf_blocks[b].att.w_key.weight = assign(
            gpt.trf_blocks[b].att.w_key.weight, k_w.T)
        gpt.trf_blocks[b].att.w_value.weight = assign(
            gpt.trf_blocks[b].att.w_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.w_query.bias= assign(
            gpt.trf_blocks[b].att.w_query.bias, q_b)
        gpt.trf_blocks[b].att.w_key.bias = assign(
            gpt.trf_blocks[b].att.w_key.bias, k_b)
        gpt.trf_blocks[b].att.w_value.bias = assign(
            gpt.trf_blocks[b].att.w_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
             params["blocks"][b]["attn"]["c_proj"]["b"].T)
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"].T)
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"].T)
        
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight= assign(gpt.out_head.weight, params["wte"])


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}
gpt = GPTModel(GPT_CONFIG_124M)
gpt.eval()

load_weights_into_gpt (gpt, params)
torch.save(gpt.state_dict(),"model.pth")
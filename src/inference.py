from GPT import GPTModel
import torch
from loss import generate_and_print_sample
GPT_CONFIG_MINI = {
    "vocab_size": 50257,
    "context_length": 256,   # max sequence length
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": True
}

model = GPTModel(GPT_CONFIG_MINI)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("D:\Books_Code\LLM from Scratch\gpt_epoch4.pt", map_location=device))
model.eval()

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
generate_and_print_sample(model,tokenizer,device,start_context="Maggie said to Lily, ")
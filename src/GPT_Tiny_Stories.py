import tiktoken
from Data_loader import create_dataloader
import torch
from GPT import GPTModel
from loss import train_model_simple

# ---------------------------
# Tokenizer
# ---------------------------
tokenizer = tiktoken.get_encoding("gpt2") 

# ---------------------------
# GPT-2 Mini Config
# ---------------------------
GPT_CONFIG_MINI = {
    "vocab_size": 50257,
    "context_length": 256,   # max sequence length
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": True
}

# ---------------------------
# Load Tiny Shakespeare
# ---------------------------
with open("D:\Books_Code\LLMs from Scratch\TinyStories_10000.txt",'r',encoding='utf-8') as f:
    text_data = f.read()

print("Loading the Data......")

# ---------------------------
# Train/Validation Split
# ---------------------------
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

print("Data Loading Done ✅")

# ---------------------------
# DataLoader Creation
# ---------------------------
train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_MINI["context_length"],
    stride=192,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_MINI["context_length"],
    stride=192,
    drop_last=False,
    shuffle=False,
    num_workers=0
)


print("Data Loader Complete...✅")

# ---------------------------
# Model Setup
# ---------------------------
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

model = GPTModel(GPT_CONFIG_MINI)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Model Loaded to device ", device)

# Optional: gradient checkpointing for memory
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# ---------------------------
# Optimizer
# ---------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,              
    betas=(0.9, 0.95),    
    eps=1e-8,
    weight_decay=0.1
)

# ---------------------------
# Training
# ---------------------------
num_epochs = 5

print("Training Started....")
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=1500,
    eval_iter=5,
    start_context="One day, a little girl",
    tokenizer=tokenizer
)
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math


# IMPORT base.py functions 

from base import set_seed, eli5_dataset



# MAIN SCRIPT 

def main():

    
    # 1. we Set seed FIRST 
    set_seed(seed=0)

    
    # 2. Device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    # 3. Load tokenizer + datasets 
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    BATCH_SIZE = 32
    MAX_POSITION_EMBEDDINGS = 200

    print("Loading datasets...")

    trainset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "train")
    validset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "valid")
    testset  = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "test")

    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False)

    print("Train size:", len(trainset))
    print("Valid size:", len(validset))
    print("Test size :", len(testset))

    
    # 4. Model definition (GPT-2 small custom)
    
    config = GPT2Config(
        vocab_size=50257,
        n_positions=MAX_POSITION_EMBEDDINGS,
        n_ctx=MAX_POSITION_EMBEDDINGS,
        n_embd=512,
        n_layer=6,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model = GPT2LMHeadModel(config).to(device)

    print("Model initialized with", sum(p.numel() for p in model.parameters()), "parameters.")

    
    # 5. Optimizer + Scheduler
    
    LEARNING_RATE = 2e-4
    EPOCHS = 4
    WARMUP_STEPS = 500

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    steps_per_epoch = math.ceil(len(train_dataloader))
    TOTAL_TRAIN_STEPS = EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        return max(
            0.0,
            float(TOTAL_TRAIN_STEPS - step)
            / float(max(1, TOTAL_TRAIN_STEPS - WARMUP_STEPS))
        )

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    
    # 6. Training loop
    
    def train_epoch(model, dataloader):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def eval_epoch(model, dataloader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device, dtype=torch.long)
                outputs = model(batch, labels=batch)
                loss = outputs.loss
                total_loss += loss.item()

        return total_loss / len(dataloader)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_loss = train_epoch(model, train_dataloader)
        val_loss   = eval_epoch(model, valid_dataloader)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Valid loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            print("New best model saved.")
        else:
            print("Validation loss did not improve. Early stop.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Best model reloaded.")

    
    # 7. Generate logits for test set
    
    print("\nGenerating logits for test set...")

    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device, dtype=torch.long)
            outputs = model(batch)
            logits = outputs.logits  # (B, 200, 50257)
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)  # (75, 200, 50257)

   
    # 8. Convert to float16 & save
    
    logits_np = all_logits.numpy().astype(np.float16)

    filename = "LLM_Model.npy"
    np.save(filename, logits_np)

    print("Saved logits shape:", logits_np.shape)
    print(f"Saved as {filename}")



# RUN MAIN

if __name__ == "__main__":
    main()

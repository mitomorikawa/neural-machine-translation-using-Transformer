"""
train_torchTransformer.py

Language Translation with nn.Transformer — eng → fra.

This script follows the official PyTorch tutorial:
  https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/
  _downloads/8cdd9a659f7d22e15eb4a689206e4b6b/translation_transformer.ipynb

Data is loaded from pre-built .pt tensors produced by preprocess.py:
  ../data/eng_{train,val,test}.pt   (source — English)
  ../data/fra_{train,val,test}.pt   (target — French)

Special token indices (matching preprocessor.py):
  <sos>=0  <eos>=1  <pad>=2  <unk>=3
Note: the tutorial uses BOS/EOS=2/3 and PAD=1.  We keep our project indices.
"""

import os
import math
import time
import argparse
from datetime import datetime
import pickle
from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from library.nn_architectures import Seq2SeqTransformer, create_masks

# ─────────────────────────────────────────────────────────────────────────────
# Special token indices
# ─────────────────────────────────────────────────────────────────────────────
SOS_IDX = 0    # <sos>
EOS_IDX = 1    # <eos>
PAD_IDX = 2    # <pad>
UNK_IDX = 3    # <unk>

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_tensor(path: str) -> Tensor:
    """Load a .pt tensor onto DEVICE."""
    t = torch.load(path, map_location=DEVICE, weights_only=True)
    return t

def make_dataloader(src: Tensor, tgt: Tensor, batch_size: int, shuffle: bool = True):
    dataset = TensorDataset(src, tgt)   # each sample: (S,), (T,)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# learning rate scheduler
# ─────────────────────────────────────────────────────────────────────────────
class TransformerScheduler:
    """
    Implements the learning rate schedule from the Transformer paper:
    lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lr = d_model ** -0.5
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        arg1 = self.current_step ** -0.5
        arg2 = self.current_step * (self.warmup_steps ** -1.5)
        return self.base_lr * min(arg1, arg2) 

# ─────────────────────────────────────────────────────────────────────────────
# train_epoch / evaluate
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, train_dataloader, loss_fn, optimizer, scaler, scheduler):
    model.train()
    losses = 0

    for src_b, tgt_b in tqdm(train_dataloader, desc="  train", leave=False):
        src = src_b.to(DEVICE)   # (B, S)
        tgt = tgt_b.to(DEVICE)   # (B, T)

        tgt_input = tgt[:, :-1]  # remove last token  → decoder input

        optimizer.zero_grad()

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            logits = model(src, tgt_input)
            tgt_out = tgt[:, 1:]     # remove first token (<sos>) → targets
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses += loss.item()

    return losses / len(train_dataloader)

def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    losses = 0

    with torch.no_grad():
        for src_b, tgt_b in tqdm(val_dataloader, desc="    val", leave=False):
            src = src_b.to(DEVICE)
            tgt = tgt_b.to(DEVICE)

            tgt_input = tgt[:, :-1]

            logits = model(src, tgt_input)

            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

    return losses / len(val_dataloader)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train custom Seq2SeqTransformer for NMT (eng→fra)")
    parser.add_argument("--emb_size",        type=int,   default=512)
    parser.add_argument("--nhead",           type=int,   default=8)
    parser.add_argument("--num_layers",      type=int,   default=6)
    parser.add_argument("--ffn_hid_dim",     type=int,   default=2048)
    parser.add_argument("--seq_len",         type=int,   default=100)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--num_epochs",      type=int,   default=100)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience",        type=int,   default=10)
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument("--model_dir",       type=str,   default="../models")
    parser.add_argument("--log_dir",         type=str,   default="../runs")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    print(f"Device: {DEVICE}")

    # ── Vocab sizes from saved pkl files ───────────────────────────────────
    with open(os.path.join(args.data_dir, "eng_vocab.pkl"), "rb") as f:
        src_vocab_size = pickle.load(f)["vocab_size"]
    with open(os.path.join(args.data_dir, "fra_vocab.pkl"), "rb") as f:
        tgt_vocab_size = pickle.load(f)["vocab_size"]
    print(f"src vocab size: {src_vocab_size}  |  tgt vocab size: {tgt_vocab_size}")

    # ── Data ───────────────────────────────────────────────────────────────
    print("Loading tensors...")
    src_train = load_tensor(os.path.join(args.data_dir, "eng_train.pt"))
    tgt_train = load_tensor(os.path.join(args.data_dir, "fra_train.pt"))
    src_val   = load_tensor(os.path.join(args.data_dir, "eng_val.pt"))
    tgt_val   = load_tensor(os.path.join(args.data_dir, "fra_val.pt"))

    train_dataloader = make_dataloader(src_train, tgt_train, args.batch_size, shuffle=True)
    val_dataloader   = make_dataloader(src_val,   tgt_val,   args.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────
    torch.manual_seed(0) #sets the seed for random number generation, ensures reproducibility

    transformer = Seq2SeqTransformer(
        d_model=args.emb_size,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.ffn_hid_dim,
        seq_len=args.seq_len,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dropout=args.dropout,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Loss & optimiser ────────────────────────────────
    loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = TransformerScheduler(
        optimizer,
        d_model=args.emb_size,
        warmup_steps=1000
    )
    scaler = torch.amp.GradScaler(device=DEVICE.type)

    # ── Training loop ───────────────────────────────────
    writer            = SummaryWriter(log_dir=args.log_dir)
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, train_dataloader, loss_fn, optimizer, scaler, scheduler)
        end_time   = timer()
        val_loss   = evaluate(transformer, val_dataloader, loss_fn)
        current_lr = scheduler.get_lr()
        print(
            f"Epoch: {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"Val loss: {val_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s, "
            f"Current lr = {current_lr}, "
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.flush()

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(args.model_dir, f"transformer_best.pt")
            torch.save(transformer.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping after {args.patience} epochs with no improvement.")
                break

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()

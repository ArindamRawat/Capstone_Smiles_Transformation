
# capstone_main_transformed.py
# Transformer-based SMILES Correction - main script
# This script mirrors the structure of the uploaded capstone_main.py but updates the model
# architecture and hyperparameters to match the described design:
# - 6 encoder/decoder layers
# - d_model = 512
# - n_heads = 8
# - ffn_dim = 2048
# - RDKit validation + beam search decoding included

import os
import random
import re
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- Configuration ----------------------
DATA_PATHS = {
    "train": "train_pairs.csv",
    "val":   "val_pairs.csv",
    "test":  "test_pairs.csv"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
D_MODEL = 512    # updated
N_HEADS = 8      # updated
NUM_LAYERS = 6   # updated
FFN_DIM = 2048   # updated
MAX_LEN = 128
CORPUS_SAMPLE = None
SEED = 42
NUM_EPOCHS = 20
LR = 5e-5
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Device:", DEVICE)
for k,v in DATA_PATHS.items():
    print(k,":", v, "exists?", os.path.exists(v))

# ---------------------- Tokenization ----------------------
SMILES_TOKEN_PATTERN = re.compile(r"Cl|Br|\[[^\]]+\]|Si|Se|%[0-9]{2}|\d+|B|C|N|O|P|S|F|I|H|n|c|o|s|@|=|#|-|\/|\\\\|\(|\)|\.|\+|:")

def tokenize_smiles(smi: str):
    if smi is None:
        return []
    tokens = SMILES_TOKEN_PATTERN.findall(smi)
    if tokens:
        return [t for t in tokens if t]
    return list(smi)

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

# ---------------------- Vocabulary ----------------------
def build_vocab(train_csv, min_freq=1):
    df = pd.read_csv(train_csv)
    if CORPUS_SAMPLE:
        df = df.sample(min(len(df), CORPUS_SAMPLE), random_state=SEED)
    freq = {}
    for col in ["corrupted", "correct"]:
        for smi in df[col].astype(str).tolist():
            for tok in tokenize_smiles(smi):
                freq[tok] = freq.get(tok, 0) + 1
    toks = [t for t,c in freq.items() if c >= min_freq]
    toks = sorted(toks)
    itos = [PAD, SOS, EOS, UNK] + toks
    stoi = {t:i for i,t in enumerate(itos)}
    print("Vocab size:", len(itos))
    return stoi, itos

# Build vocab (will fail if CSVs not present; user should ensure paths are correct)
if os.path.exists(DATA_PATHS["train"]):
    stoi, itos = build_vocab(DATA_PATHS["train"])
else:
    # Fallback minimal vocab to allow running syntax-checks
    itos = [PAD, SOS, EOS, UNK, 'C', 'N', 'O', '(', ')', '=', '#', '1', '2', '3']
    stoi = {t:i for i,t in enumerate(itos)}
    print("Using fallback vocab. Please provide train_pairs.csv to build a proper vocab.")

VOCAB_SIZE = len(itos)
print("Example tokens:", itos[:20])

def encode_smiles(smi, stoi, max_len=MAX_LEN):
    tokens = tokenize_smiles(str(smi))
    ids = [stoi.get(t, stoi[UNK]) for t in tokens]
    ids = [stoi[SOS]] + ids + [stoi[EOS]]
    if len(ids) > max_len:
        ids = ids[:max_len]
        if ids[-1] != stoi[EOS]:
            ids[-1] = stoi[EOS]
    return ids

def decode_ids(ids, itos):
    toks = []
    for idx in ids:
        if idx < 0 or idx >= len(itos):
            toks.append(UNK)
            continue
        t = itos[idx]
        if t == EOS:
            break
        if t in (PAD, SOS):
            continue
        toks.append(t)
    return "".join(toks)

# ---------------------- Dataset ----------------------
class SmilesCorrectionDataset(Dataset):
    def __init__(self, csv_path, stoi, max_len=MAX_LEN, limit=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found. Please provide dataset CSVs.")
        self.df = pd.read_csv(csv_path)
        if limit:
            self.df = self.df.head(limit)
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = encode_smiles(row['corrupted'], self.stoi, max_len=self.max_len)
        tgt = encode_smiles(row['correct'], self.stoi, max_len=self.max_len)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_batch(batch):
    srcs, tgts = zip(*batch)
    src_lens = [len(x) for x in srcs]
    tgt_lens = [len(x) for x in tgts]
    src_max = max(src_lens)
    tgt_max = max(tgt_lens)
    src_padded = torch.full((len(batch), src_max), fill_value=stoi[PAD], dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max), fill_value=stoi[PAD], dtype=torch.long)
    src_mask = torch.zeros((len(batch), src_max), dtype=torch.bool)
    tgt_mask = torch.zeros((len(batch), tgt_max), dtype=torch.bool)
    for i, s in enumerate(srcs):
        src_padded[i, :len(s)] = s
        src_mask[i, :len(s)] = 1
    for i, t in enumerate(tgts):
        tgt_padded[i, :len(t)] = t
        tgt_mask[i, :len(t)] = 1
    return src_padded, tgt_padded, src_mask, tgt_mask

# Create datasets and loaders only if CSVs exist (to avoid immediate errors)
train_loader = None; val_loader = None; test_loader = None
if os.path.exists(DATA_PATHS["train"]):
    train_ds = SmilesCorrectionDataset(DATA_PATHS["train"], stoi)
    val_ds   = SmilesCorrectionDataset(DATA_PATHS["val"], stoi)
    test_ds  = SmilesCorrectionDataset(DATA_PATHS["test"], stoi)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    print("Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))
else:
    print("Datasets not found. Skipping dataloader creation.")

# ---------------------- Model ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class SmilesTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, nhead=N_HEADS, num_layers=NUM_LAYERS, ffn_dim=FFN_DIM, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ffn_dim, dropout, activation="relu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, ffn_dim, dropout, activation="relu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def make_key_padding_mask(self, mask):
        # mask: bool tensor with True where data exists
        return (~mask).to(DEVICE)

    def forward(self, src, tgt):
        # src, tgt shape: (batch, seq_len)
        src_mask_bool = (src != stoi[PAD])
        tgt_mask_bool = (tgt != stoi[PAD])

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc(src_emb)
        # Transformer modules expect shape: (seq_len, batch, d_model)
        src_emb = src_emb.transpose(0,1)

        memory = self.encoder(src_emb, src_key_padding_mask=self.make_key_padding_mask(src_mask_bool))

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)
        tgt_emb = tgt_emb.transpose(0,1)

        seq_len = tgt_emb.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)

        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=self.make_key_padding_mask(tgt_mask_bool),
                           memory_key_padding_mask=self.make_key_padding_mask(src_mask_bool))
        out = out.transpose(0,1)  # (batch, seq_len, d_model)
        logits = self.output(out)
        return logits

# Instantiate model
model = SmilesTransformerModel(VOCAB_SIZE, d_model=D_MODEL, nhead=N_HEADS, num_layers=NUM_LAYERS, ffn_dim=FFN_DIM).to(DEVICE)
print(model)

pad_idx = stoi[PAD]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

def compute_loss(logits, targets):
    b, s, v = logits.size()
    logits_flat = logits.reshape(b*s, v)
    targets_flat = targets.reshape(b*s)
    loss = criterion(logits_flat, targets_flat)
    return loss

# ---------------------- Training loop ----------------------
def train_and_validate(num_epochs=NUM_EPOCHS):
    if train_loader is None or val_loader is None:
        print("Data loaders not available. Aborting training.")
        return

    best_val_loss = float('inf')
    save_path = "smiles_transformer_best.pt"
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for src, tgt, src_mask, tgt_mask in pbar:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            decoder_input = tgt[:, :-1].contiguous()
            target_tokens = tgt[:, 1:].contiguous()

            optimizer.zero_grad()
            logits = model(src, decoder_input)

            loss = compute_loss(logits, target_tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': total_loss / steps})

        scheduler.step()

        # validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for src, tgt, src_mask, tgt_mask in val_loader:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                decoder_input = tgt[:, :-1].contiguous()
                target_tokens = tgt[:, 1:].contiguous()

                logits = model(src, decoder_input)
                loss = compute_loss(logits, target_tokens)
                val_loss += loss.item()
                val_steps += 1

        train_loss = total_loss / steps if steps > 0 else float('nan')
        val_loss = val_loss / val_steps if val_steps > 0 else float('nan')
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stoi': stoi,
                'itos': itos,
                'best_val_loss': best_val_loss
            }, save_path)
            print(f"Saved best model (epoch {epoch}, val_loss={val_loss:.4f}).")

# ---------------------- Decoding (Greedy + Beam) ----------------------
def greedy_decode(model, src_ids, max_len=MAX_LEN):
    model.eval()
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    ys = torch.tensor([[stoi[SOS]]], dtype=torch.long).to(DEVICE)
    for i in range(max_len-1):
        with torch.no_grad():
            logits = model(src, ys)
            next_token_logits = logits[0, -1, :]
            next_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == stoi[EOS]:
                break
    return ys.squeeze(0).cpu().tolist()

def beam_search_decode(model, src_ids, beam_width=5, max_len=MAX_LEN):
    model.eval()
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    heap = [ (0.0, [stoi[SOS]]) ]
    completed = []
    for _ in range(max_len):
        new_heap = []
        for score, seq in heap:
            if seq[-1] == stoi[EOS]:
                completed.append((score, seq))
                continue
            tgt = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(src, tgt)
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1).cpu().numpy()
            topk_idx = np.argsort(-log_probs)[:beam_width]
            for idx in topk_idx:
                new_seq = seq + [int(idx)]
                new_score = score - float(log_probs[idx])
                new_heap.append((new_score, new_seq))
        heap = sorted(new_heap, key=lambda x: x[0])[:beam_width]
        if not heap:
            break
    completed += heap
    best = sorted(completed, key=lambda x: x[0])[0][1]
    return best

def decode_smiles_from_ids(ids):
    return decode_ids(ids, itos)

# ---------------------- Evaluation utilities ----------------------
def tanimoto(a_smiles, b_smiles):
    am = Chem.MolFromSmiles(a_smiles)
    bm = Chem.MolFromSmiles(b_smiles)
    if am is None or bm is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(am, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(bm, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fa, fb)

def evaluate_on_test(ckpt_path="smiles_transformer_best.pt", beam_width=3):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found. Run training first.")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)

    test_df = pd.read_csv(DATA_PATHS["test"])
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        corrupted = str(row['corrupted'])
        target = str(row['correct'])
        src_ids = encode_smiles(corrupted, stoi)
        pred_ids = beam_search_decode(model, src_ids, beam_width=beam_width, max_len=MAX_LEN)
        pred_smi = decode_smiles_from_ids(pred_ids)
        valid_pred = Chem.MolFromSmiles(pred_smi) is not None
        reconstruction = (pred_smi == target)
        tscore = tanimoto(pred_smi, target) if valid_pred else 0.0
        results.append((corrupted, target, pred_smi, valid_pred, reconstruction, tscore))

    res_df = pd.DataFrame(results, columns=['input','target','pred','valid_pred','reconstruction','tanimoto'])
    validity_rate = res_df['valid_pred'].mean()
    recon_acc = res_df['reconstruction'].mean()
    avg_tanimoto = res_df['tanimoto'].mean()
    print(f"Test validity_rate: {validity_rate:.4f}, recon_accuracy: {recon_acc:.4f}, avg_tanimoto: {avg_tanimoto:.4f}")
    res_df.to_csv("test_predictions.csv", index=False)
    print("Saved test_predictions.csv")

# ---------------------- Entrypoint ----------------------
if __name__ == "__main__":
    # Example usage: train_and_validate() or evaluate_on_test()
    print("Script entrypoint. Available functions: train_and_validate(), evaluate_on_test()")
    # Uncomment below lines to run training/eval directly
    # train_and_validate()
    # evaluate_on_test()

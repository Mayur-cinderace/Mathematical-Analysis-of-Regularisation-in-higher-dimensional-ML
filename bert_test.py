import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

from datasets import load_dataset
import numpy as np

# -----------------------------
# 1. SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

# Try different weight decay values
WEIGHT_DECAYS = [0.0, 1e-4, 1e-2]

torch.manual_seed(42)

# -----------------------------
# 2. LOAD SMALL DATASET (to induce overfitting)
# -----------------------------
dataset = load_dataset("imdb")

# intentionally small training set (overparameterized regime)
train_texts = dataset["train"]["text"][:500]
train_labels = dataset["train"]["label"][:500]

val_texts = dataset["test"]["text"][:500]
val_labels = dataset["test"]["label"][:500]

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts, labels):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    enc["labels"] = torch.tensor(labels)
    return enc

train_enc = tokenize(train_texts, train_labels)
val_enc = tokenize(val_texts, val_labels)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}

train_loader = DataLoader(TextDataset(train_enc), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TextDataset(val_enc), batch_size=BATCH_SIZE)

# -----------------------------
# 3. TRAINING FUNCTION
# -----------------------------
def train_model(weight_decay):
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=weight_decay
    )

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # ---- VALIDATION ----
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

        val_losses.append(total_val_loss / len(val_loader))

        print(
            f"[WD={weight_decay}] "
            f"Epoch {epoch+1}: "
            f"Train={train_losses[-1]:.4f}, "
            f"Val={val_losses[-1]:.4f}"
        )

    return train_losses, val_losses

# -----------------------------
# 4. RUN EXPERIMENTS
# -----------------------------
results = {}

for wd in WEIGHT_DECAYS:
    print("\n==============================")
    print(f"Training with weight_decay = {wd}")
    print("==============================")
    results[wd] = train_model(wd)

# -----------------------------
# 5. PRINT SUMMARY
# -----------------------------
print("\nFinal losses:")
for wd, (train_l, val_l) in results.items():
    print(
        f"WD={wd:>6} | "
        f"Train={train_l[-1]:.4f} | "
        f"Val={val_l[-1]:.4f}"
    )

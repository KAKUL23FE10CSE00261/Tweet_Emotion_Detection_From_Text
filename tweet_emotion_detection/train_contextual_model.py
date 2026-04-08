import os, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
CSV_PATH = "/kaggle/input/datasets/kakulthisside/contextual-emotion-dataset-csv/contextual_emotion_dataset.csv"

# 🔥 SWITCH MODEL HERE
BASE_MODEL_DIR = "roberta-base"   # 🔥 change to bert-base-uncased if needed

OUTPUT_DIR = "/kaggle/working/bert_contextual_model"

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥 Device: {DEVICE}")

# ─────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────
EMOTIONS = [
    "anger","boredom","empty","enthusiasm","fun",
    "happiness","hate","love","neutral","relief",
    "sadness","surprise","worry"
]
label2id = {e: i for i, e in enumerate(EMOTIONS)}
id2label = {i: e for i, e in enumerate(EMOTIONS)}

# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class ContextualEmotionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding = self.tokenizer(
            str(row["previous_text"]),
            str(row["current_text"]),
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label2id[row["emotion"]])
        }

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["emotion"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["emotion"])

# ─────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["emotion"]),
    y=df["emotion"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_DIR,
    num_labels=len(EMOTIONS),
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

print("✅ Model loaded")

# ─────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────
train_dataset = ContextualEmotionDataset(train_df, tokenizer)
val_dataset   = ContextualEmotionDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ─────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

scaler = GradScaler()

# ─────────────────────────────────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────────────────────────────────
def train_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total

# ─────────────────────────────────────────────────────────
# VALIDATION FUNCTION
# ─────────────────────────────────────────────────────────
def evaluate():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ─────────────────────────────────────────────────────────
# TRAIN LOOP + EARLY STOPPING
# ─────────────────────────────────────────────────────────
best_val_acc = 0
patience = 2
counter = 0

print("\n🚀 Training started...\n")

for epoch in range(EPOCHS):
    loss, train_acc = train_epoch()
    val_acc = evaluate()

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("✅ Best model saved")
    else:
        counter += 1

    if counter >= patience:
        print("⛔ Early stopping triggered")
        break

# ─────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────
print("\n🎉 Training Complete!")
print("Best Validation Accuracy:", best_val_acc)
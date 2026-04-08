"""
=============================================================================
train_model2_window3.py
=============================================================================
Model 2 — INDEPENDENT TRAINING
Dataset : window_3.csv  (20,001 pairs)
Base    : roberta-base  (fresh — no dependency on any other model)
Output  : /kaggle/working/model_window3/

What this dataset contains:
  - previous_text = two messages joined: "msg1 | msg2"
  - current_text  = third message
  - Emotion label = current message emotion
  - Teaches model: multi-turn rolling context (3-message window)

NOTE: MAX_LEN = 192 because previous_text is longer (~148 chars avg)
=============================================================================
"""

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

try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = "cuda"

# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH   = "/kaggle/input/datasets/kakulthisside/context-dataset/window_3.csv"
BASE_MODEL = "roberta-base"                  # ✅ Fresh start — no dependency
OUTPUT_DIR = "/kaggle/working/model_window3"

MAX_LEN      = 192        # ← Longer than others! prev_text has 2 messages joined
BATCH_SIZE   = 16         # ← Smaller batch because longer sequences use more memory
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1
PATIENCE     = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP      = DEVICE.type == "cuda"

print("=" * 55)
print("  MODEL 2 — window_3.csv")
print("  Independent training from roberta-base")
print("=" * 55)
print(f"  Device  : {DEVICE}")
print(f"  Base    : {BASE_MODEL}  (fresh start)")
print(f"  MaxLen  : {MAX_LEN}  (longer — 2 messages in previous_text)")
print(f"  Output  : {OUTPUT_DIR}")
print("=" * 55)

# =============================================================================
# LABELS
# =============================================================================
EMOTIONS = ["anger","boredom","empty","enthusiasm","fun",
            "happiness","hate","love","neutral","relief",
            "sadness","surprise","worry"]
label2id = {e: i for i, e in enumerate(EMOTIONS)}
id2label = {i: e for i, e in enumerate(EMOTIONS)}

# =============================================================================
# DATASET
# =============================================================================
class ContextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        enc = self.tokenizer(
            str(row["previous_text"]),
            str(row["current_text"]),
            padding      = "max_length",
            truncation   = True,
            max_length   = MAX_LEN,
            return_tensors = "pt"
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(label2id[row["emotion"]], dtype=torch.long)
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze()
        return item

# =============================================================================
# LOAD DATA
# =============================================================================
print(f"\n📂 Loading dataset...")
df = pd.read_csv(CSV_PATH).dropna(subset=["previous_text","current_text","emotion"])
df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)
print(f"   Total rows : {len(df):,}")
print(f"   Avg prev_text length : {df['previous_text'].str.len().mean():.0f} chars  (2 messages joined by '|')")
print(f"   Emotion distribution:\n{df['emotion'].value_counts().to_string()}")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=42)
print(f"\n   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# =============================================================================
# CLASS WEIGHTS
# =============================================================================
weights       = compute_class_weight("balanced", classes=np.array(EMOTIONS), y=df["emotion"].values)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"\n⚖️  Imbalance ratio: {weights.max()/weights.min():.1f}x — class weights applied")

# =============================================================================
# MODEL
# =============================================================================
print(f"\n🤖 Loading: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels = len(EMOTIONS),
    id2label   = id2label,
    label2id   = label2id
).to(DEVICE)
print("✅ Model ready")

# =============================================================================
# DATALOADERS
# =============================================================================
train_loader = DataLoader(ContextDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(ContextDataset(val_df,   tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(ContextDataset(test_df,  tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# =============================================================================
# OPTIMIZER + SCHEDULER
# =============================================================================
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = GradScaler() if USE_AMP else None

# =============================================================================
# TRAIN + EVAL
# =============================================================================
def train_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(train_loader, desc="  Training", leave=False):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["label"].to(DEVICE)
        kw   = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in batch:
            kw["token_type_ids"] = batch["token_type_ids"].to(DEVICE)

        optimizer.zero_grad()
        if USE_AMP:
            with autocast(device_type=AMP_DEVICE):
                out  = model(**kw)
                loss = torch.nn.CrossEntropyLoss(weight=class_weights)(out.logits, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(**kw)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights)(out.logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(out.logits, dim=1)
        correct    += (preds == lbls).sum().item()
        total      += lbls.size(0)
    return total_loss / len(train_loader), correct / total


def evaluate(loader, desc="Eval"):
    model.eval()
    correct, total       = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {desc}", leave=False):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)
            kw   = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in batch:
                kw["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
            out   = model(**kw)
            preds = torch.argmax(out.logits, dim=1)
            correct    += (preds == lbls).sum().item()
            total      += lbls.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    return correct / total, all_preds, all_labels

# =============================================================================
# TRAINING LOOP
# =============================================================================
best_val_acc = 0.0
patience_ctr = 0
print(f"\n🚀 Training — {EPOCHS} epochs, patience={PATIENCE}\n")

for epoch in range(1, EPOCHS + 1):
    print(f"── Epoch {epoch}/{EPOCHS} ──")
    train_loss, train_acc = train_epoch()
    val_acc, _, _         = evaluate(val_loader, "Validation")
    print(f"  Train loss: {train_loss:.4f}  | Train acc: {train_acc:.4f}")
    print(f"  Val   acc : {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_ctr = 0
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  ✅ Best model saved → {OUTPUT_DIR}")
    else:
        patience_ctr += 1
        print(f"  ⚠️  No improvement ({patience_ctr}/{PATIENCE})")
        if patience_ctr >= PATIENCE:
            print("  ⛔ Early stopping")
            break

# =============================================================================
# TEST
# =============================================================================
from transformers import AutoModelForSequenceClassification as AMSC
model = AMSC.from_pretrained(OUTPUT_DIR).to(DEVICE)
test_acc, preds, labels = evaluate(test_loader, "Test")

print(f"\n{'='*55}")
print(f"  ✅ MODEL 2 COMPLETE — window_3")
print(f"  Best Val Acc : {best_val_acc:.4f}")
print(f"  Test Acc     : {test_acc:.4f}")
print(f"  Saved to     : {OUTPUT_DIR}/")
print(f"{'='*55}")
print(classification_report(labels, preds, target_names=EMOTIONS, digits=3, zero_division=0))
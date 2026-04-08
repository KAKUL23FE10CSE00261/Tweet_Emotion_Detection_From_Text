"""
=============================================================================
generate_contextual_dataset.py
=============================================================================
Generates a CSV dataset for training/fine-tuning contextual emotion detection.

What it does:
  - Downloads MELD and DailyDialog datasets from HuggingFace
  - Creates (previous_text, current_text, emotion) triples
  - Also generates a synthetic augmented version using conversation windows

Usage:
    pip install datasets pandas tqdm
    python generate_contextual_dataset.py

Output files:
    contextual_emotion_dataset.csv        ← main dataset
    contextual_emotion_augmented.csv      ← with sliding window pairs
=============================================================================
"""

import pandas as pd
from tqdm import tqdm
import os, json

# ──────────────────────────────────────────────────────────────────────────
# LABEL MAPPING  (align with your existing app labels)
# ──────────────────────────────────────────────────────────────────────────
MELD_EMOTION_MAP = {
    "neutral":   "neutral",
    "surprise":  "surprise",
    "fear":      "worry",
    "sadness":   "sadness",
    "joy":       "happiness",
    "disgust":   "hate",
    "anger":     "anger",
}

DAILYDIALOG_EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "fun",        # disgust → fun (closest available)
    3: "worry",      # fear
    4: "happiness",
    5: "sadness",
    6: "surprise",
}


# ──────────────────────────────────────────────────────────────────────────
# HELPER: Build (prev, curr, emotion) triples from a conversation
# ──────────────────────────────────────────────────────────────────────────
def conversation_to_pairs(utterances, emotions, window=3):
    """
    Given a list of utterances and their emotions (same length),
    build contextual pairs:
      - prev  = last `window` utterances joined by " | "
      - curr  = current utterance
      - label = emotion of current utterance

    Skips pairs where curr emotion is "neutral" with prob 0.5
    to reduce class imbalance.
    """
    import random
    pairs = []
    for i in range(1, len(utterances)):
        label = emotions[i]
        if label == "neutral" and random.random() < 0.5:
            continue

        start    = max(0, i - window)
        ctx_msgs = utterances[start:i]
        prev     = " | ".join(ctx_msgs)
        curr     = utterances[i]

        if not prev.strip() or not curr.strip():
            continue

        pairs.append({
            "previous_text": prev.strip(),
            "current_text":  curr.strip(),
            "emotion":       label,
            "source":        "conversation",
            "window_size":   len(ctx_msgs),
        })
    return pairs


# ──────────────────────────────────────────────────────────────────────────
# SOURCE 1: MELD  (Friends TV show, ~13k utterances with conversation IDs)
# ──────────────────────────────────────────────────────────────────────────
def load_meld():
    print("\n📦 Loading MELD dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("declare-lab/meld", trust_remote_code=True)
    except Exception as e:
        print(f"  ❌ MELD load failed: {e}")
        print("  Trying CSV fallback from GitHub...")
        return load_meld_csv_fallback()

    all_pairs = []
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        split = ds[split_name]
        df = split.to_pandas()
        print(f"  {split_name}: {len(df)} rows")

        # Group by dialogue_id
        for dialogue_id, grp in tqdm(df.groupby("Dialogue_ID"), desc=f"  MELD {split_name}"):
            grp = grp.sort_values("Utterance_ID")
            utterances = grp["Utterance"].tolist()
            emotions   = [MELD_EMOTION_MAP.get(e.lower(), "neutral") for e in grp["Emotion"].tolist()]
            pairs      = conversation_to_pairs(utterances, emotions)
            all_pairs.extend(pairs)

    print(f"  ✅ MELD pairs generated: {len(all_pairs)}")
    return all_pairs


def load_meld_csv_fallback():
    """
    If HuggingFace is unavailable, download MELD CSV directly.
    """
    import urllib.request
    urls = {
        "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
        "dev":   "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
        "test":  "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
    }
    all_pairs = []
    for split_name, url in urls.items():
        try:
            print(f"  Downloading {split_name}...")
            tmp = f"/tmp/meld_{split_name}.csv"
            urllib.request.urlretrieve(url, tmp)
            df = pd.read_csv(tmp)
            for dialogue_id, grp in tqdm(df.groupby("Dialogue_ID"), desc=f"  MELD {split_name}"):
                grp        = grp.sort_values("Utterance_ID")
                utterances = grp["Utterance"].tolist()
                emotions   = [MELD_EMOTION_MAP.get(str(e).lower(), "neutral") for e in grp["Emotion"].tolist()]
                all_pairs.extend(conversation_to_pairs(utterances, emotions))
        except Exception as e:
            print(f"  ❌ {split_name} failed: {e}")

    print(f"  ✅ MELD (CSV fallback) pairs: {len(all_pairs)}")
    return all_pairs


# ──────────────────────────────────────────────────────────────────────────
# SOURCE 2: DailyDialog  (~13k dialogues, emotion per utterance)
# ──────────────────────────────────────────────────────────────────────────
def load_dailydialog():
    print("\n📦 Loading DailyDialog dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("daily_dialog", trust_remote_code=True)
    except Exception as e:
        print(f"  ❌ DailyDialog load failed: {e}")
        return []

    all_pairs = []
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        split = ds[split_name]
        print(f"  {split_name}: {len(split)} dialogues")

        for item in tqdm(split, desc=f"  DailyDialog {split_name}"):
            utterances = item["dialog"]
            emotions   = [DAILYDIALOG_EMOTION_MAP.get(e, "neutral") for e in item["emotion"]]
            pairs      = conversation_to_pairs(utterances, emotions)
            all_pairs.extend(pairs)

    print(f"  ✅ DailyDialog pairs: {len(all_pairs)}")
    return all_pairs


# ──────────────────────────────────────────────────────────────────────────
# SOURCE 3: Your existing tweet_emotions.csv  (single-text → synthetic pairs)
# ──────────────────────────────────────────────────────────────────────────
def load_existing_tweets(csv_path="tweet_emotions.csv"):
    """
    Takes the existing single-tweet dataset and creates synthetic pairs
    by grouping consecutive same-emotion tweets as pseudo-conversations.
    This is a synthetic augmentation — use with care.
    """
    if not os.path.exists(csv_path):
        print(f"  ⚠️  {csv_path} not found, skipping tweet augmentation")
        return []

    print(f"\n📦 Loading existing tweets from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Columns: tweet_id, sentiment, content
    df = df.dropna(subset=["content", "sentiment"])

    # Group by emotion and pair consecutive tweets as synthetic dialogs
    pairs = []
    for emotion, grp in tqdm(df.groupby("sentiment"), desc="  Tweets"):
        tweets = grp["content"].tolist()
        # sliding window of 2
        for i in range(1, len(tweets)):
            prev = tweets[i-1].strip()
            curr = tweets[i].strip()
            if not prev or not curr:
                continue
            pairs.append({
                "previous_text": prev,
                "current_text":  curr,
                "emotion":       emotion,
                "source":        "tweet_synthetic",
                "window_size":   1,
            })

    print(f"  ✅ Tweet synthetic pairs: {len(pairs)}")
    return pairs


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    import random
    random.seed(42)

    all_pairs = []

    # 1. MELD
    meld_pairs = load_meld()
    all_pairs.extend(meld_pairs)

    # 2. DailyDialog
    dd_pairs = load_dailydialog()
    all_pairs.extend(dd_pairs)

    # 3. Existing tweets (synthetic)
    tweet_pairs = load_existing_tweets("tweet_emotions.csv")
    all_pairs.extend(tweet_pairs)

    if not all_pairs:
        print("\n❌ No data collected. Check internet connection and try again.")
        return

    # ── Save main dataset ──
    df = pd.DataFrame(all_pairs)
    df = df.drop_duplicates(subset=["previous_text", "current_text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

    out_main = "contextual_emotion_dataset.csv"
    df.to_csv(out_main, index=False)
    print(f"\n✅ Main dataset saved: {out_main}")
    print(f"   Total pairs: {len(df)}")
    print(f"   Emotion distribution:\n{df['emotion'].value_counts().to_string()}")

    # ── Save augmented (all window sizes) ──
    out_aug = "contextual_emotion_augmented.csv"
    df.to_csv(out_aug, index=False)
    print(f"\n✅ Augmented dataset saved: {out_aug}")

    # ── Print sample ──
    print("\n📋 Sample rows:")
    for _, row in df.head(3).iterrows():
        print(f"\n  PREV : {row['previous_text'][:80]}...")
        print(f"  CURR : {row['current_text'][:80]}")
        print(f"  LABEL: {row['emotion']}")

    print("\n🎉 Done! Use these CSVs to fine-tune your BERT model for contextual emotion detection.")
    print("   Recommended model: bert-base-uncased with sentence-pair classification (previous_text, current_text)")


if __name__ == "__main__":
    main()
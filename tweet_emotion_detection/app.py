from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path
from flask_cors import CORS   # NEW
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import google.generativeai as genai

app = Flask("Tweet Emotion Detection")
CORS(app)   # NEW

GEMINI_KEY = "AIzaSyBTFc17pWhYuGFD32hUTL-7Sy70QIlxwk4"

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    print("Gemini Connected")
else:
    gemini_model = None
    print("Gemini Not Connected")

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- LOAD MODEL ----------------
tokenizer, model = None, None
try:
    model_dir = Path(__file__).parent / "bert_model"
    model_dir = model_dir.resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.as_posix(),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir.as_posix(),
        local_files_only=True,
        use_safetensors=True,
        output_attentions=True
    ).to(device)

    model.eval()
    print("✅ BERT Loaded Successfully")

except Exception as e:
    print(f"❌ MODEL LOAD FAILED: {e}")

print(model.config.id2label)

# ---------------- AUTO LABELS ----------------

# AFTER model.eval()

if hasattr(model.config, "id2label") and model.config.id2label:
    emotion_labels = list(model.config.id2label.values())
    print("Labels from model:", emotion_labels)
else:
    emotion_labels = ['anger','joy','love','sadness','surprise','neutral']

# ---------------- LABEL MAP ----------------
label_map = {
    "LABEL_0": "anger",
    "LABEL_1": "boredom",
    "LABEL_2": "empty",
    "LABEL_3": "enthusiasm",
    "LABEL_4": "fun",
    "LABEL_5": "happiness",
    "LABEL_6": "hate",
    "LABEL_7": "love",
    "LABEL_8": "neutral",
    "LABEL_9": "relief",
    "LABEL_10": "sadness",
    "LABEL_11": "surprise",
    "LABEL_12": "worry"
}
 
# ---------------- METRICS ----------------
#accuracy_value = 78.09

#try:
#    sample_texts = ["I am happy", "I am sad", "I am angry"]
#    sample_labels = ["joy", "sadness", "anger"]

#    preds = []
#    for t in sample_texts:
#        inputs = tokenizer(t, return_tensors="pt").to(device)
#        with torch.no_grad():
#            out = model(**inputs)
#        idx = torch.argmax(out.logits, dim=1).item()
#        preds.append(emotion_labels[idx])
#
#    accuracy_value = accuracy_score(sample_labels, preds)

#    cm = confusion_matrix(sample_labels, preds, labels=emotion_labels)

#    plt.figure(figsize=(6,5))
#    sns.heatmap(cm, annot=True, fmt='d')
#    plt.xlabel("Predicted")
#    plt.ylabel("Actual")
#    plt.title("Confusion Matrix")
#    plt.savefig("static/confusion_matrix.png")
#    plt.close()

#    print("✅ Metrics Generated")

#except Exception as e:
#    print("⚠ Metrics Error:", e)


# ---------------- XAI FUNCTION ----------------
def get_xai_data(text):
    if not model:
        return []

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions[-1][0].mean(dim=0)
    cls_attention = attentions[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []
    for i, token in enumerate(tokens):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            results.append({
                "token": token.replace("##", ""),
                "score": float(cls_attention[i]),
                "prefix": "" if token.startswith("##") else " "
            })
    return results

# ---------------- LLM EXPLANATION FUNCTION ----------------
def get_llm_free(label, text):
    try:
        if gemini_model:
            prompt = f"""
Explain clearly in 2 short sentences why the emotion is {label}.
Mention important keywords from the sentence.

Text: "{text}"
"""

            resp = gemini_model.generate_content(prompt)
            print("GEMINI RAW:", resp)

            # 🔹 IMPORTANT FIX
            if resp and resp.candidates:
                return resp.candidates[0].content.parts[0].text

    except Exception as e:
        print("Gemini Error:", e)

    # fallback
    return f"The sentence expresses {label} emotion because of emotional keywords and tone."


def get_controlled_explanation(label, text):
    try:
        if gemini_model:
            prompt = f"""
You are an Emotion Detection Assistant.
Analyze emotion carefully and give only 2 line explanation.

Examples:

Text: "I just won tickets to my favorite concert!!!"
Emotion: joy
Explanation: The sentence shows joy because the speaker is excited and happy.

Text: "I miss my old friends so much."
Emotion: love
Explanation: The sentence expresses love due to emotional attachment and affection.

Text: "I am so frustrated with everything today."
Emotion: anger
Explanation: The sentence shows anger because of irritation and negative tone.

Text: "Last day at work, feeling really emotional."
Emotion: sadness
Explanation: The sentence expresses sadness due to emotional farewell.

Text: "Wow I didn’t expect that at all!"
Emotion: surprise
Explanation: The sentence shows surprise due to unexpected reaction.

Text: "I am sitting at home watching TV."
Emotion: neutral
Explanation: The sentence is neutral because it has no strong emotion.

Now analyze:

Text: "{text}"
Emotion: {label}
Explanation:
"""
            
            resp = gemini_model.generate_content(prompt)
            print("GEMINI CONTROLLED RAW:", resp)
            if resp and resp.candidates:
                return resp.candidates[0].content.parts[0].text
        
    except Exception as e:
        print("Gemini Controlled Error:", e)

    return f"This sentence expresses {label} emotion based on tone and keywords."


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("PREDICT ROUTE HIT")

    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    text = None

    # ---- 1. JSON ----
    data = request.get_json(silent=True)
    if data:
        text = data.get("text") or data.get("message") or data.get("input") or data.get("sentence")

    # ---- 2. FORM ----
    if not text:
        text = request.form.get("text")

    # ---- 3. RAW BODY ----
    if not text:
        raw = request.data.decode("utf-8").strip()
        if raw:
            text = raw

    print("TEXT RECEIVED:", text)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # ---------------- MODEL ----------------
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    conf, idx = torch.max(probs, dim=1)

    raw_label = model.config.id2label[idx.item()]
    label = label_map.get(raw_label, raw_label)

    all_probs = {
        label_map.get(model.config.id2label[i], model.config.id2label[i]): float(p)
        for i, p in enumerate(probs[0])
   }



    # -------- GEMINI EXPLANATION --------
    explanation = get_llm_free(label, text)
    controlled_explanation = get_controlled_explanation(label, text)

    return jsonify({
        "emotion": label,
        "confidence": f"{conf.item()*100:.2f}%",
        "attention_map": get_xai_data(text),
        "all_probs": all_probs,
        "explanation": explanation,
        "controlled_explanation": controlled_explanation
    })

#@app.route("/metrics")
#def metrics():
#    return jsonify({
#       "accuracy": accuracy_value,
#        "confusion_matrix": "/static/confusion_matrix.png"
#    })


# ---------------- RUN ----------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
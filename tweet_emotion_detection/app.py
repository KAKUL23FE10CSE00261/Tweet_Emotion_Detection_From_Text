from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path
from flask_cors import CORS
import numpy as np
import google.generativeai as genai

app = Flask("Tweet Emotion Detection")
CORS(app)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- LOAD MODEL ----------------
tokenizer, model = None, None
try:
    model_dir = (Path(__file__).parent / "bert_model").resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix(), local_files_only=True)
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

# ---------------- AUTO LABELS ----------------
if model and hasattr(model.config, "id2label") and model.config.id2label:
    emotion_labels = list(model.config.id2label.values())
else:
    emotion_labels = ['anger', 'joy', 'love', 'sadness', 'surprise', 'neutral']

# ---------------- LABEL MAP ----------------
label_map = {
    "LABEL_0": "anger",     "LABEL_1": "boredom",  "LABEL_2": "empty",
    "LABEL_3": "enthusiasm","LABEL_4": "fun",       "LABEL_5": "happiness",
    "LABEL_6": "hate",      "LABEL_7": "love",      "LABEL_8": "neutral",
    "LABEL_9": "relief",    "LABEL_10": "sadness",  "LABEL_11": "surprise",
    "LABEL_12": "worry"
}

# Emotion → emoji + color
EMOTION_META = {
    "anger":      {"emoji": "😠", "color": "#ef4444"},
    "boredom":    {"emoji": "😑", "color": "#94a3b8"},
    "empty":      {"emoji": "😶", "color": "#64748b"},
    "enthusiasm": {"emoji": "🤩", "color": "#f59e0b"},
    "fun":        {"emoji": "😄", "color": "#22c55e"},
    "happiness":  {"emoji": "😊", "color": "#facc15"},
    "hate":       {"emoji": "🤬", "color": "#dc2626"},
    "love":       {"emoji": "❤️",  "color": "#ec4899"},
    "neutral":    {"emoji": "😐", "color": "#94a3b8"},
    "relief":     {"emoji": "😮‍💨", "color": "#6ee7b7"},
    "sadness":    {"emoji": "😢", "color": "#60a5fa"},
    "surprise":   {"emoji": "😲", "color": "#a78bfa"},
    "worry":      {"emoji": "😟", "color": "#fb923c"},
}


# ================================================================
# XAI HELPERS
# ================================================================

def _get_attention_scores(inputs, outputs):
    """Average CLS attention across all heads of the last layer."""
    last_layer_attn = outputs.attentions[-1]       # [batch, heads, seq, seq]
    cls_attn = last_layer_attn[0].mean(dim=0)[0]   # avg heads → [seq]
    return cls_attn.cpu().detach().numpy()


def _get_gradient_saliency(inputs, target_idx):
    """
    Gradient saliency: L2 norm of input-embedding gradient w.r.t. predicted class.
    Returns numpy array [seq_len] or None.
    """
    try:
        emb_layer = model.bert.embeddings if hasattr(model, "bert") else None
        if emb_layer is None:
            return None

        input_ids      = inputs["input_ids"]
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))
        position_ids   = torch.arange(input_ids.size(1), device=device).unsqueeze(0)

        embeds   = emb_layer.word_embeddings(input_ids).requires_grad_(True)
        pos_emb  = emb_layer.position_embeddings(position_ids)
        tok_emb  = emb_layer.token_type_embeddings(token_type_ids)
        full_emb = emb_layer.LayerNorm(embeds + pos_emb + tok_emb)
        full_emb = emb_layer.dropout(full_emb)

        out   = model(inputs_embeds=full_emb, attention_mask=inputs.get("attention_mask"))
        score = out.logits[0, target_idx]
        score.backward()

        grad     = embeds.grad[0].cpu().detach().numpy()   # [seq, hidden]
        saliency = np.linalg.norm(grad, axis=-1)           # [seq]
        return saliency

    except Exception as e:
        print("Gradient saliency error:", e)
        return None


def _build_token_list(tokens, attn_scores, saliency, token_type_ids_tensor=None):
    """
    Build per-token payload combining attention + saliency into combined_score.
    """
    results  = []
    sep_count = 0
    attn_max = attn_scores.max() or 1.0
    sal_max  = (saliency.max() if saliency is not None else 1.0) or 1.0

    for i, token in enumerate(tokens):
        if token == '[SEP]':
            sep_count += 1
            results.append({
                "token": "[SEP]", "score": 0, "prefix": " ",
                "is_sep": True, "segment": sep_count,
                "saliency": 0, "combined": 0
            })
            continue
        if token in ['[CLS]', '[PAD]']:
            continue

        attn_norm = float(attn_scores[i]) / attn_max
        sal_norm  = float(saliency[i]) / sal_max if saliency is not None else attn_norm
        combined  = 0.5 * attn_norm + 0.5 * sal_norm

        seg = 0
        if token_type_ids_tensor is not None:
            seg = int(token_type_ids_tensor[0][i].item())

        results.append({
            "token":    token.replace("##", ""),
            "score":    attn_norm,
            "saliency": sal_norm,
            "combined": combined,
            "prefix":   "" if token.startswith("##") else " ",
            "is_sep":   False,
            "segment":  seg
        })
    return results


def get_xai_data(text):
    """Standard single-text XAI. Returns (token_list, top_keywords)."""
    if not model:
        return [], []

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_idx  = torch.argmax(outputs.logits, dim=1).item()
    attn_scores = _get_attention_scores(inputs, outputs)
    saliency    = _get_gradient_saliency(inputs, target_idx)
    tokens      = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    results     = _build_token_list(tokens, attn_scores, saliency)

    word_tokens  = [r for r in results if not r["is_sep"]]
    top_keywords = sorted(word_tokens, key=lambda x: x["combined"], reverse=True)[:5]
    return results, top_keywords


def get_xai_data_contextual(previous_text, current_text):
    """Context-aware XAI for Previous [SEP] Current. Returns (token_list, top_keywords)."""
    if not model:
        return [], []

    inputs = tokenizer(previous_text, current_text, return_tensors="pt",
                       truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_idx  = torch.argmax(outputs.logits, dim=1).item()
    attn_scores = _get_attention_scores(inputs, outputs)
    saliency    = _get_gradient_saliency(inputs, target_idx)
    tokens      = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tti         = inputs.get("token_type_ids", None)
    results     = _build_token_list(tokens, attn_scores, saliency, tti)

    word_tokens  = [r for r in results if not r["is_sep"]]
    top_keywords = sorted(word_tokens, key=lambda x: x["combined"], reverse=True)[:5]
    return results, top_keywords


# ================================================================
# XAI EXPLANATION BUILDER  ← WHY this emotion was predicted
# ================================================================

def build_xai_explanation(label, text, top_keywords, confidence,
                           previous_text=None, contextual_mode=False):
    """
    Returns a structured dict that fully explains WHY the emotion was predicted:
      - trigger_words   : top tokens by attention + gradient saliency
      - attention_note  : plain-English note about BERT attention focus
      - gradient_note   : gradient saliency insight
      - context_note    : how previous message influenced (contextual mode only)
      - confidence_note : what the confidence level means
      - llm_why         : Gemini 3-part deep-dive explanation
    """
    kw_labels    = [k["token"] for k in top_keywords]
    emotion_meta = EMOTION_META.get(label, {"emoji": "🎭", "color": "#6366f1"})

    # Attention note
    attention_note = (
        f"BERT's attention focused most on: {', '.join(kw_labels[:3])}. "
        f"These tokens contributed most to the final [{label.upper()}] classification."
    )

    # Gradient saliency note
    has_saliency = any(k.get("saliency", 0) > 0 for k in top_keywords)
    if has_saliency:
        sal_top = sorted(top_keywords, key=lambda x: x.get("saliency", 0), reverse=True)[:2]
        gradient_note = (
            f"Gradient saliency confirms '{sal_top[0]['token']}'"
            + (f" and '{sal_top[1]['token']}'" if len(sal_top) > 1 else "")
            + " as the most decisive words for this prediction."
        )
    else:
        gradient_note = (
            "Attention scores were used as the primary signal "
            "(gradient saliency unavailable for this model configuration)."
        )

    # Context note (contextual mode only)
    context_note = None
    if contextual_mode and previous_text:
        prev_tokens = [k["token"] for k in top_keywords if k.get("segment") == 0]
        curr_tokens = [k["token"] for k in top_keywords if k.get("segment") == 1]
        short_prev  = previous_text[:60] + ("..." if len(previous_text) > 60 else "")
        context_note = (
            f"In contextual mode, BERT read the previous message (\"{short_prev}\") as background. "
        )
        if prev_tokens:
            context_note += f"Key context words from the previous message: {', '.join(prev_tokens)}. "
        if curr_tokens:
            context_note += f"Key trigger words from the current message: {', '.join(curr_tokens)}."

    # Confidence note
    conf_val = float(confidence.replace("%", ""))
    if conf_val >= 85:
        confidence_note = f"Very high confidence ({confidence}) — the model is strongly certain about this prediction."
    elif conf_val >= 65:
        confidence_note = f"Moderate confidence ({confidence}) — the prediction is likely correct but some ambiguity exists."
    else:
        confidence_note = f"Lower confidence ({confidence}) — the text may contain mixed or subtle emotional signals."

    # Gemini deep-dive
    llm_why = _get_gemini_xai_why(label, text, kw_labels, previous_text, contextual_mode)

    return {
        "emotion":         label,
        "emoji":           emotion_meta["emoji"],
        "color":           emotion_meta["color"],
        "trigger_words":   kw_labels,
        "attention_note":  attention_note,
        "gradient_note":   gradient_note,
        "context_note":    context_note,
        "confidence_note": confidence_note,
        "llm_why":         llm_why,
    }


def _get_gemini_xai_why(label, text, top_keywords, previous_text, contextual_mode):
    """Ask Gemini for a structured 3-part WHY explanation."""
    try:
        if not gemini_model:
            raise ValueError("No Gemini model")

        kw_str    = ", ".join(f'"{w}"' for w in top_keywords)
        ctx_block = ""
        if contextual_mode and previous_text:
            ctx_block = f'\nPrevious message (context): "{previous_text}"'

        prompt = f"""You are an expert NLP explainability assistant.

A BERT model predicted the emotion "{label}" for the following text.
The model's attention and gradient saliency highlighted these key words: {kw_str}.{ctx_block}

Text: "{text}"

Explain in exactly 3 parts, each 1-2 sentences:
1. LINGUISTIC REASON: What words, phrases, or tone signal "{label}"?
2. BERT ATTENTION INSIGHT: Why would BERT focus on those specific keywords ({kw_str})?
3. EMOTIONAL LOGIC: How does the overall context confirm "{label}" as the right emotion?

Format your answer like this (use these exact labels):
LINGUISTIC REASON: ...
BERT ATTENTION INSIGHT: ...
EMOTIONAL LOGIC: ...
"""
        resp = gemini_model.generate_content(prompt)
        if resp and resp.candidates:
            return resp.candidates[0].content.parts[0].text.strip()

    except Exception as e:
        print("Gemini XAI WHY error:", e)

    return (
        f"LINGUISTIC REASON: The words {', '.join(top_keywords[:3])} carry strong {label} signals.\n"
        f"BERT ATTENTION INSIGHT: BERT focused on these tokens because they are emotionally charged in the context of {label}.\n"
        f"EMOTIONAL LOGIC: The overall tone and vocabulary of the sentence align with {label} emotion."
    )


# ================================================================
# LLM EXPLANATION FUNCTIONS (original, preserved)
# ================================================================

def get_llm_free(label, text, previous_text=None):
    try:
        if gemini_model:
            if previous_text:
                prompt = (
                    f'You are analyzing an emotion in a conversation.\n'
                    f'Previous message: "{previous_text}"\nCurrent message: "{text}"\n'
                    f'Detected emotion: {label}\n'
                    f'Explain clearly in 2 short sentences why the current message expresses {label}. '
                    f'Consider how the previous message provides context. Mention important keywords.'
                )
            else:
                prompt = (
                    f'Explain clearly in 2 short sentences why the emotion is {label}. '
                    f'Mention important keywords from the sentence.\nText: "{text}"'
                )
            resp = gemini_model.generate_content(prompt)
            if resp and resp.candidates:
                return resp.candidates[0].content.parts[0].text
    except Exception as e:
        print("Gemini Error:", e)
    return f"The sentence expresses {label} emotion because of emotional keywords and tone."


def get_controlled_explanation(label, text, previous_text=None):
    try:
        if gemini_model:
            ctx = f'Previous message: "{previous_text}"\n' if previous_text else ""
            prompt = f"""You are an Emotion Detection Assistant.
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

Now analyze:
{ctx}Text: "{text}"
Emotion: {label}
Explanation:"""
            resp = gemini_model.generate_content(prompt)
            if resp and resp.candidates:
                return resp.candidates[0].content.parts[0].text
    except Exception as e:
        print("Gemini Controlled Error:", e)
    return f"This sentence expresses {label} emotion based on tone and keywords."


# ================================================================
# ROUTES
# ================================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print("PREDICT ROUTE HIT")

    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    text, previous_text = None, None

    data = request.get_json(silent=True)
    if data:
        text          = data.get("text") or data.get("message") or data.get("input") or data.get("sentence")
        previous_text = data.get("previous_text") or data.get("context") or None

    if not text:
        text = request.form.get("text")
    if not previous_text:
        previous_text = request.form.get("previous_text") or None

    if not text:
        raw = request.data.decode("utf-8").strip()
        if raw:
            text = raw

    print("TEXT:", text)
    print("PREV:", previous_text)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    contextual_mode = bool(previous_text and previous_text.strip())

    # ── Tokenise ──
    if contextual_mode:
        inputs = tokenizer(previous_text, text, return_tensors="pt",
                           truncation=True, padding=True, max_length=512).to(device)
        print("🔗 Contextual mode")
    else:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding=True).to(device)
        print("📝 Standard mode")

    with torch.no_grad():
        outputs = model(**inputs)

    probs     = F.softmax(outputs.logits, dim=1)
    conf, idx = torch.max(probs, dim=1)

    raw_label = model.config.id2label[idx.item()]
    label     = label_map.get(raw_label, raw_label)

    all_probs = {
        label_map.get(model.config.id2label[i], model.config.id2label[i]): float(p)
        for i, p in enumerate(probs[0])
    }

    confidence_str = f"{conf.item()*100:.2f}%"

    # ── XAI ──
    if contextual_mode:
        attention_map, top_keywords = get_xai_data_contextual(previous_text, text)
    else:
        attention_map, top_keywords = get_xai_data(text)

    # ── Structured XAI WHY Explanation ──
    xai_explanation = build_xai_explanation(
        label, text, top_keywords, confidence_str,
        previous_text=previous_text, contextual_mode=contextual_mode
    )

    # ── LLM Explanations (original, preserved) ──
    explanation            = get_llm_free(label, text, previous_text if contextual_mode else None)
    controlled_explanation = get_controlled_explanation(label, text, previous_text if contextual_mode else None)

    return jsonify({
        "emotion":                label,
        "confidence":             confidence_str,
        "attention_map":          attention_map,
        "all_probs":              all_probs,
        "explanation":            explanation,
        "controlled_explanation": controlled_explanation,
        "xai_explanation":        xai_explanation,   # ← NEW structured WHY block
        "contextual_mode":        contextual_mode,
    })


# ── RUN ──
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
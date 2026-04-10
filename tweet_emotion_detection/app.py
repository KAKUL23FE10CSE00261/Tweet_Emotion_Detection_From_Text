from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path
from flask_cors import CORS
import numpy as np
try:
    from google import genai as genai
    _GENAI_NEW = True
except ImportError:
    try:
        import google.generativeai as _old_genai
        _GENAI_NEW = False
    except ImportError:
        _old_genai = None
        _GENAI_NEW = False
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask("Tweet Emotion Detection")
CORS(app)

# ===== BERT =====
def _model_weights_exist(folder):
    """Check if model weights are present (not just tokenizer/config files)."""
    weight_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5",
                    "model.ckpt.index", "flax_model.msgpack"]
    return any(os.path.exists(os.path.join(folder, f)) for f in weight_files)

if not os.path.exists("bert_model") or not _model_weights_exist("bert_model"):
    print("Downloading BERT model...")

    # delete old folder if exists (may have partial/tokenizer-only files)
    os.system("rm -rf bert_model bert_model.zip")

    import gdown
    gdown.download(
        "https://drive.google.com/uc?export=download&id=1apSKRaK5zs5Hy93cUnp6HTDYld3uMzKO",
        "bert_model.zip",
        quiet=False
    )

    os.system("unzip -o bert_model.zip -d .")
    
# ===== ROBERTA =====
if not os.path.exists("bert_contextual_model") or not _model_weights_exist("bert_contextual_model"):
    print("Downloading ROBERTA model...")
    gdown.download(id="1zHcKqWcQ2E1Tcq1kEvbsTNtR0GLkFEAU", output="roberta.zip", quiet=False)
    os.system("unzip -o roberta.zip")

# DEBUG
print("Files in root:", os.listdir())
print("Files in bert_model:", os.listdir("bert_model") if os.path.exists("bert_model") else "NOT FOUND")

# ================================================================
# GEMINI SETUP
# ================================================================
GEMINI_KEY = "AIzaSyBTFc17pWhYuGFD32hUTL-7Sy70QIlxwk4"

gemini_client = None
gemini_model_name = "gemini-2.0-flash"

try:
    if GEMINI_KEY:
        if _GENAI_NEW:
            gemini_client = genai.Client(api_key=GEMINI_KEY)
            print("✅ Gemini Connected (new SDK)")
        else:
            _old_genai.configure(api_key=GEMINI_KEY)
            gemini_client = _old_genai.GenerativeModel(gemini_model_name)
            print("✅ Gemini Connected (legacy SDK)")
    else:
        print("❌ Gemini Not Connected")
except Exception as e:
    gemini_client = None
    print(f"❌ Gemini setup failed: {e}")

def _gemini_generate(prompt):
    """Call Gemini regardless of SDK version."""
    if not gemini_client:
        return None
    try:
        if _GENAI_NEW:
            resp = gemini_client.models.generate_content(model=gemini_model_name, contents=prompt)
            return resp.text if resp else None
        else:
            resp = gemini_client.generate_content(prompt)
            if resp and resp.candidates:
                return resp.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"Gemini call error: {e}")
    return None


# ================================================================
# DEVICE
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  Using device: {device}")

# ================================================================
# LAZY MODEL LOADING — saves RAM at startup.
# Models load on first request, not at boot.
# float16 only used on GPU — CPU does NOT support float16 for most ops.
# ================================================================
bert_tokenizer, bert_model = None, None
roberta_tokenizer, roberta_model = None, None
_bert_loaded = False
_roberta_loaded = False

# Always use float32 — float16 on CPU does NOT support backward() (gradient saliency crashes)
_dtype = torch.float32

def _load_bert():
    global bert_tokenizer, bert_model, _bert_loaded
    if _bert_loaded:
        return
    try:
        bert_dir = (Path(__file__).parent / "bert_model").resolve()
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir.as_posix(), local_files_only=True)
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            bert_dir.as_posix(), local_files_only=True,
            output_attentions=True, torch_dtype=_dtype
        ).to(device)
        bert_model.eval()
        _bert_loaded = True
        print("✅ Original BERT Loaded  →  bert_model/")
    except Exception as e:
        print(f"❌ BERT LOAD FAILED: {e}")

def _load_roberta():
    global roberta_tokenizer, roberta_model, _roberta_loaded
    if _roberta_loaded:
        return
    try:
        roberta_dir = (Path(__file__).parent / "bert_contextual_model").resolve()
        roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_dir.as_posix(), local_files_only=True)
        roberta_model = AutoModelForSequenceClassification.from_pretrained(
            roberta_dir.as_posix(), local_files_only=True,
            output_attentions=True, torch_dtype=_dtype
        ).to(device)
        roberta_model.eval()
        _roberta_loaded = True
        print("✅ RoBERTa Contextual Loaded  →  bert_contextual_model/")
    except Exception as e:
        print(f"❌ ROBERTA LOAD FAILED: {e}")

# ================================================================
# HELPER — pick correct model/tokenizer
# ================================================================
def _get_model(contextual=False):
    if contextual:
        _load_roberta()
        if roberta_model is not None:
            return roberta_tokenizer, roberta_model, "roberta"
        _load_bert()
        if bert_model is not None:
            print("⚠️  RoBERTa not loaded, falling back to BERT")
            return bert_tokenizer, bert_model, "bert"
    else:
        _load_bert()
        if bert_model is not None:
            return bert_tokenizer, bert_model, "bert"
        _load_roberta()
        if roberta_model is not None:
            print("⚠️  BERT not loaded, falling back to RoBERTa")
            return roberta_tokenizer, roberta_model, "roberta"
    return None, None, None

# ================================================================
# LABELS
# ================================================================
label_map = {
    "LABEL_0":  "anger",      "LABEL_1":  "boredom",  "LABEL_2":  "empty",
    "LABEL_3":  "enthusiasm", "LABEL_4":  "fun",      "LABEL_5":  "happiness",
    "LABEL_6":  "hate",       "LABEL_7":  "love",     "LABEL_8":  "neutral",
    "LABEL_9":  "relief",     "LABEL_10": "sadness",  "LABEL_11": "surprise",
    "LABEL_12": "worry"
}

EMOTION_META = {
    "anger":      {"emoji": "😠",  "color": "#ef4444"},
    "boredom":    {"emoji": "😑",  "color": "#94a3b8"},
    "empty":      {"emoji": "😶",  "color": "#64748b"},
    "enthusiasm": {"emoji": "🤩",  "color": "#f59e0b"},
    "fun":        {"emoji": "😄",  "color": "#22c55e"},
    "happiness":  {"emoji": "😊",  "color": "#facc15"},
    "hate":       {"emoji": "🤬",  "color": "#dc2626"},
    "love":       {"emoji": "❤️",  "color": "#ec4899"},
    "neutral":    {"emoji": "😐",  "color": "#94a3b8"},
    "relief":     {"emoji": "😮‍💨", "color": "#6ee7b7"},
    "sadness":    {"emoji": "😢",  "color": "#60a5fa"},
    "surprise":   {"emoji": "😲",  "color": "#a78bfa"},
    "worry":      {"emoji": "😟",  "color": "#fb923c"},
}

# ================================================================
# CONTEXT TYPE META — UI mein dikhega
# ================================================================
CONTEXT_TYPE_META = {
    "emotion_shift": {
        "label":       "Emotion Shift Detected",
        "description": "Emotion changed from previous to current message",
        "icon":        "🔄",
        "color":       "#f59e0b",
        "badge":       "SHIFT"
    },
    "same_emotion": {
        "label":       "Emotion Consistent",
        "description": "Emotion remained same across the conversation",
        "icon":        "➡️",
        "color":       "#22c55e",
        "badge":       "SAME"
    },
    "window3": {
        "label":       "Multi-turn Context",
        "description": "Emotion influenced by multi-message conversation pattern",
        "icon":        "💬",
        "color":       "#6366f1",
        "badge":       "CONV"
    },
    "single": {
        "label":       "Single Message",
        "description": "No previous context — analyzed independently",
        "icon":        "📝",
        "color":       "#94a3b8",
        "badge":       "SOLO"
    }
}

def _resolve_label(model_obj, idx):
    raw = model_obj.config.id2label.get(idx, f"LABEL_{idx}")
    return label_map.get(raw, raw)


def _quick_predict(tok, mdl, text):
    """Single text ka emotion quickly predict karo (context type detection ke liye)."""
    try:
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = mdl(**inputs)
        idx = torch.argmax(outputs.logits, dim=1).item()
        return _resolve_label(mdl, idx)
    except Exception:
        return "unknown"


def _detect_context_type_single(prev_emotion, curr_emotion, has_previous):
    """
    Tab 2 ke liye — sirf prev + current compare karo.
    """
    if not has_previous:
        return "single"
    if prev_emotion == curr_emotion:
        return "same_emotion"
    return "emotion_shift"


def _detect_context_type(history_emotions):
    """
    Tab 3 ke liye — puri conversation ki emotion timeline se context type detect karo.

    Rules:
      - 1 message only          → "single"
      - 3+ messages, shift hua  → "emotion_shift"
      - 3+ messages, same raha  → "window3"  (multi-turn stable)
      - 2 messages, same        → "same_emotion"
      - 2 messages, different   → "emotion_shift"
    """
    if not history_emotions or len(history_emotions) <= 1:
        return "single"

    emotions = [e["emotion"] for e in history_emotions
                if e.get("emotion") and e["emotion"] != "unknown"]

    if len(emotions) < 2:
        return "single"

    has_shift = any(emotions[i] != emotions[i + 1] for i in range(len(emotions) - 1))

    if len(emotions) >= 3:
        return "emotion_shift" if has_shift else "window3"

    # Exactly 2
    return "emotion_shift" if has_shift else "same_emotion"


# ================================================================
# JSON SAFETY HELPER
# ================================================================
def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj

# ================================================================
# XAI HELPERS
# ================================================================
def _get_attention_scores(outputs):
    last_layer_attn = outputs.attentions[-1]
    cls_attn        = last_layer_attn[0].mean(dim=0)[0]
    return cls_attn.cpu().detach().numpy()


def _get_gradient_saliency(inputs, target_idx, model_obj, arch):
    try:
        if arch == "roberta":
            emb_layer = model_obj.roberta.embeddings
        elif arch == "bert":
            emb_layer = model_obj.bert.embeddings
        else:
            return np.zeros(inputs["input_ids"].shape[1])

        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))
        position_ids   = torch.arange(input_ids.size(1), device=device).unsqueeze(0)

        word_emb = emb_layer.word_embeddings(input_ids).float()
        word_emb.requires_grad_(True)
        word_emb.retain_grad()

        pos_emb  = emb_layer.position_embeddings(position_ids).float()
        tok_emb  = emb_layer.token_type_embeddings(token_type_ids).float()
        full_emb = emb_layer.LayerNorm(word_emb + pos_emb + tok_emb)
        full_emb = emb_layer.dropout(full_emb)

        out   = model_obj(inputs_embeds=full_emb, attention_mask=attention_mask)
        score = out.logits[0, target_idx]

        model_obj.zero_grad()
        score.backward()

        if word_emb.grad is None:
            return np.zeros(input_ids.shape[1])

        return word_emb.grad[0].norm(dim=-1).cpu().detach().numpy()

    except Exception as e:
        print(f"Gradient saliency error ({arch}):", e)
        return np.zeros(inputs["input_ids"].shape[1])


def _build_token_list(tokens, attn_scores, saliency, token_type_ids_tensor=None):
    results   = []
    sep_count = 0
    attn_max  = float(attn_scores.max()) or 1.0
    sal_max   = (float(saliency.max()) if saliency is not None else 1.0) or 1.0

    for i, token in enumerate(tokens):
        if token in ['[SEP]', '</s>']:
            sep_count += 1
            results.append({
                "token": "[SEP]", "score": 0.0, "prefix": " ",
                "is_sep": True, "segment": sep_count,
                "saliency": 0.0, "combined": 0.0
            })
            continue
        if token in ['[CLS]', '[PAD]', '<s>', '<pad>', '<mask>']:
            continue

        attn_norm = float(attn_scores[i]) / attn_max
        sal_norm  = float(saliency[i]) / sal_max if saliency is not None else attn_norm
        combined  = 0.5 * attn_norm + 0.5 * sal_norm

        seg = 0
        if token_type_ids_tensor is not None:
            try:
                seg = int(token_type_ids_tensor[0][i].item())
            except Exception:
                seg = 0

        is_bert_subword = token.startswith("##")
        clean_token     = token.replace("##", "").lstrip("Ġ")
        prefix          = "" if is_bert_subword else " "

        results.append({
            "token":    clean_token,
            "score":    float(attn_norm),
            "saliency": float(sal_norm),
            "combined": float(combined),
            "prefix":   prefix,
            "is_sep":   False,
            "segment":  int(seg)
        })
    return results


def _run_xai(tokenizer_obj, model_obj, arch, text_a, text_b=None):
    if model_obj is None:
        return [], []

    if text_b:
        inputs = tokenizer_obj(
            text_a, text_b,
            return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
    else:
        inputs = tokenizer_obj(
            text_a, return_tensors="pt", truncation=True, padding=True
        ).to(device)

    with torch.no_grad():
        outputs = model_obj(**inputs)

    target_idx   = torch.argmax(outputs.logits, dim=1).item()
    attn_scores  = _get_attention_scores(outputs)
    saliency     = _get_gradient_saliency(inputs, target_idx, model_obj, arch)
    tokens       = tokenizer_obj.convert_ids_to_tokens(inputs["input_ids"][0])
    tti          = inputs.get("token_type_ids", None)
    results      = _build_token_list(tokens, attn_scores, saliency, tti)
    word_tokens  = [r for r in results if not r["is_sep"]]
    top_keywords = sorted(word_tokens, key=lambda x: x["combined"], reverse=True)[:5]
    return results, top_keywords

# ================================================================
# XAI EXPLANATION BUILDER
# ================================================================
def build_xai_explanation(label, text, top_keywords, confidence,
                           previous_text=None, contextual_mode=False,
                           context_type=None):
    kw_labels    = [k["token"] for k in top_keywords]
    emotion_meta = EMOTION_META.get(label, {"emoji": "🎭", "color": "#6366f1"})

    attention_note = (
        f"Model's attention focused most on: {', '.join(kw_labels[:3])}. "
        f"These tokens contributed most to the [{label.upper()}] classification."
    )

    has_saliency = any(k.get("saliency", 0) > 0 for k in top_keywords)
    if has_saliency:
        sal_top = sorted(top_keywords, key=lambda x: x.get("saliency", 0), reverse=True)[:2]
        gradient_note = (
            f"Gradient saliency confirms '{sal_top[0]['token']}'"
            + (f" and '{sal_top[1]['token']}'" if len(sal_top) > 1 else "")
            + " as the most decisive words."
        )
    else:
        gradient_note = "Attention scores used as primary signal (gradient saliency unavailable)."

    context_note = None
    if contextual_mode and previous_text:
        prev_tokens = [k["token"] for k in top_keywords if k.get("segment") == 0]
        curr_tokens = [k["token"] for k in top_keywords if k.get("segment") == 1]
        short_prev  = previous_text[:60] + ("..." if len(previous_text) > 60 else "")
        context_note = f"Model read previous message (\"{short_prev}\") as context. "
        if prev_tokens:
            context_note += f"Context trigger words: {', '.join(prev_tokens)}. "
        if curr_tokens:
            context_note += f"Current trigger words: {', '.join(curr_tokens)}."

    conf_val = float(confidence.replace("%", ""))
    if conf_val >= 85:
        confidence_note = f"Very high confidence ({confidence}) — model is strongly certain."
    elif conf_val >= 65:
        confidence_note = f"Moderate confidence ({confidence}) — likely correct, some ambiguity."
    else:
        confidence_note = f"Lower confidence ({confidence}) — mixed emotional signals possible."

    llm_why  = _get_gemini_xai_why(label, text, kw_labels, previous_text, contextual_mode)
    ctx_meta = CONTEXT_TYPE_META.get(context_type or "single", CONTEXT_TYPE_META["single"])

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
        "context_type":    ctx_meta,   # ← NEW: context type in XAI panel bhi
    }


def _get_gemini_xai_why(label, text, top_keywords, previous_text, contextual_mode):
    try:
        if not gemini_client:
            raise ValueError("No Gemini")
        kw_str    = ", ".join(f'"{w}"' for w in top_keywords)
        ctx_block = f'\nPrevious message: "{previous_text}"' if (contextual_mode and previous_text) else ""
        prompt = f"""You are an expert NLP explainability assistant.
A model predicted the emotion "{label}" for the text below.
Key words highlighted: {kw_str}.{ctx_block}
Text: "{text}"

Explain in exactly 3 parts (1-2 sentences each):
LINGUISTIC REASON: ...
MODEL ATTENTION INSIGHT: ...
EMOTIONAL LOGIC: ...
"""
        resp_text = _gemini_generate(prompt)
        if resp_text:
            return resp_text.strip()
    except Exception as e:
        print("Gemini XAI error:", e)
    return (
        f"LINGUISTIC REASON: The words {', '.join(top_keywords[:3])} carry strong {label} signals.\n"
        f"MODEL ATTENTION INSIGHT: These tokens are emotionally charged in this context.\n"
        f"EMOTIONAL LOGIC: Overall tone and vocabulary align with {label}."
    )

# ================================================================
# LLM EXPLANATION FUNCTIONS
# ================================================================
def get_llm_free(label, text, previous_text=None):
    try:
        if gemini_client:
            if previous_text:
                prompt = (
                    f'Analyze this conversation.\nPrevious: "{previous_text}"\nCurrent: "{text}"\n'
                    f'Emotion: {label}\nExplain in 2 sentences why current message expresses {label}, '
                    f'considering context. Mention key words.'
                )
            else:
                prompt = (
                    f'Explain in 2 sentences why emotion is {label}. '
                    f'Mention key words.\nText: "{text}"'
                )
            resp_text = _gemini_generate(prompt)
            if resp_text:
                return resp_text
    except Exception as e:
        print("Gemini Error:", e)
    return f"The sentence expresses {label} emotion based on emotional keywords and tone."


def get_controlled_explanation(label, text, previous_text=None):
    try:
        if gemini_client:
            ctx = f'Previous: "{previous_text}"\n' if previous_text else ""
            prompt = f"""You are an Emotion Detection Assistant. Give only 2 line explanation.

Text: "I just won tickets to my favorite concert!!!"
Emotion: joy
Explanation: Shows joy because the speaker is excited and happy.

Text: "I miss my old friends so much."
Emotion: love
Explanation: Expresses love due to emotional attachment and affection.

Now analyze:
{ctx}Text: "{text}"
Emotion: {label}
Explanation:"""
            resp_text = _gemini_generate(prompt)
            if resp_text:
                return resp_text
    except Exception as e:
        print("Gemini Controlled Error:", e)
    return f"Expresses {label} emotion based on tone and keywords."

# ================================================================
# CONVERSATION HELPERS
# ================================================================
def _build_rolling_context(history, max_chars=1600):
    if not history or len(history) < 2:
        return None
    parts = []
    for msg in reversed(history[:-1]):
        text = msg.get("text", "").strip()
        line = f"[{msg.get('role','user').upper()}]: {text}"
        if len(" | ".join(parts + [line])) > max_chars:
            break
        parts.insert(0, line)
    return " | ".join(parts) if parts else None


def _get_gemini_conv_explanation(label, current_text, history, top_keywords, context_type):
    try:
        if not gemini_client:
            raise ValueError("No Gemini")
        history_str = "\n".join(
            f"  [{m.get('role','user').upper()}]: {m.get('text','')}"
            for m in history[:-1][-5:]
        )
        kw_str = ", ".join(f'"{w}"' for w in top_keywords)

        # Context type aware hint for Gemini
        ctx_hint = {
            "emotion_shift": "Note: The emotion has SHIFTED from the previous message. Explain this shift clearly.",
            "same_emotion":  "Note: The emotion remained CONSISTENT throughout the conversation.",
            "window3":       "Note: This is a multi-turn conversation. Emotion is shaped by multiple prior messages.",
            "single":        ""
        }.get(context_type, "")

        prompt = f"""You are an expert in conversational emotion analysis.

Conversation history:
{history_str}

Current message: "{current_text}"
Detected emotion: {label}
Key words: {kw_str}
{ctx_hint}

Explain in exactly 3 parts:
CONVERSATION CONTEXT: ...
CURRENT TRIGGER: ...
EMOTION SHIFT: ..."""
        resp_text = _gemini_generate(prompt)
        if resp_text:
            return resp_text.strip()
    except Exception as e:
        print("Gemini conv error:", e)
    return (
        f"CONVERSATION CONTEXT: Previous messages provide emotional background.\n"
        f"CURRENT TRIGGER: Words {', '.join(top_keywords[:3])} signal {label}.\n"
        f"EMOTION SHIFT: Emotion analyzed relative to full conversation context."
    )

# ================================================================
# ROUTES
# ================================================================
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------------------------------------------
# /predict — Tab 1 (Single Text) → BERT | Tab 2 (Contextual) → RoBERTa
# ----------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    print("\n── /predict ──")

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
    if not text:
        return jsonify({"error": "No text provided"}), 400

    contextual_mode = bool(previous_text and previous_text.strip())
    tok, mdl, arch  = _get_model(contextual=contextual_mode)
    if mdl is None:
        return jsonify({"error": "No model loaded"}), 500

    print(f"   Mode: {'Contextual → RoBERTa' if contextual_mode else 'Standard → BERT'}")

    if contextual_mode:
        inputs = tok(
            previous_text, text,
            return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
    else:
        inputs = tok(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(device)

    with torch.no_grad():
        outputs = mdl(**inputs)

    probs          = F.softmax(outputs.logits, dim=1)
    conf, idx      = torch.max(probs, dim=1)
    label          = _resolve_label(mdl, idx.item())
    all_probs      = {_resolve_label(mdl, i): float(p) for i, p in enumerate(probs[0])}
    confidence_str = f"{float(conf.item())*100:.2f}%"

    # ── Context type detection (Tab 2) ──
    context_type = "single"
    prev_emotion = None
    if contextual_mode and previous_text:
        prev_emotion = _quick_predict(tok, mdl, previous_text)
        context_type = _detect_context_type_single(prev_emotion, label, has_previous=True)

    ctx_meta = CONTEXT_TYPE_META.get(context_type, CONTEXT_TYPE_META["single"])

    attention_map, top_keywords = _run_xai(
        tok, mdl, arch,
        text_a=previous_text if contextual_mode else text,
        text_b=text if contextual_mode else None
    )

    xai_explanation        = build_xai_explanation(label, text, top_keywords, confidence_str,
                                                    previous_text=previous_text,
                                                    contextual_mode=contextual_mode,
                                                    context_type=context_type)
    explanation            = get_llm_free(label, text, previous_text if contextual_mode else None)
    controlled_explanation = get_controlled_explanation(label, text, previous_text if contextual_mode else None)

    response = _to_json_safe({
        "emotion":                label,
        "confidence":             confidence_str,
        "attention_map":          attention_map,
        "all_probs":              all_probs,
        "explanation":            explanation,
        "controlled_explanation": controlled_explanation,
        "xai_explanation":        xai_explanation,
        "contextual_mode":        contextual_mode,
        "model_used":             "RoBERTa (bert_contextual_model/)" if contextual_mode else "BERT (bert_model/)",
        # ── NEW fields ──
        "context_type":           context_type,
        "context_type_meta":      ctx_meta,
        "previous_emotion":       prev_emotion,
    })
    return jsonify(response)


# ----------------------------------------------------------------
# /predict_conversation — Tab 3 (Conversation History) → RoBERTa
# ----------------------------------------------------------------
@app.route("/predict_conversation", methods=["POST"])
def predict_conversation():
    print("\n── /predict_conversation ──")

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    history = data.get("history", [])
    if not history or not isinstance(history, list):
        return jsonify({"error": "history must be a non-empty list"}), 400

    current_text = history[-1].get("text", "").strip()
    if not current_text:
        return jsonify({"error": "Last history item must have non-empty 'text'"}), 400

    tok, mdl, arch = _get_model(contextual=True)
    if mdl is None:
        return jsonify({"error": "No model loaded"}), 500

    rolling_context = _build_rolling_context(history)
    contextual_mode = bool(rolling_context)

    print(f"   Messages: {len(history)}, Context: {'YES' if contextual_mode else 'NO'}")

    if contextual_mode:
        inputs = tok(
            rolling_context, current_text,
            return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
    else:
        inputs = tok(
            current_text, return_tensors="pt", truncation=True, padding=True
        ).to(device)

    with torch.no_grad():
        outputs = mdl(**inputs)

    probs          = F.softmax(outputs.logits, dim=1)
    conf, idx      = torch.max(probs, dim=1)
    label          = _resolve_label(mdl, idx.item())
    all_probs      = {_resolve_label(mdl, i): float(p) for i, p in enumerate(probs[0])}
    confidence_str = f"{float(conf.item())*100:.2f}%"

    attention_map, top_keywords = _run_xai(
        tok, mdl, arch,
        text_a=rolling_context if contextual_mode else current_text,
        text_b=current_text if contextual_mode else None
    )

    # ── Per-message emotion timeline ──
    history_emotions = []
    for msg in history:
        msg_text = msg.get("text", "").strip()
        if not msg_text:
            continue
        try:
            msg_inputs = tok(
                msg_text, return_tensors="pt", truncation=True, padding=True
            ).to(device)
            with torch.no_grad():
                msg_outputs = mdl(**msg_inputs)
            msg_probs         = F.softmax(msg_outputs.logits, dim=1)
            msg_conf, msg_idx = torch.max(msg_probs, dim=1)
            msg_label         = _resolve_label(mdl, msg_idx.item())
            history_emotions.append({
                "text":       msg_text,
                "role":       msg.get("role", "user"),
                "emotion":    msg_label,
                "confidence": f"{float(msg_conf.item())*100:.1f}%",
                "emoji":      EMOTION_META.get(msg_label, {}).get("emoji", "🎭"),
                "color":      EMOTION_META.get(msg_label, {}).get("color", "#6366f1"),
            })
        except Exception as e:
            print(f"Per-message error: {e}")
            history_emotions.append({
                "text": msg_text, "role": msg.get("role", "user"),
                "emotion": "unknown", "confidence": "0%",
                "emoji": "❓", "color": "#64748b"
            })

    # ── Context type detect karo emotion timeline se ──
    context_type = _detect_context_type(history_emotions)
    ctx_meta     = CONTEXT_TYPE_META.get(context_type, CONTEXT_TYPE_META["single"])

    xai_explanation  = build_xai_explanation(label, current_text, top_keywords, confidence_str,
                                              previous_text=rolling_context,
                                              contextual_mode=contextual_mode,
                                              context_type=context_type)
    kw_labels        = [k["token"] for k in top_keywords]
    conv_explanation = _get_gemini_conv_explanation(label, current_text, history, kw_labels, context_type)

    response = _to_json_safe({
        "emotion":              label,
        "confidence":           confidence_str,
        "all_probs":            all_probs,
        "attention_map":        attention_map,
        "xai_explanation":      xai_explanation,
        "conv_explanation":     conv_explanation,
        "history_emotions":     history_emotions,
        "contextual_mode":      contextual_mode,
        "rolling_context_used": rolling_context,
        "model_used":           "RoBERTa (bert_contextual_model/)",
        # ── NEW fields ──
        "context_type":         context_type,
        "context_type_meta":    ctx_meta,
    })
    return jsonify(response)


# ── RUN ──
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

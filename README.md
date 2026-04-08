# 🧠 Tweet Emotion Detection using BERT with Contextual Learning

## 📌 Overview

This project focuses on detecting emotions from tweets using **BERT (Bidirectional Encoder Representations from Transformers)**.
Unlike basic models, this project incorporates **context-aware learning strategies** to improve emotion classification accuracy.

The model is trained in **three different stages**, each capturing different contextual patterns in text.

---

## 🚀 Key Features

* Emotion classification from tweet text
* Context-aware prediction using multiple strategies
* Implementation of **multi-stage training pipeline**
* Interactive web interface (Flask / Streamlit)
* Modular and scalable ML pipeline

---

## 🧠 Model Training Strategy (Core Highlight)

The model is trained in **3 stages**:

### 🔹 Stage 1: Same Emotion Training

* Trained on tweets with **consistent emotional labels**
* Helps model learn **basic emotion patterns**

### 🔹 Stage 2: Window-Based Context (Window Size = 3)

* Considers neighboring tweets (context window)
* Improves understanding of **contextual flow**

### 🔹 Stage 3: Emotion Shift Detection

* Focuses on tweets where emotion changes
* Helps model capture **dynamic emotional transitions**

---

## 🛠️ Tech Stack

* Python
* BERT (Transformers)
* PyTorch
* Pandas, NumPy
* Matplotlib / Seaborn (EDA)
* Flask / Streamlit (Frontend)

---

## 📂 Project Structure

```
tweet_emotion_detection/
│
├── train_stage1_same_emotion.py
├── train_stage2_window3.py
├── train_stage3_emotion_shift.py
├── generate_contextual_dataset.py
├── app.py
│
├── templates/
├── static/
│
├── contextual_emotion_dataset.csv
├── same_emotion.csv
├── window_3.csv
├── emotion_shift.csv
```

---

## ⚙️ How It Works

1. User inputs tweet text
2. Text is preprocessed and tokenized
3. Passed through trained BERT model
4. Emotion is predicted and displayed

---

## 📊 Applications

* Social media sentiment analysis
* Customer feedback analysis
* Mental health monitoring
* Chatbots & recommendation systems

---

## ⚠️ Model File Note

The trained model file (`bert_emotion_model.pth`) is not included in this repository due to size limitations.

### To run the project:

1. Download the model from: https://drive.google.com/drive/folders/1gkoNqbWTdomB3sng6fRlpDxTDyRGLX0Q?usp=sharing
2. Place it in the root directory

---

## ▶️ How to Run

```bash
# Clone repo
git clone <your-repo-link>

# Go to project folder
cd tweet_emotion_detection

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

---

## 📈 Future Improvements

* Deploy model using cloud (Render / Streamlit Cloud)
* Improve accuracy using fine-tuned transformer variants
* Add real-time Twitter API integration
* Build recommendation system based on emotion

---

## 👩‍💻 Author

**Kakul Barsiya**

---

## ⭐ Acknowledgement

This project demonstrates practical implementation of **context-aware NLP using BERT**, combining research ideas with real-world application.

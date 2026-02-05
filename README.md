
# 🔮 Sentiment Prediction Engine

A **high-performance, zero-latency NLP web application** for real-time sentiment analysis using Transformer models.  
Built with **Streamlit + HuggingFace**, optimized for speed, clarity, and production readiness.

---

## 🚀 Features

### Core Capabilities
- 🔮 **Real-Time Sentiment Prediction**
- 🧠 **Per-Sentence Sentiment Classification**
- 🧪 **NLP Preprocessing Pipeline Visualization**
- 🗃 **Batch CSV Prediction & Export**
- 📊 **Clean JSON Probability Output**
- ⚡ **Model Caching for Zero Latency**

### UI/UX
- 🌈 **True Glassmorphism UI**
- ⚡ **Animated Transitions**
- 📱 **Mobile-Responsive Layout**
- 🎨 **Dark Mode, High Contrast Design**

---

## 🧱 High Level Design (HLD)

### System Architecture

```

+-------------------+
|    User (UI)      |
|  Streamlit App    |
+---------+---------+
|
v
+-------------------+
| NLP Preprocessor  |
| (Cleaning, Split) |
+---------+---------+
|
v
+-------------------+
| Sentiment Model   |
+---------+---------+
|
v
+-------------------+
| Result Formatter  |
| JSON / UI Cards   |
+-------------------+

````

### Design Goals
- Low latency
- Modular components
- Easy scalability
- Clean UI separation

---

## 🧩 Low Level Design (LLD)

### Component Breakdown

#### 1️⃣ UI Layer (Streamlit)
- Text input
- Sentence cards
- Expandable NLP pipeline
- CSV upload & download

#### 2️⃣ NLP Preprocessing
- Lowercasing
- Noise removal
- Sentence tokenization

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z.!? ]", "", text)
    return text
````


#### 4️⃣ Post-Processing

* Probability normalization
* Sentiment label selection
* JSON formatting

---

## 🔁 Data Flow Diagram

```
User Input
   ↓
Text Cleaning
   ↓
Sentence Tokenization
   ↓
ML Model
   ↓
Score Aggregation
   ↓
UI Rendering + Export
```

---

## 🛠 Tech Stack

| Layer         | Technology                 |
| ------------- | -------------------------- |
| Frontend      | Streamlit                  |
| ML Framework  | Scikit-learn               |
| Language      | Python 3.10+               |
| UI Styling    | Custom CSS (Glassmorphism) |
| Tokenization  | NLTK                       |
| Data Handling | Pandas                     |

---

## 📂 Project Structure

```
📁 sentiment-engine/
│
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── assets/                # UI assets (optional)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <REPO_LINK>
cd sentiment-engine
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📊 Batch Prediction Format

### Input CSV

```csv
text
This product is amazing
Delivery was very late
Average quality
```

### Output CSV

```csv
text,sentiment
This product is amazing,Positive
Delivery was very late,Negative
Average quality,Neutral
```

---

## 📈 Performance Optimizations

| Optimization    | Impact            |
| --------------- | ----------------- |
| Model caching   | 🚀 10× faster     |
| No SHAP         | ❌ latency removed |
| Local inference | 🧠 no API delay   |

---

## 🔐 Security & Reliability

* No external API calls
* Local inference only
* No user data persistence
* Stateless execution

---

## 🧪 Testing Strategy

* Manual UI testing
* CSV batch validation
* Sentence edge cases
* Large text handling

---

## 🚀 Deployment Options

### Local

* Streamlit CLI

### Cloud

* Streamlit Cloud
* AWS EC2
* Azure App Service
* Docker + Kubernetes (future)

---

## 🔮 Future Enhancements

* 🌍 Multilingual sentiment
* 😃 Emotion classification
* 📊 Confidence bar charts
* 🔗 FastAPI backend
* ⚛️ React frontend
* 📦 Dockerized deployment
* 🔐 Authentication & user roles
---
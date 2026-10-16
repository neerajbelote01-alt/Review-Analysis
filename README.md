# ReviewIQ AI — AI-Based Product Review Analysis Using NLP

> **Subtitle:** Aspect-Based Sentiment Analysis (ABSA) of Customer Reviews  
> **Tagline:** *Turn Customer Opinions into Actionable Intelligence.*

---

## 🌟 Overview

**ReviewIQ AI** performs granular **Aspect-Based Sentiment Analysis (ABSA)** on customer product reviews. Unlike standard sentiment classifiers that assign a single score, ReviewIQ decomposes customer opinions into individual features:

- **Review:** *"The camera is excellent and the display looks beautiful, but the battery drains quickly and the price is too high."*
- **Camera:** Positive
- **Display:** Positive
- **Battery:** Negative
- **Price:** Negative
- **Overall Sentiment:** Mixed

This standalone 7-file package includes the complete Natural Language Processing engine, 205 real smartphones and gadgets (Apple, Samsung, Google, OnePlus, Xiaomi, Sony, etc.), high-resolution product imagery, direct Amazon & Flipkart store purchase links, and an interactive Liquid Glass dashboard.

---

## 🚀 Quickstart (Run Locally in 2 Commands)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full-stack server:**
   ```bash
   python app.py
   ```

3. **Open in your browser:**
   ```text
   http://localhost:8000
   ```

The application will serve the interactive Liquid Glass UI at `/` and the interactive API docs at `/docs`.

---

## ☁️ 1-Click Deployment

### Deploy to Render ([render.com](https://render.com))
1. Create a **New Web Service** and select this repository.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `python app.py`

### Deploy with Docker
```bash
docker build -t reviewiq-ai .
docker run -p 8000:8000 reviewiq-ai
```

---

## 📊 Model Evaluation Results

- **Test Accuracy:** **90.0%**
- **Macro F1-Score:** **0.900**
- **Supported Aspects (12):** Camera, Battery, Display, Performance, Price, Design, Build quality, Audio and speakers, Software and UI, Connectivity, Charging, Durability.
- **Handled Phenomena:** Multi-aspect contrasts (`but`, `however`), negation windowing (`not bad`, `barely lasts`), evidence clause highlighting.

---

## 📦 What's Inside (7 Clean Standalone Files)

1. `app.py`: Full-stack FastAPI application with the complete ABSA NLP engine and REST API.
2. `index.html`: Liquid Glass React + Tailwind CSS dashboard with Dark/Light mode and interactive tabs.
3. `database.json`: Complete database of 205 authentic products, specs, photos, and store links.
4. `requirements.txt`: Python package requirements.
5. `Dockerfile`: Cloud container deployment configuration.
6. `README.md`: Project documentation and quickstart instructions.
7. `.gitignore`: Git exclusion rules.

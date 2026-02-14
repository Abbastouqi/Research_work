---
title: Urdu Emoji Predictor
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---

# 🎯 Urdu Emoji Predictor

An AI-powered tool that predicts relevant emojis for Urdu text using machine learning and semantic similarity.

## 🚀 Try It Out!

Simply enter Urdu text and get the most relevant emojis instantly.

## 🎯 Examples

- `میں بہت خوش ہوں` → 🎉 🎊 👌
- `دل ٹوٹ گیا ہے` → 🌚 😞 💔  
- `نیند آ رہی ہے` → 😴 😞 🌚
- `دوستوں کے ساتھ پارٹی` → 🎉 😋 🎊

## 🔧 How It Works

1. **Text Encoding**: Converts Urdu text to semantic embeddings using multilingual sentence transformers
2. **Similarity Search**: Compares text embeddings with pre-computed emoji embeddings
3. **Ranking**: Returns top emojis based on cosine similarity scores

## 🏗️ Technical Details

- **Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Emojis**: 80 most common emojis from Urdu social media
- **Method**: Cosine similarity between text and emoji embeddings
- **Framework**: Gradio + FastAPI

## 📊 Model Performance

- **Top-1 Accuracy**: ~16%
- **Top-3 Accuracy**: ~30%
- **Trained on**: 800K+ Urdu text-emoji pairs

## 🎮 Usage

```python
from urdu_specific_embedding import UrduOptimizedPredictor

predictor = UrduOptimizedPredictor("models/urdu_optimized_model")
predictions = predictor.predict_smart("میں بہت خوش ہوں", top_k=3)
# Returns: [('🎉', 0.555), ('🎊', 0.537), ('👌', 0.439)]
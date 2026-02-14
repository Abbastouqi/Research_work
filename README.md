---
title: Urdu Emoji Predictor
emoji: "🎯"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---

# Urdu Emoji Predictor

A compact, fast model that predicts the most relevant emojis for Urdu text using semantic similarity.

![App preview](frame_0001_redesigned.png.jpeg)

## Live Demo

Try the app on Hugging Face Spaces:

- https://huggingface.co/spaces/abbasNoway/Urdu_Emoji_predictor

## Highlights

- Urdu-first emoji prediction
- Lightweight inference with precomputed emoji embeddings
- Clean Gradio UI

## How It Works

1. Encode Urdu text with a multilingual sentence transformer.
2. Compare against precomputed emoji vectors using cosine similarity.
3. Return top-k emojis ranked by relevance.

## Examples

- "میں بہت خوش ہوں" -> 🎉 🎊 👌
- "دل ٹوٹ گیا ہے" -> 😞 💔 🌙
- "نیند آ رہی ہے" -> 😴 😞 🌙
- "دوستوں کے ساتھ پارٹی" -> 🎉 😊 🎊

## Usage

```python
from urdu_specific_embedding import UrduOptimizedPredictor

predictor = UrduOptimizedPredictor("models/urdu_optimized_model")
predictions = predictor.predict_smart("میں بہت خوش ہوں", top_k=3)
# Returns: [('🎉', 0.555), ('🎊', 0.537), ('👌', 0.439)]
```

## Technical Details

- Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Emojis: 80 frequent emojis from Urdu social media
- Method: cosine similarity on text/emoji embeddings
- Framework: Gradio + FastAPI

## Performance

- Top-1 Accuracy: ~16%
- Top-3 Accuracy: ~30%
- Trained on: 800K+ Urdu text-emoji pairs

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```
---
title: Urdu Emoji Predictor
emoji: ðŸŽ¯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---

# ðŸŽ¯ Urdu Emoji Predictor

An AI-powered tool that predicts relevant emojis for Urdu text using machine learning and semantic similarity.

## ðŸš€ Try It Out!

Simply enter Urdu text and get the most relevant emojis instantly.

## Research Paper

Read the full paper here:

- https://ieeexplore.ieee.org/document/11333609

## ðŸŽ¯ Examples

- `Ù…ÛŒÚº Ø¨ÛØª Ø®ÙˆØ´ ÛÙˆÚº` â†’ ðŸŽ‰ ðŸŽŠ ðŸ‘Œ
- `Ø¯Ù„ Ù¹ÙˆÙ¹ Ú¯ÛŒØ§ ÛÛ’` â†’ ðŸŒš ðŸ˜ž ðŸ’”  
- `Ù†ÛŒÙ†Ø¯ Ø¢ Ø±ÛÛŒ ÛÛ’` â†’ ðŸ˜´ ðŸ˜ž ðŸŒš
- `Ø¯ÙˆØ³ØªÙˆÚº Ú©Û’ Ø³Ø§ØªÚ¾ Ù¾Ø§Ø±Ù¹ÛŒ` â†’ ðŸŽ‰ ðŸ˜‹ ðŸŽŠ

## ðŸ”§ How It Works

1. **Text Encoding**: Converts Urdu text to semantic embeddings using multilingual sentence transformers
2. **Similarity Search**: Compares text embeddings with pre-computed emoji embeddings
3. **Ranking**: Returns top emojis based on cosine similarity scores

## ðŸ—ï¸ Technical Details

- **Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Emojis**: 80 most common emojis from Urdu social media
- **Method**: Cosine similarity between text and emoji embeddings
- **Framework**: Gradio + FastAPI

## ðŸ“Š Model Performance

- **Top-1 Accuracy**: ~16%
- **Top-3 Accuracy**: ~30%
- **Trained on**: 800K+ Urdu text-emoji pairs

## ðŸŽ® Usage

```python
from urdu_specific_embedding import UrduOptimizedPredictor

predictor = UrduOptimizedPredictor("models/urdu_optimized_model")
predictions = predictor.predict_smart("Ù…ÛŒÚº Ø¨ÛØª Ø®ÙˆØ´ ÛÙˆÚº", top_k=3)
# Returns: [('ðŸŽ‰', 0.555), ('ðŸŽŠ', 0.537), ('ðŸ‘Œ', 0.439)]
# urdu_specific_embedding.py (Updated)
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class UrduOptimizedPredictor:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.text_model.to(self.device)
        
        # Try multiple possible model file locations
        possible_paths = [
            "urdu_optimized_model.pkl",  # Direct in root
            "./urdu_optimized_model.pkl",  # Current directory
            "models/urdu_optimized_model/urdu_optimized_model.pkl",  # Local structure
            "/data/urdu_optimized_model.pkl"  # HF Spaces data directory
        ]
        
        model_loaded = False
        for model_file in possible_paths:
            if os.path.exists(model_file):
                print(f"📁 Loading model from: {model_file}")
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    self.emoji_embeddings = {k: v[0] for k, v in model_data['emoji_embeddings'].items()}
                    self.emoji_list = model_data['emoji_list']
                    
                    print(f"✅ Loaded Urdu-optimized model with {len(self.emoji_list)} meaningful emojis")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"❌ Error loading {model_file}: {e}")
                    continue
        
        if not model_loaded:
            print("❌ Could not load model file. Please make sure urdu_optimized_model.pkl is uploaded.")
            # Create empty structures to avoid crashes
            self.emoji_embeddings = {}
            self.emoji_list = []
    
    def predict_smart(self, text, top_k=3, min_confidence=0.3):
        """Smart prediction with confidence filtering"""
        # Check if model is loaded
        if not self.emoji_embeddings:
            return [("❌", 0.0)]  # Return error emoji if model not loaded
        
        # Get text embedding
        text_embedding = self.text_model.encode([text], convert_to_tensor=True)
        text_embedding_np = text_embedding.cpu().numpy()
        
        # Calculate similarities
        similarities = {}
        for emoji, emoji_embedding in self.emoji_embeddings.items():
            similarity = cosine_similarity(text_embedding_np, emoji_embedding.reshape(1, -1))[0][0]
            similarities[emoji] = similarity
        
        # Filter by confidence and return top K
        filtered = [(emoji, score) for emoji, score in similarities.items() if score >= min_confidence]
        sorted_emojis = sorted(filtered, key=lambda x: x[1], reverse=True)
        
        # If no confident predictions, return top 1 anyway
        if not sorted_emojis:
            top_overall = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:1]
            return top_overall
        
        return sorted_emojis[:top_k]
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np

class UrduOptimizedPredictor:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the multilingual model
        self.text_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.text_model.to(self.device)
        
        # Load YOUR model
        model_file = "urdu_optimized_model.pkl"
        print(f"📁 Loading YOUR model from: {model_file}")
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.emoji_embeddings = {k: v[0] for k, v in model_data['emoji_embeddings'].items()}
            self.emoji_list = model_data['emoji_list']
            
            print(f"✅ SUCCESS: Loaded YOUR Urdu-optimized model with {len(self.emoji_list)} emojis")
            print(f"📊 Your emojis: {self.emoji_list[:20]}...")  # Show first 20 emojis
            
        except Exception as e:
            print(f"❌ ERROR loading your model: {e}")
            raise e
    
    def predict_smart(self, text, top_k=3, min_confidence=0.3):
        """Use YOUR model for prediction"""
        print(f"\n🔍 PREDICTING for: '{text}'")
        
        # Get text embedding
        text_embedding = self.text_model.encode([text], convert_to_tensor=True)
        text_embedding_np = text_embedding.cpu().numpy()
        
        # Calculate similarities with YOUR emoji embeddings
        similarities = {}
        for emoji, emoji_embedding in self.emoji_embeddings.items():
            similarity = cosine_similarity(text_embedding_np, emoji_embedding.reshape(1, -1))[0][0]
            similarities[emoji] = similarity
        
        print(f"📈 Similarities calculated for {len(similarities)} emojis")
        
        # Filter by confidence and return top K
        filtered = [(emoji, score) for emoji, score in similarities.items() if score >= min_confidence]
        sorted_emojis = sorted(filtered, key=lambda x: x[1], reverse=True)
        
        print(f"🎯 Top predictions: {sorted_emojis[:top_k]}")
        
        # If no confident predictions, return top overall
        if not sorted_emojis:
            top_overall = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
            print(f"⚠️ No confident predictions, using top overall: {top_overall}")
            return top_overall
        
        return sorted_emojis[:top_k]

# Initialize predictor
print("🚀 Loading YOUR Urdu Emoji Prediction Model...")
predictor = UrduOptimizedPredictor()

def predict_emoji(urdu_text):
    """Main prediction function using YOUR model"""
    if not urdu_text.strip():
        return "⬅️ اردو متن لکھیں"
    
    try:
        # Get predictions from YOUR model
        predictions = predictor.predict_smart(urdu_text, top_k=3, min_confidence=0.3)
        
        # Format output - ONLY EMOJIS, no scores or text
        if predictions:
            # Extract just the emojis from predictions
            emojis_only = [emoji for emoji, score in predictions]
            # Join them with spaces for clean display
            result = " ".join(emojis_only)
            return result
        else:
            return "❌"
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "⚠️"

# Test your model with some examples before starting the interface
print("\n" + "="*60)
print("🧪 TESTING YOUR MODEL WITH SAMPLE TEXTS")
print("="*60)

test_texts = [
    "میں بہت خوش ہوں",
    "دل ٹوٹ گیا ہے", 
    "دوستوں کے ساتھ پارٹی کا مزہ آیا",
    "امی نے میری پسندیدہ ڈش بنائی ہے",
    "غصہ سے دماغ پھٹ رہا ہے"
]

for text in test_texts:
    print(f"\n📝 Testing: '{text}'")
    predictions = predictor.predict_smart(text, top_k=3, min_confidence=0.3)
    print(f"   → {[emoji for emoji, score in predictions]}")

print("\n" + "="*60)
print("🚀 STARTING GRADIO INTERFACE")
print("="*60)

# Create Gradio interface
demo = gr.Blocks(title="اردو ایموجی پیشنگوئی")

with demo:
    gr.Markdown(
        """
        # 🎯 اردو ایموجی پیشنگوئی
        
        اپنے اردو متن کے لیے موزوں ترین ایموجیز دریافت کریں
        
        **10 لاکھ+ اردو ٹویٹس** پر **80+ اردو ایموجیز** سے تربیت یافتہ ماڈل
        - **تین بہترین ایموجیز** کی پیشنگوئی
        - فوری اور درست نتائج
        - مکمل طور پر اردو کے لیے ڈیزائن کیا گیا
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="اردو متن درج کریں",
                placeholder="اپنا اردو متن یہاں لکھیں... مثلاً: آج میں بہت خوش ہوں",
                lines=3
            )
            
            predict_btn = gr.Button("🎯 ایموجیز حاصل کریں", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(
                label="پیشنگوئی شدہ ایموجیز",
                placeholder="یہاں ایموجیز ظاہر ہوں گی...",
                lines=2
            )
    
    gr.Markdown("### 💡 مثالیں")
    examples = gr.Examples(
        examples=[
            ["میں آج بہت خوش ہوں"],
            ["دل ٹوٹ گیا ہے"],
            ["دوستوں کے ساتھ پارٹی کا مزہ آیا"],
            ["نیند آ رہی ہے"],
            ["امی نے میری پسندیدہ کھانا بنایا ہے"],
            ["محبت میں پڑ گیا ہوں"],
            ["غصہ سے دماغ پھٹ رہا ہے"],
            ["آج کا دن بہت خاص ہے، سب خوش رہیں!"]
        ],
        inputs=input_text,
        outputs=output_text,
        fn=predict_emoji,
        cache_examples=False
    )

    # Connect button to function
    predict_btn.click(fn=predict_emoji, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
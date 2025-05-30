import os
import joblib
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'Models'))

def load_sentiment_model_and_vectorizer():
    model_path = os.path.join(MODEL_DIR, 'final_lstm_model.h5')
    vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    model = load_model(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def load_recommendation_model():
    class DummyRecommender:
        def recommend(self, user_id):
            return ["book1", "book2", "book3"]
    return DummyRecommender()
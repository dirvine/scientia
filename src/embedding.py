import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

def preprocess_text(text):
    """Preprocess input text"""
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

def get_embedding(embedding_model, text):
    """Get embedding for input text"""
    return embedding_model.encode([text])[0]

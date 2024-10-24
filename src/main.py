import os
import nltk
import faiss
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from rich import print
import numpy as np
import gradio as gr

from src.config import initialize_config
from src.device_setup import setup_device
from src.embedding import preprocess_text, get_embedding
from src.database_manager import add_text_to_database, load_and_display_database
from src.interface import create_interface

# Global variables
tokenizers = {}
databases = {}
embedding_model = None
device = None

def setup():
    """Main setup function"""
    global tokenizers, databases, embedding_model, device

    try:
        print("Starting setup...")
        config = initialize_config()

        # Setup device
        device, dev_info = setup_device()
        print(dev_info)

        # Download NLTK data
        print("Downloading NLTK data...")
        for nltk_package in config['nltk_downloads']:
            nltk.download(nltk_package)

        # Setup tokenizers
        print("Setting up tokenizers...")
        tokenizers = {
            name: AutoTokenizer.from_pretrained(
                model_config["name"],
                use_fast=model_config["use_fast"]
            )
            for name, model_config in config['tokenizer_models'].items()
        }

        # Load the Embedding Model
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(config['embedding_model_name'])

        # Initialize the Vector Databases
        vector_db_dir = config['vector_db_dir']
        os.makedirs(vector_db_dir, exist_ok=True)
        databases = {
            name: faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            for name in tokenizers
        }
        print("Finished setup")

    except Exception as e:
        print(f"An error occurred during setup: {e}")

def tokenize_and_store_input(user_input):
    """Process and store user input"""
    processed_text = preprocess_text(user_input)
    results = []
    for tokenizer_name, tokenizer in tokenizers.items():
        tokens = add_text_to_database(processed_text, tokenizer, embedding_model, databases[tokenizer_name], device, initialize_config()['vector_db_dir'], tokenizer_name)
        results.append(f"Tokenized output [{tokenizer_name}]: {tokens}")
    return "\n".join(results)

def main():
    """Main application function"""
    setup()

    def encode_text(text):
        return tokenize_and_store_input(text)

    def show_encodings(tokenizer_name):
        return load_and_display_database(initialize_config()['vector_db_dir'], tokenizer_name)

    create_interface(tokenizers, encode_text, show_encodings)

if __name__ == "__main__":
    main()

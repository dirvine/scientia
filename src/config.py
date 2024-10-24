def initialize_config():
    """Initialize configuration values and return a dictionary of settings"""
    return {
        'tokenizer_models': {
            "distillgpt2": {"name": "abhinema/distillgpt2", "use_fast": False},
            "t5-small": {"name": "google-t5/t5-small", "use_fast": True}
        },
        'embedding_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'nltk_downloads': ['punkt', 'punkt_tab'],
        'vector_db_dir': 'brain'
    }

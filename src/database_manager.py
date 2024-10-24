import os
import faiss
import numpy as np

def add_text_to_database(text, tokenizer, embedding_model, database, device, vector_db_dir, tokenizer_name):
    """Add text to database and return tokens"""
    tokens = tokenizer(text, return_tensors='pt').to(device)
    embedding = embedding_model.encode([text])[0]
    vector = np.array([embedding]).astype('float32')
    database.add(vector)
    # Save the database to the brain directory
    faiss.write_index(database, f"{vector_db_dir}/{tokenizer_name}.index")
    return tokens['input_ids']  # Return the input IDs tensor

def load_and_display_database(vector_db_dir, tokenizer_name):
    """Load and display the vector database for a specific tokenizer"""
    database_path = f"{vector_db_dir}/{tokenizer_name}.index"
    results = []
    if os.path.exists(database_path):
        results.append(f"Loading database for tokenizer [{tokenizer_name}]...")
        database = faiss.read_index(database_path)
        results.append(f"Database [{tokenizer_name}] contains {database.ntotal} vectors.")
        # Display vectors (for demonstration, limit to first 5 vectors)
        for i in range(min(5, database.ntotal)):
            vector = database.reconstruct(i)
            results.append(f"Vector {i} for [{tokenizer_name}]: {vector}")
    else:
        results.append(f"No database found for tokenizer [{tokenizer_name}].")
    return "\n".join(results)

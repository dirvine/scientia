import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login

def setup_model():
    """Setup model with authentication handling"""
    try:
        # Check for HF_TOKEN in environment
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        login(hf_token)
        
        # Initialize both models
        models = {}
        tokenizers = {}
        
        # Setup Llama 3.2 1B
        print("Loading Llama-3.2-1B...")
        model_name = "NousResearch/Llama-3.2-1B"
        
        tokenizers['llama'] = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            token=hf_token
        )
        
        models['llama'] = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        
        # Add smaller backup model in case of issues
        print("Loading backup model...")
        backup_model = "facebook/opt-125m"
        tokenizers['backup'] = AutoTokenizer.from_pretrained(backup_model, padding_side="left")
        models['backup'] = AutoModelForCausalLM.from_pretrained(
            backup_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return models, tokenizers
        
    except Exception as e:
        print(f"Error setting up model: {e}")
        print("\nPlease ensure you're authenticated with Hugging Face:")
        print("1. Get your token from https://huggingface.co/settings/tokens")
        print("2. Set it as an environment variable: export HF_TOKEN='your_token'")
        exit(1)

def generate_response(models, tokenizers, prompt, model_key='llama'):
    """Generate response using specified model"""
    try:
        model = models[model_key]
        tokenizer = tokenizers[model_key]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print(f"Error with {model_key} model, falling back to backup...")
        try:
            return generate_response(models, tokenizers, prompt, 'backup')
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    models, tokenizers = setup_model()
    print("Welcome! Ask me anything. Type 'exit' to quit.")
    print("Using Llama-3.2-1B model for chat...")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            break
            
        response = generate_response(models, tokenizers, user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()

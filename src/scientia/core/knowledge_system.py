import logging
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json
import hashlib
from .vector_store import VectorStore
from .models import KnowledgePacket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityException(Exception):
    """Custom exception for security-related issues"""
    pass

class ScientiaCore:
    """Core system for managing knowledge in Scientia"""
    
    def __init__(
        self,
        chat_model_name: str = "NousResearch/Hermes-2-Pro-Llama-3-8B",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = None
    ):
        # Better device detection that handles Mac devices
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"  # Use Metal Performance Shaders on Mac
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        try:
            # Initialize embedding model first to get dimensions
            self.embedding_model = SentenceTransformer(embedding_model_name)
            
            # Get embedding dimension by encoding a test string
            test_embedding = self.embedding_model.encode("test")
            embedding_dim = len(test_embedding)
            
            # Initialize vector store with the correct dimension
            self.vector_store = VectorStore(dimension=embedding_dim)
            
            # Initialize language model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                chat_model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto"
            )
            
            # Move models to appropriate device
            self.embedding_model = self.embedding_model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    async def query_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Query the knowledge base for relevant information
        
        Args:
            query (str): The query text
            k (int): Number of results to return
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of results with their similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Search the vector store
            results = await self.vector_store.search(query_embedding, k=k)
            
            # Format results
            formatted_results = []
            for result, score in results:
                formatted_results.append({
                    "content": result.content if hasattr(result, 'content') else str(result),
                    "similarity": float(score),
                    "metadata": result.metadata if hasattr(result, 'metadata') else {}
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            raise

    async def process_knowledge(self, input_text: str) -> Dict[str, Union[str, float]]:
        """
        Process input text as knowledge and generate a response
        
        Args:
            input_text (str): The input text to process
            
        Returns:
            Dict[str, Union[str, float]]: Dictionary containing response and confidence
        """
        try:
            # First query the knowledge base for relevant context
            relevant_knowledge = await self.query_knowledge(input_text)
            
            # Format the prompt with context
            context = "\n".join([f"Context: {item['content']}" for item in relevant_knowledge[:2]])
            prompt = f"{context}\nQuestion: {input_text}\nAnswer:"
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generation_config = {
                    "max_length": 200,
                    "num_return_sequences": 1,
                    "temperature": 0.7,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                outputs = self.model.generate(**inputs, **generation_config)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response if it's included
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Calculate confidence based on similarity scores of retrieved knowledge
            confidence = max([item['similarity'] for item in relevant_knowledge]) if relevant_knowledge else 0.85
            
            result = {
                "response": response,
                "confidence": confidence,
                "relevant_knowledge": relevant_knowledge
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing knowledge: {str(e)}")
            raise

    async def add_to_knowledge_base(self, knowledge: KnowledgePacket):
        """Add knowledge to the vector store"""
        try:
            # Generate embedding for the knowledge
            embedding = self.embedding_model.encode(knowledge.content)
            
            # Store in vector store
            await self.vector_store.add_vector(embedding, knowledge)
            
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {str(e)}")
            raise

    async def chat(self, message: str, context: str = None) -> Dict[str, Union[str, float]]:
        """
        Process a chat message and generate a response
        
        Args:
            message (str): The user's message
            context (str, optional): Additional context for the conversation
        
        Returns:
            Dict[str, Union[str, float]]: Dictionary containing response and metadata
        """
        try:
            logger.info("Querying knowledge base...")
            # Get relevant knowledge
            relevant_knowledge = await self.query_knowledge(message)
            
            logger.info("Generating response...")
            # Format the prompt with context and chat-specific formatting
            knowledge_context = "\n".join([f"Relevant information: {item['content']}" for item in relevant_knowledge[:2]])
            
            # Combine provided context with knowledge context
            full_context = f"{context}\n{knowledge_context}" if context else knowledge_context
            
            prompt = f"""Based on the following information and your knowledge, please provide a helpful response:

{full_context}

User: {message}
Assistant:"""

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generation_config = {
                    "max_length": 500,  # Longer for chat responses
                    "num_return_sequences": 1,
                    "temperature": 0.7,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2
                }
                
                outputs = self.model.generate(**inputs, **generation_config)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response by removing the prompt
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Calculate confidence based on knowledge relevance
            confidence = max([item['similarity'] for item in relevant_knowledge]) if relevant_knowledge else 0.85
            
            return {
                "response": response,
                "confidence": confidence,
                "relevant_knowledge": relevant_knowledge,
                "model": self.model.config.name_or_path
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise

import logging
from typing import Dict, List, Optional, Union, Set, BinaryIO
import torch
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    AutoProcessor,  # Change this from FuyuProcessor
)  # Remove FuyuForCausalLM
from sentence_transformers import SentenceTransformer
import json
import hashlib
from .vector_store import VectorStore
from .models import KnowledgePacket
import asyncio
from tqdm import tqdm
import PyPDF2
import docx
import PIL.Image
import io
import fitz  # PyMuPDF
import pytesseract
from pathlib import Path
from datetime import datetime

# Configure logging with a more specific format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*path of torch.classes.*")
warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size` from an ambiguous collection.*")
warnings.filterwarnings("ignore", category=UserWarning)

class SecurityException(Exception):
    """Custom exception for security-related issues"""
    pass

class DocumentProcessor:
    """Handles different document types for knowledge extraction"""
    
    @staticmethod
    def extract_text_from_image(image: PIL.Image.Image) -> str:
        """Extract text from an image using OCR"""
        try:
            # Preprocess image for better OCR results
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return ""

    @staticmethod
    def process_pdf(file: BinaryIO) -> List[Dict[str, Union[str, PIL.Image.Image]]]:
        """Process PDF files, extracting text and images with OCR"""
        results = []
        try:
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    results.append({"type": "text", "content": text})
                
                # Extract and process images
                images = page.get_images(full=True)
                for img_index, img_info in enumerate(images):
                    img_index = img_info[0]
                    base_image = pdf_document.extract_image(img_index)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = PIL.Image.open(io.BytesIO(image_bytes))
                    results.append({"type": "image", "content": image})
                    
                    # Perform OCR on the image
                    ocr_text = DocumentProcessor.extract_text_from_image(image)
                    if ocr_text:
                        results.append({
                            "type": "text",
                            "content": ocr_text,
                            "metadata": {"source": "image_ocr", "page": page_num + 1}
                        })
                
                # Extract text from any figures or diagrams on the page
                pix = page.get_pixmap()
                img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = DocumentProcessor.extract_text_from_image(img)
                if ocr_text:
                    results.append({
                        "type": "text",
                        "content": ocr_text,
                        "metadata": {"source": "page_ocr", "page": page_num + 1}
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    @staticmethod
    def process_docx(file: BinaryIO) -> List[Dict[str, Union[str, PIL.Image.Image]]]:
        """Process Word documents, extracting text and images with OCR"""
        results = []
        try:
            doc = docx.Document(file)
            
            # Extract text
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    results.append({"type": "text", "content": paragraph.text})
            
            # Extract and process images
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_data = rel.target_part.blob
                    image = PIL.Image.open(io.BytesIO(image_data))
                    results.append({"type": "image", "content": image})
                    
                    # Perform OCR on the image
                    ocr_text = DocumentProcessor.extract_text_from_image(image)
                    if ocr_text:
                        results.append({
                            "type": "text",
                            "content": ocr_text,
                            "metadata": {"source": "image_ocr"}
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise

    @staticmethod
    def process_image(file: BinaryIO) -> List[Dict[str, Union[str, PIL.Image.Image]]]:
        """Process image files, extracting visual content and OCR text"""
        results = []
        try:
            # Load image
            image = PIL.Image.open(file)
            results.append({"type": "image", "content": image})
            
            # Perform OCR with enhanced preprocessing
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            ocr_text = DocumentProcessor.extract_text_from_image(image)
            if ocr_text:
                results.append({
                    "type": "text",
                    "content": ocr_text,
                    "metadata": {"source": "primary_ocr"}
                })
            
            return results
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

class ScientiaCore:
    """Core system for managing knowledge in Scientia"""
    
    def __init__(
        self,
        chat_model_name: str = "NousResearch/Hermes-2-Pro-Llama-3-8B",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        fuyu_model_name: str = "adept/fuyu-8b",  # Updated model name
        device: str = None,
        enable_multimodal: bool = False  # Add flag to control multimodal features
    ):
        # Suppress warnings during initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Device detection
            if device is None:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            else:
                self.device = device
            
            logger.info(f"Using device: {self.device}")
            
            try:
                # Initialize models
                self.embedding_model = SentenceTransformer(embedding_model_name)
                test_embedding = self.embedding_model.encode("test")
                embedding_dim = len(test_embedding)
                self.vector_store = VectorStore(dimension=embedding_dim)
                
                self.tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    chat_model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto"
                )
                
                # Initialize Fuyu only if multimodal is enabled
                self.multimodal_enabled = False
                if enable_multimodal:
                    try:
                        logger.info("Initializing multimodal capabilities...")
                        self.fuyu_processor = AutoProcessor.from_pretrained(fuyu_model_name)
                        self.fuyu_model = AutoModelForCausalLM.from_pretrained(
                            fuyu_model_name,
                            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                            device_map="auto"
                        )
                        self.multimodal_enabled = True
                        logger.info("Multimodal capabilities initialized successfully")
                    except Exception as e:
                        logger.warning(f"Failed to initialize multimodal capabilities: {str(e)}")
                        logger.warning("Continuing with text-only processing")
                
                # Initialize document processor
                self.doc_processor = DocumentProcessor()
                
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
                    "max_new_tokens": 200,  # Changed from max_length to max_new_tokens
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
        """
        try:
            logger.info("Querying knowledge base...")
            # Get relevant knowledge with increased number of results
            relevant_knowledge = await self.query_knowledge(message, k=10)  # Increased from 5 to 10
            
            logger.info("Generating response...")
            # Format the prompt with context and chat-specific formatting
            knowledge_context = "\n".join([
                f"Important context: {item['content']}" 
                for item in relevant_knowledge 
                if item['similarity'] > 0.5  # Only include highly relevant context
            ])
            
            # Enhanced prompt with stronger emphasis on using provided knowledge
            prompt = f"""You are a helpful AI assistant with access to a knowledge base. 
            Use the following verified information from the knowledge base to inform your response:

{knowledge_context}

If the knowledge base contains relevant information, prioritize using it in your response.
If no relevant information is found in the knowledge base, indicate that and provide a general response.

User: {message}
Assistant: Let me help you based on the information available."""

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with adjusted parameters
            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": 500,
                    "num_return_sequences": 1,
                    "temperature": 0.7,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3
                }
                
                outputs = self.model.generate(**inputs, **generation_config)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response by removing the prompt
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Calculate confidence based on knowledge relevance
            relevant_scores = [item['similarity'] for item in relevant_knowledge]
            confidence = max(relevant_scores) if relevant_scores else 0.85
            
            # Add debug information
            logger.info(f"Found {len(relevant_knowledge)} relevant items in knowledge base")
            logger.info(f"Top similarity score: {confidence}")
            
            return {
                "response": response,
                "confidence": confidence,
                "relevant_knowledge": relevant_knowledge,
                "model": self.model.config.name_or_path,
                "knowledge_used": bool(relevant_knowledge)  # Indicate if knowledge was found
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise

    async def explore_topic(self, topic: str, max_queries: int = 10) -> Dict[str, Union[str, List[str]]]:
        """
        Deeply explore a topic by generating and executing multiple related queries
        
        Args:
            topic (str): The main topic to explore
            max_queries (int): Maximum number of sub-queries to generate
            
        Returns:
            Dict[str, Union[str, List[str]]]: Comprehensive knowledge about the topic
        """
        try:
            # Initial query to get topic overview
            logger.info(f"Starting deep exploration of topic: {topic}")
            
            # Generate exploration queries
            exploration_prompt = f"""Generate {max_queries} different specific questions to deeply understand the topic: {topic}. 
            Focus on different aspects like history, current state, future implications, technical details, and practical applications.
            Format each question on a new line."""
            
            query_generation = await self.chat(exploration_prompt)
            sub_queries = [q.strip() for q in query_generation['response'].split('\n') if q.strip()]
            
            # Execute all sub-queries
            results = []
            explored_aspects = set()
            
            for query in tqdm(sub_queries[:max_queries], desc="Exploring topic"):
                if query not in explored_aspects:
                    response = await self.chat(query)
                    results.append({
                        "question": query,
                        "answer": response['response'],
                        "confidence": response['confidence'],
                        "sources": [k['content'] for k in response['relevant_knowledge']]
                    })
                    explored_aspects.add(query)
                    
                    # Small delay to prevent rate limiting
                    await asyncio.sleep(0.5)
            
            # Synthesize findings
            synthesis_prompt = f"""Based on all these findings about {topic}:
            {' '.join([f"Q: {r['question']} A: {r['answer']}" for r in results])}
            
            Please provide:
            1. A comprehensive summary
            2. Key insights
            3. Areas that need further exploration"""
            
            final_synthesis = await self.chat(synthesis_prompt)
            
            return {
                "topic": topic,
                "summary": final_synthesis['response'],
                "detailed_findings": results,
                "confidence": sum(r['confidence'] for r in results) / len(results),
                "exploration_breadth": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error in topic exploration: {str(e)}")
            raise

    async def process_document(self, file: BinaryIO, filename: str) -> List[KnowledgePacket]:
        """Process uploaded document and extract knowledge packets"""
        try:
            file_extension = Path(filename).suffix.lower()
            
            # Process different file types
            if file_extension == '.pdf':
                contents = self.doc_processor.process_pdf(file)
            elif file_extension in ['.docx', '.doc']:
                contents = self.doc_processor.process_docx(file)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                contents = self.doc_processor.process_image(file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            knowledge_packets = []
            
            # Process each content item
            for item in contents:
                if item["type"] == "text":
                    # Process text directly
                    embedding = self.embedding_model.encode(item["content"])
                    
                    # Include OCR metadata if available
                    metadata = {
                        "file_source": filename,
                        "content_type": "text",
                        **item.get("metadata", {})  # Include any OCR-specific metadata
                    }
                    
                    packet = KnowledgePacket(
                        content=item["content"],
                        embeddings=embedding,
                        source_type="DOCUMENT",
                        timestamp=datetime.now().isoformat(),
                        confidence=0.9 if "ocr" not in str(metadata.get("source", "")) else 0.7,  # Lower confidence for OCR
                        context_hash=hashlib.sha256(item["content"].encode()).hexdigest(),
                        privacy_level="PUBLIC",
                        metadata=metadata
                    )
                    knowledge_packets.append(packet)
                
                elif item["type"] == "image" and self.multimodal_enabled:
                    # Process image with Fuyu if available
                    try:
                        inputs = self.fuyu_processor(
                            text="Describe this image in detail:",
                            images=item["content"],
                            return_tensors="pt"
                        ).to(self.device)
                        
                        outputs = self.fuyu_model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.7
                        )
                        
                        image_description = self.fuyu_processor.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        logger.warning(f"Failed to process image with Fuyu: {str(e)}")
                        image_description = "Image content (processing unavailable)"
                    
                    # Create embedding for the image description
                    embedding = self.embedding_model.encode(image_description)
                    
                    packet = KnowledgePacket(
                        content=image_description,
                        embeddings=embedding,
                        source_type="DOCUMENT",
                        timestamp=datetime.now().isoformat(),
                        confidence=0.85,
                        context_hash=hashlib.sha256(image_description.encode()).hexdigest(),
                        privacy_level="PUBLIC",
                        metadata={
                            "file_source": filename,
                            "content_type": "image_description"
                        }
                    )
                    knowledge_packets.append(packet)
            
            return knowledge_packets
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise


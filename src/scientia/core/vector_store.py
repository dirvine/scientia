from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import faiss
import torch
import logging
from .models import KnowledgePacket, SearchResult

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages storage and retrieval of vector embeddings"""
    
    def __init__(self, dimension: int):
        """Initialize the vector store"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.data_store = []  # Initialize data store list
        
        # Store metadata separately
        self.metadata = {}
        self.content = {}
        
        # Enable GPU if available
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
            logger.info("Using GPU for vector store")
    
    def add_packet(self, packet: 'KnowledgePacket') -> None:
        """Add a knowledge packet to the store"""
        try:
            # Prepare vector for insertion
            vector = np.array([packet.embeddings]).astype('float32')
            
            # Add to FAISS index
            self.index.add(vector)
            
            # Store metadata and content
            idx = self.index.ntotal - 1
            self.metadata[idx] = {
                'metadata': packet.metadata,
                'source_type': packet.source_type,
                'timestamp': packet.timestamp.isoformat(),
                'confidence': packet.confidence,
                'context_hash': packet.context_hash
            }
            self.content[idx] = packet.content
            
            logger.debug(f"Added packet {packet.context_hash} to vector store")
            
        except Exception as e:
            logger.error(f"Error adding packet to vector store: {str(e)}")
            raise
    
    def semantic_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.8
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        try:
            # Ensure query vector is in correct format
            query_vector = query_vector.reshape(1, -1).astype('float32')
            
            # Perform search
            distances, indices = self.index.search(query_vector, top_k)
            
            # Process results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                # Convert distance to similarity score (0-1)
                score = 1 / (1 + dist)
                
                # Skip results below threshold
                if score < threshold:
                    continue
                
                # Get metadata and content
                meta = self.metadata.get(int(idx), {})
                content = self.content.get(int(idx), "")
                
                results.append(SearchResult(
                    content=content,
                    metadata=meta.get('metadata', {}),
                    score=float(score),
                    source_type=meta.get('source_type', 'unknown'),
                    timestamp=meta.get('timestamp', '')
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            raise
    
    def save_index(self, path: str) -> None:
        """Save the FAISS index and metadata to disk"""
        try:
            # Convert GPU index to CPU if necessary
            if faiss.get_num_gpus() > 0:
                index_cpu = faiss.index_gpu_to_cpu(self.index)
            else:
                index_cpu = self.index
                
            # Save FAISS index
            faiss.write_index(index_cpu, f"{path}.index")
            
            # Save metadata and content
            np.savez(
                f"{path}.npz",
                metadata=np.array([self.metadata], dtype=object),
                content=np.array([self.content], dtype=object)
            )
            
            logger.info(f"Saved vector store to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_index(self, path: str) -> None:
        """Load the FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.index")
            
            # Move to GPU if available
            if faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(),
                    0,
                    self.index
                )
            
            # Load metadata and content
            data = np.load(f"{path}.npz", allow_pickle=True)
            self.metadata = data['metadata'][0].item()
            self.content = data['content'][0].item()
            
            logger.info(f"Loaded vector store from {path}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    async def add_vector(self, vector: np.ndarray, data: Any) -> None:
        """
        Add a vector and its associated data to the store
        
        Args:
            vector (np.ndarray): The vector to store
            data: The associated data
        """
        try:
            # Ensure vector is the right shape and type
            vector = np.array(vector).reshape(1, -1).astype('float32')
            if vector.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension {vector.shape[1]} does not match store dimension {self.dimension}")
            
            # Add to FAISS index
            self.index.add(vector)
            # Store the data
            self.data_store.append(data)
            
        except Exception as e:
            logger.error(f"Error adding vector to store: {str(e)}")
            raise

    async def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Search for the k nearest vectors
        
        Args:
            query_vector (np.ndarray): The query vector
            k (int): Number of results to return
            
        Returns:
            List[Tuple[Any, float]]: List of (data, similarity_score) pairs
        """
        try:
            # Ensure vector is the right shape and type
            query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
            if query_vector.shape[1] != self.dimension:
                raise ValueError(f"Query vector dimension {query_vector.shape[1]} does not match store dimension {self.dimension}")
            
            # If we have fewer items than k, adjust k
            k = min(k, len(self.data_store))
            if k == 0:
                return []
            
            # Search the index
            distances, indices = self.index.search(query_vector, k)
            
            # Convert distances to similarity scores (1 / (1 + distance))
            similarities = 1 / (1 + distances[0])
            
            # Return results with their similarity scores
            results = [(self.data_store[idx], float(sim)) 
                      for idx, sim in zip(indices[0], similarities) 
                      if idx != -1]  # Filter out any invalid indices
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def __len__(self) -> int:
        """Return the number of vectors in the store"""
        return len(self.data_store)

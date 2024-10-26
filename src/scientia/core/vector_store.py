import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(
        self,
        dimension: int,
        collection_name: str = "scientia_knowledge",
        persist_directory: str = "knowledge_base"
    ):
        # Ensure the persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except InvalidCollectionException:
            logger.info(f"Creating new collection: {collection_name}")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"dimension": dimension}
            )
            # Initialize with a dummy document to prevent future errors
            self.collection.add(
                embeddings=[[0.0] * dimension],
                documents=["initialization document"],
                metadatas=[{"type": "system", "initialization": True}],
                ids=["init_doc"]
            )
            logger.info("Collection initialized with dummy document")

    async def add_vector(self, embedding: np.ndarray, knowledge_packet: Any) -> None:
        """Add a vector and its associated knowledge to the store"""
        try:
            # Convert numpy array to list for ChromaDB
            embedding_list = embedding.tolist()
            
            # Convert knowledge packet to metadata
            metadata = {
                "source_type": knowledge_packet.source_type,
                "timestamp": knowledge_packet.timestamp,
                "confidence": float(knowledge_packet.confidence),
                "privacy_level": knowledge_packet.privacy_level,
                "metadata": json.dumps(knowledge_packet.metadata)  # Serialize nested metadata
            }
            
            # Generate a unique ID based on the content hash
            doc_id = knowledge_packet.context_hash
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding_list],
                documents=[knowledge_packet.content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added document with ID: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error adding vector to ChromaDB: {str(e)}")
            raise

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> List[Tuple[Any, float]]:
        """Search for similar vectors"""
        try:
            # Convert query vector to list
            query_vector_list = query_vector.tolist()
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_vector_list],
                n_results=k,
                where=filter_criteria  # Apply any filters
            )
            
            # Format results
            formatted_results = []
            if results['ids'][0]:  # Check if we have any results
                for idx in range(len(results['ids'][0])):
                    try:
                        # Safely get metadata
                        metadata = results['metadatas'][0][idx]
                        nested_metadata = json.loads(metadata.get('metadata', '{}'))
                        
                        # Create knowledge packet-like object for compatibility
                        knowledge_obj = type('KnowledgePacket', (), {
                            'content': results['documents'][0][idx],
                            'metadata': nested_metadata,
                            'source_type': metadata.get('source_type', 'UNKNOWN'),
                            'confidence': float(metadata.get('confidence', 0.0)),
                            'privacy_level': metadata.get('privacy_level', 'PUBLIC'),
                            'timestamp': metadata.get('timestamp', '')
                        })
                        
                        # Get distance score (ChromaDB returns distances)
                        similarity = 1.0 - float(results['distances'][0][idx])
                        
                        formatted_results.append((knowledge_obj, similarity))
                    except Exception as e:
                        logger.warning(f"Error formatting result {idx}: {str(e)}")
                        continue
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {str(e)}")
            raise

    async def get_all(self) -> List[Any]:
        """Retrieve all documents from the store"""
        try:
            results = self.collection.get()
            
            # Format results
            documents = []
            for idx in range(len(results['ids'])):
                knowledge_obj = type('KnowledgePacket', (), {
                    'content': results['documents'][idx],
                    'metadata': json.loads(results['metadatas'][idx]['metadata']),
                    'source_type': results['metadatas'][idx]['source_type'],
                    'confidence': results['metadatas'][idx]['confidence'],
                    'privacy_level': results['metadatas'][idx]['privacy_level'],
                    'timestamp': results['metadatas'][idx]['timestamp']
                })
                documents.append(knowledge_obj)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving all documents from ChromaDB: {str(e)}")
            raise

    def __len__(self) -> int:
        """Get the number of documents in the store"""
        return self.collection.count()

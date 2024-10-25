from dataclasses import dataclass
from datetime import datetime
from typing import Dict
import numpy as np

@dataclass
class KnowledgePacket:
    """Represents a unit of knowledge in the Scientia system"""
    content: str
    embeddings: np.ndarray
    metadata: Dict
    source_type: str  # 'conversation', 'document', 'shared'
    timestamp: datetime
    confidence: float
    context_hash: str
    privacy_level: str  # 'public', 'private', 'strictly_private'

@dataclass
class SearchResult:
    """Represents a search result from the vector store"""
    content: str
    metadata: Dict
    score: float
    source_type: str
    timestamp: str

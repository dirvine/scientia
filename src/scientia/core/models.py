"""
Data models for the Scientia system
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

@dataclass
class KnowledgePacket:
    """Container for knowledge with associated metadata"""
    content: str
    embeddings: Optional[np.ndarray]
    source_type: str
    timestamp: str
    confidence: float
    context_hash: str
    privacy_level: str
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """Represents a search result from the vector store"""
    content: str
    metadata: Dict
    score: float
    source_type: str
    timestamp: str

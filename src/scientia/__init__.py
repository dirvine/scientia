"""
Scientia: An AI-powered knowledge exploration and management system
"""

from .core.knowledge_system import ScientiaCore, DocumentProcessor
from .core.models import KnowledgePacket

__version__ = "0.1.0"
__all__ = ["ScientiaCore", "KnowledgePacket", "DocumentProcessor"]

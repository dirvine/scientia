import numpy as np
from typing import Union, List, Dict, Any
import struct
import zlib
from dataclasses import dataclass
import pickle
from collections import defaultdict
import mmh3  # MurmurHash3 for efficient hashing

@dataclass
class NeuralToken:
    """Represents a fundamental unit of encoded information"""
    vector: np.ndarray
    frequency: float
    context_window: int
    positional_encoding: np.ndarray
    attention_mask: np.ndarray

class CrossModelEncoder:
    def __init__(self, 
                 dimension: int = 1024,
                 max_sequence_length: int = 8192,
                 compression_level: int = 9):
        """
        Initialize the cross-model encoder
        
        Args:
            dimension: Base dimensionality for neural representations
            max_sequence_length: Maximum sequence length to consider
            compression_level: Level of compression for the binary format
        """
        self.dimension = dimension
        self.max_sequence_length = max_sequence_length
        self.compression_level = compression_level
        
        # Initialize positional encoding matrix
        self.position_encoding = self._create_positional_encoding()
        
        # Token frequency dictionary for adaptive compression
        self.token_frequencies = defaultdict(float)
        
        # Cached neural patterns for frequent tokens
        self.pattern_cache = {}
        
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encodings"""
        position = np.arange(self.max_sequence_length)[:, np.newaxis]
        dim_pos = np.arange(self.dimension)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dim_pos // 2)) / self.dimension)
        pos_encoding = position * angle_rates
        
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        return pos_encoding

    def _hash_to_vector(self, content: str) -> np.ndarray:
        """Convert content to a stable neural representation using locality-sensitive hashing"""
        # Use multiple hash seeds for better distribution
        hash_values = [mmh3.hash(content, seed=i) for i in range(self.dimension)]
        
        # Convert hash values to normalized vectors
        vector = np.array(hash_values, dtype=np.float32)
        vector = (vector - np.mean(vector)) / np.std(vector)
        
        return vector

    def _compress_vector(self, vector: np.ndarray) -> bytes:
        """Compress vector to binary format"""
        # Quantize to 16-bit floats for efficiency
        quantized = vector.astype(np.float16)
        
        # Custom binary format: [dimension(4 bytes)][data(2*dimension bytes)]
        header = struct.pack('I', len(vector))
        data = quantized.tobytes()
        
        # Compress using zlib
        return zlib.compress(header + data, level=self.compression_level)

    def _create_attention_mask(self, content_length: int) -> np.ndarray:
        """Create causal attention mask"""
        mask = np.triu(np.ones((content_length, content_length)), k=1)
        return mask.astype(np.float32) * -1e9

    def encode_content(self, content: Union[str, List[str]]) -> bytes:
        """Encode content into cross-model neural representation"""
        if isinstance(content, str):
            content = [content]
            
        encoded_tokens = []
        
        for text in content:
            # Create base neural representation
            base_vector = self._hash_to_vector(text)
            
            # Update token frequencies
            self.token_frequencies[text] += 1
            
            # Create neural token
            token = NeuralToken(
                vector=base_vector,
                frequency=self.token_frequencies[text],
                context_window=min(len(text), self.max_sequence_length),
                positional_encoding=self.position_encoding[:len(text)],
                attention_mask=self._create_attention_mask(len(text))
            )
            
            encoded_tokens.append(token)
            
        # Serialize and compress the encoded tokens
        return self._compress_tokens(encoded_tokens)
    
    def _compress_tokens(self, tokens: List[NeuralToken]) -> bytes:
        """Compress neural tokens into efficient binary format"""
        compressed_data = []
        
        for token in tokens:
            # Combine all neural representations
            combined = np.concatenate([
                token.vector,
                token.positional_encoding.flatten(),
                token.attention_mask.flatten()
            ])
            
            # Add frequency information
            freq_bytes = struct.pack('f', token.frequency)
            
            # Compress the combined representation
            compressed = self._compress_vector(combined)
            
            # Format: [freq(4 bytes)][compressed_size(4 bytes)][compressed_data]
            size_bytes = struct.pack('I', len(compressed))
            compressed_data.extend([freq_bytes, size_bytes, compressed])
            
        return b''.join(compressed_data)
    
    def decode_binary(self, binary_data: bytes) -> List[NeuralToken]:
        """Decode binary format back into neural tokens"""
        tokens = []
        offset = 0
        
        while offset < len(binary_data):
            # Read frequency
            freq = struct.unpack('f', binary_data[offset:offset + 4])[0]
            offset += 4
            
            # Read compressed data size
            size = struct.unpack('I', binary_data[offset:offset + 4])[0]
            offset += 4
            
            # Read and decompress data
            compressed = binary_data[offset:offset + size]
            decompressed = zlib.decompress(compressed)
            
            # Parse vector dimension
            vec_dim = struct.unpack('I', decompressed[:4])[0]
            vector_data = np.frombuffer(decompressed[4:], dtype=np.float16)
            
            # Reconstruct neural token
            vector_size = self.dimension
            pos_enc_size = self.max_sequence_length * self.dimension
            attention_size = self.max_sequence_length * self.max_sequence_length
            
            token = NeuralToken(
                vector=vector_data[:vector_size].astype(np.float32),
                frequency=freq,
                context_window=self.max_sequence_length,
                positional_encoding=vector_data[vector_size:vector_size + pos_enc_size].reshape(
                    self.max_sequence_length, self.dimension),
                attention_mask=vector_data[-attention_size:].reshape(
                    self.max_sequence_length, self.max_sequence_length)
            )
            
            tokens.append(token)
            offset += size
            
        return tokens
    
    def get_binary_stats(self, binary_data: bytes) -> Dict[str, Any]:
        """Get statistics about the binary encoded data"""
        tokens = self.decode_binary(binary_data)
        return {
            'num_tokens': len(tokens),
            'total_size_bytes': len(binary_data),
            'avg_bytes_per_token': len(binary_data) / len(tokens),
            'compression_ratio': sum(len(t.vector.tobytes()) for t in tokens) / len(binary_data)
        }
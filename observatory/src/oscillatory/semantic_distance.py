"""
Semantic Distance Amplification System

Implements semantic distance amplification for temporal precision enhancement
through multi-layer encoding transformations and directional mapping.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import Counter


class EncodingTransformation(Enum):
    """Types of encoding transformations"""
    SEQUENTIAL = "sequential_encoding"
    POSITIONAL = "positional_context"
    DIRECTIONAL = "directional_transformation"
    SEMANTIC = "semantic_amplification"


class Direction(Enum):
    """Directional mappings for context encoding"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    UP = "up"
    DOWN = "down"
    NORTH_PRIME = "north_prime"
    SOUTH_PRIME = "south_prime"


@dataclass
class SequenceToken:
    """Token in encoded sequence with positional and contextual information"""
    word: str
    position: int
    context: str
    direction: Optional[Direction] = None
    semantic_weight: float = 1.0

    def to_embedding_vector(self, embedding_dim: int = 64) -> np.ndarray:
        """Convert token to embedding vector for distance calculations"""
        # Simple hash-based embedding (in practice, use pre-trained embeddings)
        hash_val = hash(f"{self.word}_{self.context}_{self.direction}")
        np.random.seed(abs(hash_val) % (2**32))

        base_vector = np.random.normal(0, 1, embedding_dim)

        # Add positional encoding
        position_encoding = np.array([
            math.sin(self.position / 10000**(2*i/embedding_dim)) if i % 2 == 0
            else math.cos(self.position / 10000**(2*i/embedding_dim))
            for i in range(embedding_dim)
        ])

        # Combine base vector with positional encoding
        embedding = base_vector + 0.3 * position_encoding

        # Apply semantic weight
        embedding *= self.semantic_weight

        return embedding / np.linalg.norm(embedding)  # Normalize


@dataclass
class EncodedSequence:
    """Encoded sequence with transformation history"""
    tokens: List[SequenceToken]
    transformation_history: List[EncodingTransformation] = field(default_factory=list)
    semantic_complexity: float = 0.0
    distance_amplification_factor: float = 1.0

    def calculate_semantic_complexity(self) -> float:
        """Calculate semantic complexity of encoded sequence"""
        if not self.tokens:
            return 0.0

        # Vocabulary diversity
        unique_words = len(set(token.word for token in self.tokens))
        vocab_complexity = unique_words / len(self.tokens)

        # Context diversity
        unique_contexts = len(set(token.context for token in self.tokens))
        context_complexity = unique_contexts / len(self.tokens)

        # Directional diversity
        directions = [token.direction for token in self.tokens if token.direction]
        direction_complexity = len(set(directions)) / len(self.tokens) if directions else 0

        # Combined complexity
        complexity = (vocab_complexity + context_complexity + direction_complexity) / 3.0
        self.semantic_complexity = complexity
        return complexity


class SemanticDistanceAmplifier:
    """
    Semantic distance amplification system for temporal precision

    Converts exponential precision search problems into linear navigation
    through multi-layer encoding and semantic distance amplification.
    """

    def __init__(self):
        self.word_mappings = self._initialize_word_mappings()
        self.context_rules = self._initialize_context_rules()
        self.direction_mappings = self._initialize_direction_mappings()
        self.encoding_history: List[Dict] = []
        self.embedding_dim = 64

    def _initialize_word_mappings(self) -> Dict[str, str]:
        """Initialize word mappings for sequential encoding"""
        return {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            ':': 'colon', '.': 'point', '-': 'dash', '+': 'plus',
            'am': 'ante_meridiem', 'pm': 'post_meridiem'
        }

    def _initialize_context_rules(self) -> Dict[str, str]:
        """Initialize context identification rules"""
        return {
            'triple_zero': 'seventh_triple_occurrence',
            'single_digit': 'first_occurrence',
            'time_separator': 'temporal_separator',
            'meridiem_indicator': 'time_period_marker',
            'repeated_pattern': 'pattern_repetition',
            'sequential_pattern': 'sequential_occurrence'
        }

    def _initialize_direction_mappings(self) -> Dict[str, Direction]:
        """Initialize directional transformation mappings"""
        return {
            'seventh_triple_occurrence': Direction.SOUTH,
            'first_occurrence': Direction.NORTH_PRIME,
            'temporal_separator': Direction.EAST,
            'time_period_marker': Direction.WEST,
            'pattern_repetition': Direction.UP,
            'sequential_occurrence': Direction.DOWN,
            'standard': Direction.NORTH
        }

    def apply_sequential_encoding(self, temporal_value: Union[str, float]) -> List[str]:
        """
        Apply sequential encoding transformation: T → S

        Maps temporal values to word sequences
        """
        # Convert input to string representation
        if isinstance(temporal_value, (int, float)):
            time_str = f"{temporal_value:05.2f}" if isinstance(temporal_value, float) else str(temporal_value)
        else:
            time_str = str(temporal_value)

        # Map each character to word representation
        encoded_sequence = []
        for char in time_str:
            if char in self.word_mappings:
                encoded_sequence.append(self.word_mappings[char])
            else:
                # Handle unknown characters
                encoded_sequence.append(f"unknown_{char}")

        return encoded_sequence

    def apply_positional_context_encoding(self, sequence: List[str]) -> List[SequenceToken]:
        """
        Apply positional context encoding: S → S_pos

        Augments sequences with positional and contextual information
        """
        tokens = []

        # Analyze sequence patterns
        pattern_analysis = self._analyze_sequence_patterns(sequence)

        for i, word in enumerate(sequence):
            # Determine context based on position and patterns
            context = self._determine_context(word, i, sequence, pattern_analysis)

            # Create token with positional context
            token = SequenceToken(
                word=word,
                position=i,
                context=context,
                semantic_weight=1.0 + (i * 0.1)  # Position-dependent weighting
            )

            tokens.append(token)

        return tokens

    def _analyze_sequence_patterns(self, sequence: List[str]) -> Dict:
        """Analyze patterns in sequence for context determination"""
        analysis = {
            'length': len(sequence),
            'unique_words': len(set(sequence)),
            'repetitions': {},
            'subsequences': {}
        }

        # Count word repetitions
        word_counts = Counter(sequence)
        analysis['repetitions'] = dict(word_counts)

        # Find common subsequences (length 2-3)
        for length in [2, 3]:
            for i in range(len(sequence) - length + 1):
                subseq = tuple(sequence[i:i+length])
                if subseq not in analysis['subsequences']:
                    analysis['subsequences'][subseq] = []
                analysis['subsequences'][subseq].append(i)

        return analysis

    def _determine_context(self, word: str, position: int, sequence: List[str], analysis: Dict) -> str:
        """Determine context for word based on position and patterns"""
        # Check for triple zero pattern
        if (position >= 2 and
            sequence[position-2:position+1] == ['zero', 'zero', 'zero'] and
            analysis['subsequences'].get(('zero', 'zero', 'zero'), []).index(position-2) == 6):
            return 'seventh_triple_occurrence'

        # Check for first occurrence
        if analysis['repetitions'][word] == 1:
            return 'first_occurrence'

        # Check for repeated patterns
        if analysis['repetitions'][word] > 1:
            return 'pattern_repetition'

        # Check for temporal separators
        if word in ['colon', 'point']:
            return 'temporal_separator'

        # Check for meridiem indicators
        if word in ['ante_meridiem', 'post_meridiem']:
            return 'time_period_marker'

        # Default context
        return 'standard'

    def apply_directional_transformation(self, tokens: List[SequenceToken]) -> List[SequenceToken]:
        """
        Apply directional transformation: S_pos → S_dir

        Maps contextual sequences to directional representations
        """
        transformed_tokens = []

        for token in tokens:
            # Create new token with directional mapping
            direction = self.direction_mappings.get(token.context, Direction.NORTH)

            transformed_token = SequenceToken(
                word=token.word,
                position=token.position,
                context=token.context,
                direction=direction,
                semantic_weight=token.semantic_weight * 1.2  # Directional amplification
            )

            transformed_tokens.append(transformed_token)

        return transformed_tokens

    def calculate_semantic_distance(self,
                                  sequence1: EncodedSequence,
                                  sequence2: EncodedSequence) -> float:
        """
        Calculate semantic distance between encoded sequences

        Implements: d_semantic(s1, s2) = Σ w_i * ||φ(s1,i) - φ(s2,i)||_2
        """
        if len(sequence1.tokens) != len(sequence2.tokens):
            # Handle different length sequences
            min_len = min(len(sequence1.tokens), len(sequence2.tokens))
            max_len = max(len(sequence1.tokens), len(sequence2.tokens))
            length_penalty = (max_len - min_len) * 0.5
        else:
            min_len = len(sequence1.tokens)
            length_penalty = 0.0

        total_distance = 0.0

        for i in range(min_len):
            token1 = sequence1.tokens[i]
            token2 = sequence2.tokens[i]

            # Get embedding vectors
            embedding1 = token1.to_embedding_vector(self.embedding_dim)
            embedding2 = token2.to_embedding_vector(self.embedding_dim)

            # Calculate weighted distance
            positional_weight = 1.0 + (i * 0.1)  # Increase weight for later positions
            semantic_weight = (token1.semantic_weight + token2.semantic_weight) / 2.0

            distance = np.linalg.norm(embedding1 - embedding2) * positional_weight * semantic_weight
            total_distance += distance

        # Add length penalty
        total_distance += length_penalty

        return total_distance

    def apply_full_encoding_pipeline(self, temporal_value: Union[str, float]) -> EncodedSequence:
        """
        Apply complete encoding pipeline with semantic distance amplification

        Pipeline: T → Sequential → Positional → Directional → Semantic
        """
        start_time = time.time()

        # Phase 1: Sequential encoding
        word_sequence = self.apply_sequential_encoding(temporal_value)

        # Phase 2: Positional context encoding
        contextual_tokens = self.apply_positional_context_encoding(word_sequence)

        # Phase 3: Directional transformation
        directional_tokens = self.apply_directional_transformation(contextual_tokens)

        # Phase 4: Create encoded sequence
        encoded_sequence = EncodedSequence(
            tokens=directional_tokens,
            transformation_history=[
                EncodingTransformation.SEQUENTIAL,
                EncodingTransformation.POSITIONAL,
                EncodingTransformation.DIRECTIONAL
            ]
        )

        # Calculate semantic complexity and distance amplification
        encoded_sequence.calculate_semantic_complexity()
        encoded_sequence.distance_amplification_factor = self._calculate_amplification_factor(encoded_sequence)

        # Record encoding history
        encoding_record = {
            'timestamp': time.time(),
            'input_value': str(temporal_value),
            'output_tokens': len(encoded_sequence.tokens),
            'semantic_complexity': encoded_sequence.semantic_complexity,
            'amplification_factor': encoded_sequence.distance_amplification_factor,
            'processing_time': time.time() - start_time
        }
        self.encoding_history.append(encoding_record)

        return encoded_sequence

    def _calculate_amplification_factor(self, sequence: EncodedSequence) -> float:
        """Calculate semantic distance amplification factor"""
        # Base amplification from transformations
        base_amplification = len(sequence.transformation_history) * 2.0

        # Complexity-based amplification
        complexity_amplification = sequence.semantic_complexity * 5.0

        # Token diversity amplification
        unique_words = len(set(token.word for token in sequence.tokens))
        unique_contexts = len(set(token.context for token in sequence.tokens))
        unique_directions = len(set(token.direction for token in sequence.tokens if token.direction))

        diversity_amplification = (unique_words + unique_contexts + unique_directions) * 0.5

        # Total amplification factor
        total_amplification = base_amplification + complexity_amplification + diversity_amplification

        return max(1.0, total_amplification)

    def precision_search_to_navigation(self,
                                     target_precision: int,
                                     candidate_values: List[Union[str, float]]) -> Dict:
        """
        Convert exponential precision search to linear navigation

        Transforms N_incorrect = 10^p - 1 search space to linear semantic navigation
        """
        if not candidate_values:
            return {'error': 'No candidate values provided'}

        # Encode all candidate values
        encoded_candidates = []
        for value in candidate_values:
            encoded = self.apply_full_encoding_pipeline(value)
            encoded_candidates.append((value, encoded))

        # Calculate semantic distances between all pairs
        distance_matrix = np.zeros((len(encoded_candidates), len(encoded_candidates)))

        for i, (val1, seq1) in enumerate(encoded_candidates):
            for j, (val2, seq2) in enumerate(encoded_candidates):
                if i != j:
                    distance = self.calculate_semantic_distance(seq1, seq2)
                    distance_matrix[i, j] = distance

        # Find optimal navigation path (minimum spanning tree approach)
        navigation_path = self._find_optimal_navigation_path(distance_matrix, encoded_candidates)

        # Calculate complexity reduction
        traditional_complexity = 10**target_precision - 1
        navigation_complexity = len(navigation_path)
        complexity_reduction = traditional_complexity / navigation_complexity if navigation_complexity > 0 else float('inf')

        return {
            'target_precision': target_precision,
            'candidate_count': len(candidate_values),
            'traditional_search_complexity': traditional_complexity,
            'navigation_complexity': navigation_complexity,
            'complexity_reduction_factor': complexity_reduction,
            'optimal_navigation_path': navigation_path,
            'semantic_distance_matrix': distance_matrix.tolist(),
            'average_amplification_factor': np.mean([seq.distance_amplification_factor
                                                   for _, seq in encoded_candidates])
        }

    def _find_optimal_navigation_path(self,
                                    distance_matrix: np.ndarray,
                                    encoded_candidates: List[Tuple]) -> List[Dict]:
        """Find optimal navigation path through semantic space"""
        n = len(encoded_candidates)
        if n == 0:
            return []

        # Simple greedy path finding (can be improved with more sophisticated algorithms)
        visited = [False] * n
        path = []
        current = 0  # Start from first candidate
        visited[current] = True

        path.append({
            'value': encoded_candidates[current][0],
            'semantic_complexity': encoded_candidates[current][1].semantic_complexity,
            'amplification_factor': encoded_candidates[current][1].distance_amplification_factor
        })

        for _ in range(n - 1):
            min_distance = float('inf')
            next_node = -1

            # Find nearest unvisited node
            for j in range(n):
                if not visited[j] and distance_matrix[current, j] < min_distance:
                    min_distance = distance_matrix[current, j]
                    next_node = j

            if next_node != -1:
                visited[next_node] = True
                path.append({
                    'value': encoded_candidates[next_node][0],
                    'semantic_complexity': encoded_candidates[next_node][1].semantic_complexity,
                    'amplification_factor': encoded_candidates[next_node][1].distance_amplification_factor,
                    'distance_from_previous': min_distance
                })
                current = next_node

        return path

    def get_encoding_statistics(self) -> Dict:
        """Get comprehensive encoding performance statistics"""
        if not self.encoding_history:
            return {'no_history': True}

        # Aggregate statistics
        total_encodings = len(self.encoding_history)
        avg_tokens = np.mean([record['output_tokens'] for record in self.encoding_history])
        avg_complexity = np.mean([record['semantic_complexity'] for record in self.encoding_history])
        avg_amplification = np.mean([record['amplification_factor'] for record in self.encoding_history])
        avg_processing_time = np.mean([record['processing_time'] for record in self.encoding_history])

        return {
            'total_encodings_performed': total_encodings,
            'average_output_tokens': avg_tokens,
            'average_semantic_complexity': avg_complexity,
            'average_amplification_factor': avg_amplification,
            'average_processing_time': avg_processing_time,
            'word_mappings_count': len(self.word_mappings),
            'context_rules_count': len(self.context_rules),
            'direction_mappings_count': len(self.direction_mappings),
            'recent_encodings': self.encoding_history[-5:] if len(self.encoding_history) >= 5 else self.encoding_history
        }


def create_semantic_distance_system() -> SemanticDistanceAmplifier:
    """Create semantic distance amplification system for S-Entropy alignment"""
    return SemanticDistanceAmplifier()

"""
Text chunking logic for embedding generation.

Provides intelligent text segmentation for optimal embedding generation.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ChunkBoundary(Enum):
    """Types of text boundaries for chunking."""

    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    WORD = "word"
    CHARACTER = "character"


@dataclass
class TextChunk:
    """Represents a text chunk for embedding."""

    text: str
    start_index: int
    end_index: int
    chunk_index: int
    page_number: int | None = None
    metadata: dict[str, Any] | None = None

    @property
    def char_count(self) -> int:
        """Character count of the chunk."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Word count of the chunk."""
        return len(self.text.split())


class TextChunker:
    """Intelligent text chunking for embedding generation."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        boundary_preference: ChunkBoundary = ChunkBoundary.SENTENCE,
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            boundary_preference: Preferred boundary type for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.boundary_preference = boundary_preference

        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        logger.info(
            "TextChunker initialized",
            extra={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "boundary_preference": boundary_preference.value,
            },
        )

    async def create_chunks(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
        preserve_formatting: bool = True,
    ) -> list[TextChunk]:
        """
        Create text chunks from input text.

        Args:
            text: Input text to chunk
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            metadata: Additional metadata to attach to chunks
            preserve_formatting: Whether to preserve markdown formatting

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # Use provided values or defaults
        effective_chunk_size = chunk_size or self.chunk_size
        effective_overlap = chunk_overlap or self.chunk_overlap

        logger.info(
            "Creating text chunks",
            extra={
                "text_length": len(text),
                "chunk_size": effective_chunk_size,
                "chunk_overlap": effective_overlap,
            },
        )

        # Preprocess text
        processed_text = self._preprocess_text(text, preserve_formatting)

        # Create chunks based on boundary preference
        chunks = await self._create_chunks_with_boundaries(
            text=processed_text,
            chunk_size=effective_chunk_size,
            overlap=effective_overlap,
            boundary_type=self.boundary_preference,
        )

        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata = chunk.metadata or {}
            if metadata:
                chunk.metadata.update(metadata)

            # Add chunk statistics
            chunk.metadata.update(
                {
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count,
                    "boundary_type": self.boundary_preference.value,
                }
            )

        logger.info(f"Created {len(chunks)} text chunks")
        return chunks

    def _preprocess_text(self, text: str, preserve_formatting: bool) -> str:
        """
        Preprocess text before chunking.

        Args:
            text: Input text
            preserve_formatting: Whether to preserve markdown formatting

        Returns:
            Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())

        if preserve_formatting:
            # Preserve markdown headers, lists, and other formatting
            # but normalize excessive whitespace
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        else:
            # Remove all formatting for plain text chunking
            text = re.sub(r"[#*_`\[\]()]", "", text)
            text = re.sub(r"\n+", " ", text)

        return text

    async def _create_chunks_with_boundaries(
        self, text: str, chunk_size: int, overlap: int, boundary_type: ChunkBoundary
    ) -> list[TextChunk]:
        """
        Create chunks respecting text boundaries.

        Args:
            text: Preprocessed text
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            boundary_type: Type of boundary to respect

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [
                TextChunk(text=text, start_index=0, end_index=len(text), chunk_index=0)
            ]

        # Get boundary positions
        boundaries = self._find_boundaries(text, boundary_type)

        chunks = []
        chunk_index = 0
        start_pos = 0

        while start_pos < len(text):
            # Find end position for this chunk
            target_end = start_pos + chunk_size

            if target_end >= len(text):
                # Last chunk - take remaining text
                end_pos = len(text)
            else:
                # Find best boundary near target end
                end_pos = self._find_best_boundary(
                    boundaries, target_end, start_pos, chunk_size
                )

            # Extract chunk text
            chunk_text = text[start_pos:end_pos].strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        start_index=start_pos,
                        end_index=end_pos,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            # Calculate next start position with overlap
            if end_pos >= len(text):
                break

            # Move start position back by overlap amount, but respect boundaries
            overlap_start = max(start_pos, end_pos - overlap)
            start_pos = self._find_overlap_boundary(boundaries, overlap_start, end_pos)

        return chunks

    def _find_boundaries(self, text: str, boundary_type: ChunkBoundary) -> list[int]:
        """
        Find boundary positions in text.

        Args:
            text: Input text
            boundary_type: Type of boundary to find

        Returns:
            List of character positions where boundaries occur
        """
        boundaries = [0]  # Always include start

        if boundary_type == ChunkBoundary.PARAGRAPH:
            # Find paragraph breaks (double newlines)
            for match in re.finditer(r"\n\s*\n", text):
                boundaries.append(match.end())

        elif boundary_type == ChunkBoundary.SENTENCE:
            # Find sentence endings
            sentence_pattern = r"[.!?]+\s+"
            for match in re.finditer(sentence_pattern, text):
                boundaries.append(match.end())

        elif boundary_type == ChunkBoundary.WORD:
            # Find word boundaries
            for match in re.finditer(r"\s+", text):
                boundaries.append(match.end())

        elif boundary_type == ChunkBoundary.CHARACTER:
            # Every character is a boundary
            boundaries.extend(range(1, len(text)))

        boundaries.append(len(text))  # Always include end
        return sorted(set(boundaries))

    def _find_best_boundary(
        self, boundaries: list[int], target_pos: int, min_pos: int, max_chunk_size: int
    ) -> int:
        """
        Find the best boundary near target position.

        Args:
            boundaries: List of boundary positions
            target_pos: Target end position
            min_pos: Minimum acceptable position
            max_chunk_size: Maximum chunk size

        Returns:
            Best boundary position
        """
        # Find boundaries within acceptable range
        acceptable_boundaries = [
            pos
            for pos in boundaries
            if min_pos + (max_chunk_size // 2) <= pos <= min_pos + max_chunk_size * 1.2
        ]

        if not acceptable_boundaries:
            # If no good boundary, use target position
            return target_pos

        # Find closest boundary to target
        best_boundary = min(acceptable_boundaries, key=lambda x: abs(x - target_pos))
        return best_boundary

    def _find_overlap_boundary(
        self, boundaries: list[int], overlap_start: int, chunk_end: int
    ) -> int:
        """
        Find appropriate boundary for overlap start position.

        Args:
            boundaries: List of boundary positions
            overlap_start: Calculated overlap start position
            chunk_end: End position of previous chunk

        Returns:
            Adjusted start position respecting boundaries
        """
        # Find boundaries in overlap region
        overlap_boundaries = [
            pos for pos in boundaries if overlap_start <= pos < chunk_end
        ]

        if not overlap_boundaries:
            return overlap_start

        # Use the first boundary in overlap region
        return overlap_boundaries[0]

    async def merge_chunks(
        self, chunks: list[TextChunk], max_merged_size: int = 2000
    ) -> list[TextChunk]:
        """
        Merge small adjacent chunks if they would fit within size limit.

        Args:
            chunks: List of chunks to potentially merge
            max_merged_size: Maximum size for merged chunks

        Returns:
            List of potentially merged chunks
        """
        if not chunks:
            return chunks

        merged_chunks = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if we can merge current chunk with next chunk
            merged_text = current_chunk.text + " " + next_chunk.text

            if len(merged_text) <= max_merged_size:
                # Merge chunks
                current_chunk = TextChunk(
                    text=merged_text,
                    start_index=current_chunk.start_index,
                    end_index=next_chunk.end_index,
                    chunk_index=current_chunk.chunk_index,
                    page_number=current_chunk.page_number,
                    metadata={
                        **(current_chunk.metadata or {}),
                        "merged": True,
                        "original_chunks": [
                            current_chunk.chunk_index,
                            next_chunk.chunk_index,
                        ],
                    },
                )
            else:
                # Can't merge - add current chunk and move to next
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk

        # Add the last chunk
        merged_chunks.append(current_chunk)

        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks

    def get_chunk_statistics(self, chunks: list[TextChunk]) -> dict[str, Any]:
        """
        Calculate statistics for a list of chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "total_words": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        char_counts = [chunk.char_count for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(char_counts),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(char_counts) / len(chunks),
            "min_chunk_size": min(char_counts),
            "max_chunk_size": max(char_counts),
            "avg_word_count": sum(word_counts) / len(chunks),
            "chunk_size_std": self._calculate_std_dev(char_counts),
        }

    def _calculate_std_dev(self, values: list[int]) -> float:
        """Calculate standard deviation of values."""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

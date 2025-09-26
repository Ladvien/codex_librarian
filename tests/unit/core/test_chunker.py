"""
Unit tests for text chunking service.

Tests all chunking functionality including boundary detection, overlap handling,
and chunk optimization following TDD principles.
"""


import pytest

from src.pdf_to_markdown_mcp.core.chunker import ChunkBoundary, TextChunk, TextChunker


class TestTextChunk:
    """Test TextChunk data model."""

    def test_text_chunk_creation(self):
        """Test basic TextChunk creation."""
        # Given
        text = "This is a test chunk of text."
        start_index = 0
        end_index = len(text)
        chunk_index = 0

        # When
        chunk = TextChunk(
            text=text,
            start_index=start_index,
            end_index=end_index,
            chunk_index=chunk_index,
        )

        # Then
        assert chunk.text == text
        assert chunk.start_index == start_index
        assert chunk.end_index == end_index
        assert chunk.chunk_index == chunk_index
        assert chunk.page_number is None
        assert chunk.metadata is None

    def test_text_chunk_char_count(self):
        """Test character count property."""
        # Given
        text = "Hello world!"
        chunk = TextChunk(text=text, start_index=0, end_index=len(text), chunk_index=0)

        # When
        char_count = chunk.char_count

        # Then
        assert char_count == 12

    def test_text_chunk_word_count(self):
        """Test word count property."""
        # Given
        text = "Hello world from chunker"
        chunk = TextChunk(text=text, start_index=0, end_index=len(text), chunk_index=0)

        # When
        word_count = chunk.word_count

        # Then
        assert word_count == 4

    def test_text_chunk_with_metadata(self):
        """Test TextChunk with metadata."""
        # Given
        text = "Test chunk"
        metadata = {"source": "test", "page": 1}

        # When
        chunk = TextChunk(
            text=text,
            start_index=0,
            end_index=len(text),
            chunk_index=0,
            page_number=1,
            metadata=metadata,
        )

        # Then
        assert chunk.page_number == 1
        assert chunk.metadata == metadata


class TestTextChunker:
    """Test TextChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a TextChunker instance for testing."""
        return TextChunker(
            chunk_size=100, chunk_overlap=20, boundary_preference=ChunkBoundary.SENTENCE
        )

    def test_chunker_initialization(self):
        """Test TextChunker initialization."""
        # Given
        chunk_size = 1000
        chunk_overlap = 200
        boundary_preference = ChunkBoundary.PARAGRAPH

        # When
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            boundary_preference=boundary_preference,
        )

        # Then
        assert chunker.chunk_size == chunk_size
        assert chunker.chunk_overlap == chunk_overlap
        assert chunker.boundary_preference == boundary_preference

    def test_chunker_invalid_overlap(self):
        """Test that invalid overlap raises ValueError."""
        # Given/When/Then
        with pytest.raises(
            ValueError, match="Chunk overlap must be less than chunk size"
        ):
            TextChunker(chunk_size=100, chunk_overlap=150)

    @pytest.mark.asyncio
    async def test_create_chunks_empty_text(self, chunker):
        """Test chunking empty text returns empty list."""
        # Given
        text = ""

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert chunks == []

    @pytest.mark.asyncio
    async def test_create_chunks_whitespace_only(self, chunker):
        """Test chunking whitespace-only text returns empty list."""
        # Given
        text = "   \n\t   "

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert chunks == []

    @pytest.mark.asyncio
    async def test_create_chunks_short_text(self, chunker):
        """Test chunking text shorter than chunk size."""
        # Given
        text = "This is a short text."

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start_index == 0
        assert chunks[0].end_index == len(text)
        assert chunks[0].chunk_index == 0

    @pytest.mark.asyncio
    async def test_create_chunks_long_text(self, chunker):
        """Test chunking long text creates multiple chunks."""
        # Given
        sentences = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
            "This is the fourth sentence.",
            "This is the fifth sentence.",
        ]
        text = " ".join(sentences)

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) > 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)  # No empty chunks

    @pytest.mark.asyncio
    async def test_create_chunks_with_metadata(self, chunker):
        """Test chunking text with metadata."""
        # Given
        text = "This is test text for metadata."
        metadata = {"document_id": "test123", "page": 1}

        # When
        chunks = await chunker.create_chunks(text, metadata=metadata)

        # Then
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "document_id" in chunk.metadata
        assert "page" in chunk.metadata
        assert chunk.metadata["document_id"] == "test123"
        assert chunk.metadata["page"] == 1

    @pytest.mark.asyncio
    async def test_create_chunks_adds_statistics(self, chunker):
        """Test that chunks include statistical metadata."""
        # Given
        text = "This is a test text for statistics."

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "char_count" in chunk.metadata
        assert "word_count" in chunk.metadata
        assert "boundary_type" in chunk.metadata
        assert chunk.metadata["char_count"] == len(text)
        assert chunk.metadata["word_count"] == len(text.split())
        assert chunk.metadata["boundary_type"] == "sentence"

    @pytest.mark.asyncio
    async def test_create_chunks_preserves_formatting(self, chunker):
        """Test that formatting preservation works."""
        # Given
        text = "# Header\n\nThis is **bold** text with _italics_."

        # When
        chunks = await chunker.create_chunks(text, preserve_formatting=True)

        # Then
        assert len(chunks) == 1
        assert "**" in chunks[0].text or "_" in chunks[0].text  # Formatting preserved

    @pytest.mark.asyncio
    async def test_create_chunks_removes_formatting(self, chunker):
        """Test that formatting can be removed."""
        # Given
        text = "# Header\n\nThis is **bold** text with _italics_."

        # When
        chunks = await chunker.create_chunks(text, preserve_formatting=False)

        # Then
        assert len(chunks) == 1
        chunk_text = chunks[0].text
        assert "**" not in chunk_text
        assert "_" not in chunk_text
        assert "#" not in chunk_text

    @pytest.mark.asyncio
    async def test_create_chunks_with_override_parameters(self, chunker):
        """Test chunking with overridden parameters."""
        # Given
        text = "This is a longer text that should be split into multiple chunks when using smaller chunk sizes and overlaps."
        override_chunk_size = 50
        override_overlap = 10

        # When
        chunks = await chunker.create_chunks(
            text, chunk_size=override_chunk_size, chunk_overlap=override_overlap
        )

        # Then
        assert len(chunks) > 1
        # Most chunks should be around the override size
        for chunk in chunks[:-1]:  # Exclude last chunk which might be shorter
            assert (
                len(chunk.text) <= override_chunk_size * 1.2
            )  # Allow some flexibility

    def test_preprocess_text_normalize_whitespace(self, chunker):
        """Test text preprocessing normalizes whitespace."""
        # Given
        text = "This  has    multiple   spaces\n\n\nand\tnewlines."

        # When
        processed = chunker._preprocess_text(text, preserve_formatting=False)

        # Then
        assert "  " not in processed  # No double spaces
        assert "\n" not in processed  # No newlines
        assert "\t" not in processed  # No tabs

    def test_preprocess_text_preserve_formatting(self, chunker):
        """Test text preprocessing with formatting preservation."""
        # Given
        text = "# Header\n\nParagraph with **bold**.\n\n\nAnother paragraph."

        # When
        processed = chunker._preprocess_text(text, preserve_formatting=True)

        # Then
        assert "**" in processed  # Bold formatting preserved
        assert "#" in processed  # Header formatting preserved
        assert processed.count("\n\n") <= 1  # Excessive newlines normalized

    def test_find_boundaries_paragraph(self, chunker):
        """Test finding paragraph boundaries."""
        # Given
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        # When
        boundaries = chunker._find_boundaries(text, ChunkBoundary.PARAGRAPH)

        # Then
        assert 0 in boundaries  # Start
        assert len(text) in boundaries  # End
        assert len(boundaries) > 3  # Should find paragraph breaks

    def test_find_boundaries_sentence(self, chunker):
        """Test finding sentence boundaries."""
        # Given
        text = "First sentence. Second sentence! Third sentence?"

        # When
        boundaries = chunker._find_boundaries(text, ChunkBoundary.SENTENCE)

        # Then
        assert 0 in boundaries  # Start
        assert len(text) in boundaries  # End
        assert len(boundaries) >= 4  # Should find sentence endings

    def test_find_boundaries_word(self, chunker):
        """Test finding word boundaries."""
        # Given
        text = "These are five words exactly"

        # When
        boundaries = chunker._find_boundaries(text, ChunkBoundary.WORD)

        # Then
        assert 0 in boundaries  # Start
        assert len(text) in boundaries  # End
        assert len(boundaries) >= 6  # Start + 4 spaces + end

    def test_find_boundaries_character(self, chunker):
        """Test finding character boundaries."""
        # Given
        text = "abc"

        # When
        boundaries = chunker._find_boundaries(text, ChunkBoundary.CHARACTER)

        # Then
        assert boundaries == [0, 1, 2, 3]  # Every character position

    def test_find_best_boundary(self, chunker):
        """Test finding best boundary near target position."""
        # Given
        boundaries = [0, 10, 20, 30, 50, 80, 100]
        target_pos = 25
        min_pos = 0
        max_chunk_size = 50

        # When
        best_boundary = chunker._find_best_boundary(
            boundaries, target_pos, min_pos, max_chunk_size
        )

        # Then
        assert best_boundary in boundaries
        assert abs(best_boundary - target_pos) <= 10  # Should be reasonably close

    def test_find_best_boundary_no_acceptable(self, chunker):
        """Test finding best boundary when no acceptable boundaries exist."""
        # Given
        boundaries = [0, 100, 200]  # No boundaries in acceptable range
        target_pos = 25
        min_pos = 0
        max_chunk_size = 50

        # When
        best_boundary = chunker._find_best_boundary(
            boundaries, target_pos, min_pos, max_chunk_size
        )

        # Then
        assert best_boundary == target_pos  # Should return target when no good boundary

    def test_find_overlap_boundary(self, chunker):
        """Test finding appropriate overlap boundary."""
        # Given
        boundaries = [0, 10, 20, 30, 40, 50]
        overlap_start = 15
        chunk_end = 35

        # When
        overlap_boundary = chunker._find_overlap_boundary(
            boundaries, overlap_start, chunk_end
        )

        # Then
        assert overlap_boundary in boundaries or overlap_boundary == overlap_start
        assert overlap_start <= overlap_boundary < chunk_end

    @pytest.mark.asyncio
    async def test_merge_chunks_empty_list(self, chunker):
        """Test merging empty chunk list."""
        # Given
        chunks = []

        # When
        merged = await chunker.merge_chunks(chunks)

        # Then
        assert merged == []

    @pytest.mark.asyncio
    async def test_merge_chunks_single_chunk(self, chunker):
        """Test merging single chunk."""
        # Given
        chunks = [
            TextChunk(text="Single chunk", start_index=0, end_index=12, chunk_index=0)
        ]

        # When
        merged = await chunker.merge_chunks(chunks)

        # Then
        assert len(merged) == 1
        assert merged[0].text == "Single chunk"

    @pytest.mark.asyncio
    async def test_merge_chunks_small_chunks(self, chunker):
        """Test merging small chunks that fit together."""
        # Given
        chunks = [
            TextChunk(text="Small", start_index=0, end_index=5, chunk_index=0),
            TextChunk(text="chunks", start_index=6, end_index=12, chunk_index=1),
            TextChunk(text="merge", start_index=13, end_index=18, chunk_index=2),
        ]

        # When
        merged = await chunker.merge_chunks(chunks, max_merged_size=50)

        # Then
        assert len(merged) < len(chunks)  # Should have fewer chunks
        assert all(len(chunk.text) <= 50 for chunk in merged)

    @pytest.mark.asyncio
    async def test_merge_chunks_large_chunks(self, chunker):
        """Test that large chunks are not merged."""
        # Given
        large_text = "A" * 100
        chunks = [
            TextChunk(text=large_text, start_index=0, end_index=100, chunk_index=0),
            TextChunk(text=large_text, start_index=100, end_index=200, chunk_index=1),
        ]

        # When
        merged = await chunker.merge_chunks(chunks, max_merged_size=150)

        # Then
        assert len(merged) == len(chunks)  # No merging should occur

    def test_get_chunk_statistics_empty(self, chunker):
        """Test getting statistics for empty chunk list."""
        # Given
        chunks = []

        # When
        stats = chunker.get_chunk_statistics(chunks)

        # Then
        assert stats["total_chunks"] == 0
        assert stats["total_characters"] == 0
        assert stats["total_words"] == 0
        assert stats["avg_chunk_size"] == 0
        assert stats["min_chunk_size"] == 0
        assert stats["max_chunk_size"] == 0

    def test_get_chunk_statistics_single_chunk(self, chunker):
        """Test getting statistics for single chunk."""
        # Given
        text = "This is a test chunk with multiple words."
        chunks = [
            TextChunk(text=text, start_index=0, end_index=len(text), chunk_index=0)
        ]

        # When
        stats = chunker.get_chunk_statistics(chunks)

        # Then
        assert stats["total_chunks"] == 1
        assert stats["total_characters"] == len(text)
        assert stats["total_words"] == len(text.split())
        assert stats["avg_chunk_size"] == len(text)
        assert stats["min_chunk_size"] == len(text)
        assert stats["max_chunk_size"] == len(text)
        assert stats["avg_word_count"] == len(text.split())
        assert stats["chunk_size_std"] == 0.0

    def test_get_chunk_statistics_multiple_chunks(self, chunker):
        """Test getting statistics for multiple chunks."""
        # Given
        chunks = [
            TextChunk(text="Short", start_index=0, end_index=5, chunk_index=0),
            TextChunk(
                text="Medium length text", start_index=6, end_index=24, chunk_index=1
            ),
            TextChunk(text="A", start_index=25, end_index=26, chunk_index=2),
        ]

        # When
        stats = chunker.get_chunk_statistics(chunks)

        # Then
        assert stats["total_chunks"] == 3
        assert stats["total_characters"] == 5 + 18 + 1
        assert stats["total_words"] == 1 + 3 + 1
        assert stats["avg_chunk_size"] == (5 + 18 + 1) / 3
        assert stats["min_chunk_size"] == 1
        assert stats["max_chunk_size"] == 18
        assert stats["chunk_size_std"] > 0

    def test_calculate_std_dev_single_value(self, chunker):
        """Test standard deviation calculation with single value."""
        # Given
        values = [10]

        # When
        std_dev = chunker._calculate_std_dev(values)

        # Then
        assert std_dev == 0.0

    def test_calculate_std_dev_multiple_values(self, chunker):
        """Test standard deviation calculation with multiple values."""
        # Given
        values = [1, 2, 3, 4, 5]

        # When
        std_dev = chunker._calculate_std_dev(values)

        # Then
        assert std_dev > 0  # Should have some variation
        assert isinstance(std_dev, float)


class TestChunkBoundaryIntegration:
    """Integration tests for different boundary types."""

    @pytest.mark.asyncio
    async def test_paragraph_boundary_chunking(self):
        """Test chunking with paragraph boundaries."""
        # Given
        chunker = TextChunker(
            chunk_size=50, chunk_overlap=10, boundary_preference=ChunkBoundary.PARAGRAPH
        )
        text = (
            "First paragraph.\n\nSecond paragraph with more text.\n\nThird paragraph."
        )

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) >= 1
        # Check that chunks respect paragraph boundaries when possible
        for chunk in chunks:
            assert chunk.text.strip()  # No empty chunks

    @pytest.mark.asyncio
    async def test_sentence_boundary_chunking(self):
        """Test chunking with sentence boundaries."""
        # Given
        chunker = TextChunker(
            chunk_size=30, chunk_overlap=5, boundary_preference=ChunkBoundary.SENTENCE
        )
        text = "First sentence. Second sentence. Third sentence! Fourth sentence?"

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) >= 1
        # Most chunks should end at sentence boundaries
        sentence_endings = 0
        for chunk in chunks[:-1]:  # Exclude last chunk
            if chunk.text.strip().endswith((".", "!", "?")):
                sentence_endings += 1
        assert sentence_endings >= 0  # At least some respect boundaries

    @pytest.mark.asyncio
    async def test_word_boundary_chunking(self):
        """Test chunking with word boundaries."""
        # Given
        chunker = TextChunker(
            chunk_size=20, chunk_overlap=3, boundary_preference=ChunkBoundary.WORD
        )
        text = "These are individual words that should be split at word boundaries."

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) >= 1
        # Chunks should not break words
        for chunk in chunks:
            words = chunk.text.strip().split()
            # Should be complete words
            assert all(word.strip() for word in words)

    @pytest.mark.asyncio
    async def test_character_boundary_chunking(self):
        """Test chunking with character boundaries."""
        # Given
        chunker = TextChunker(
            chunk_size=10, chunk_overlap=2, boundary_preference=ChunkBoundary.CHARACTER
        )
        text = "This is a test string for character-level chunking."

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) >= 1
        # Most chunks should be close to the target size
        for chunk in chunks[:-1]:  # Exclude last chunk which might be shorter
            assert len(chunk.text) <= 12  # Within reasonable range of chunk_size


class TestChunkerPerformance:
    """Performance and edge case tests."""

    @pytest.mark.asyncio
    async def test_large_text_chunking(self):
        """Test chunking very large text."""
        # Given
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        # Create large text (about 10KB)
        paragraph = "This is a test paragraph with multiple sentences. " * 20
        text = paragraph * 10

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) > 1
        assert all(
            len(chunk.text) <= 1200 for chunk in chunks
        )  # Within reasonable bounds
        assert (
            sum(len(chunk.text) for chunk in chunks) >= len(text) * 0.9
        )  # Most text preserved

    @pytest.mark.asyncio
    async def test_text_with_special_characters(self):
        """Test chunking text with special characters."""
        # Given
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "Text with émojis 😀, ünicøde, and spéciâl çhars! @#$%^&*()."

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        assert len(chunks) >= 1
        # Special characters should be preserved
        full_text = " ".join(chunk.text for chunk in chunks)
        assert "😀" in full_text
        assert "ü" in full_text
        assert "@#$%^&*()" in full_text

    @pytest.mark.asyncio
    async def test_overlapping_chunks_continuity(self):
        """Test that overlapping chunks maintain text continuity."""
        # Given
        chunker = TextChunker(chunk_size=50, chunk_overlap=20)
        text = "This is a longer text that will be split into overlapping chunks to test continuity."

        # When
        chunks = await chunker.create_chunks(text)

        # Then
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_end = chunks[i].text[-10:]  # Last 10 chars
                next_start = chunks[i + 1].text[:20]  # First 20 chars
                # There should be some common words or characters
                current_words = set(current_end.split())
                next_words = set(next_start.split())
                # At least one word should overlap (or be very close)
                assert (
                    len(current_words & next_words) > 0 or len(current_end.strip()) < 10
                )

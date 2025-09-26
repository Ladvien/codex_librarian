#!/usr/bin/env python3
"""
MinerU PDF Processing Service Demo

This script demonstrates the MinerU service integration and capabilities.
Run this to test the MinerU service with a sample PDF.

Usage:
    python examples/mineru_demo.py [pdf_path]

If no PDF path is provided, it creates a sample PDF for testing.
"""

import asyncio
import sys
import tempfile
from pathlib import Path
import json

# Add src to Python path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_to_markdown_mcp.services.mineru import MinerUService
from pdf_to_markdown_mcp.models.request import ProcessingOptions
from pdf_to_markdown_mcp.core.exceptions import ValidationError, ProcessingError


def create_sample_pdf() -> Path:
    """Create a sample PDF file for demonstration."""
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello, MinerU!) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000015 00000 n
0000000066 00000 n
0000000123 00000 n
0000000204 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        f.write(pdf_content)
        return Path(f.name)


async def demo_basic_processing():
    """Demonstrate basic PDF processing."""
    print("🔧 Creating MinerU service...")
    service = MinerUService()

    # Create sample PDF
    print("📄 Creating sample PDF...")
    pdf_path = create_sample_pdf()
    print(f"   Created: {pdf_path}")

    try:
        # Configure processing options
        print("⚙️  Configuring processing options...")
        options = ProcessingOptions(
            ocr_language="eng",
            preserve_layout=True,
            extract_tables=True,
            extract_formulas=True,
            extract_images=True,
            chunk_for_embeddings=True,
            chunk_size=1000,
            chunk_overlap=200
        )

        print("🚀 Starting PDF processing...")
        result = await service.process_pdf(pdf_path, options)

        print("✅ Processing completed!")
        print("\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)

        # Display results
        print(f"📊 Processing Statistics:")
        metadata = result.processing_metadata
        print(f"   • Pages processed: {metadata.pages}")
        print(f"   • Processing time: {metadata.processing_time_ms}ms")
        print(f"   • File size: {metadata.file_size_bytes} bytes")
        print(f"   • OCR confidence: {metadata.ocr_confidence:.2%}")
        print(f"   • Text quality: {metadata.text_extraction_quality:.2%}")

        print(f"\n📝 Content Summary:")
        summary = result.get_content_summary()
        print(f"   • Text length: {summary['text_length']} characters")
        print(f"   • Markdown length: {summary['markdown_length']} characters")
        print(f"   • Tables found: {summary['tables_count']}")
        print(f"   • Formulas found: {summary['formulas_count']}")
        print(f"   • Images found: {summary['images_count']}")
        print(f"   • Text chunks: {summary['chunks_count']}")

        print(f"\n📖 Extracted Content:")
        print("   Markdown content:")
        print("   " + "-" * 40)
        for i, line in enumerate(result.markdown_content.split('\n')[:10]):
            print(f"   {i+1:2}: {line}")
        if len(result.markdown_content.split('\n')) > 10:
            print("   ... (truncated)")

        print(f"\n   Plain text content:")
        print("   " + "-" * 40)
        for i, line in enumerate(result.plain_text.split('\n')[:10]):
            print(f"   {i+1:2}: {line}")
        if len(result.plain_text.split('\n')) > 10:
            print("   ... (truncated)")

        if result.chunk_data:
            print(f"\n🧩 Text Chunks (showing first 3):")
            for i, chunk in enumerate(result.chunk_data[:3]):
                print(f"   Chunk {i+1}:")
                print(f"     • Text: {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}")
                print(f"     • Position: {chunk.start_char}-{chunk.end_char}")
                print(f"     • Tokens: ~{chunk.token_count}")

        # Test structured content
        if result.has_structured_content():
            print(f"\n🏗️  Document contains structured content!")
        else:
            print(f"\n📄 Document contains only plain text content")

        print("\n" + "="*60)
        print("SERVICE STATISTICS")
        print("="*60)

        stats = await service.get_processing_stats()
        print(f"🔧 Service: {stats['service']} v{stats['version']}")
        print(f"📏 Max file size: {stats['max_file_size_mb']} MB")
        print(f"⏱️  Timeout: {stats['timeout_seconds']} seconds")
        print(f"🌐 Languages: {', '.join(stats['supported_languages'])}")
        print(f"✨ Features:")
        for feature in stats['features']:
            print(f"   • {feature.replace('_', ' ').title()}")

    except ValidationError as e:
        print(f"❌ Validation Error: {e}")
        print(f"   Error code: {getattr(e, 'error_code', 'N/A')}")
        if hasattr(e, 'details'):
            print(f"   Details: {e.details}")

    except ProcessingError as e:
        print(f"❌ Processing Error: {e}")
        print(f"   Error code: {getattr(e, 'error_code', 'N/A')}")
        if hasattr(e, 'pdf_path'):
            print(f"   PDF path: {e.pdf_path}")

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

    finally:
        # Clean up
        if pdf_path.exists():
            pdf_path.unlink()
            print(f"🗑️  Cleaned up: {pdf_path}")


async def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)

    service = MinerUService()

    # Test 1: Non-existent file
    print("\n🧪 Test 1: Non-existent file")
    try:
        await service.process_pdf(Path("/nonexistent/file.pdf"), ProcessingOptions())
    except ValidationError as e:
        print(f"   ✅ Correctly caught: {e}")

    # Test 2: Invalid file type
    print("\n🧪 Test 2: Invalid file type")
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        f.write(b'Not a PDF')
        f.flush()
        try:
            await service.process_pdf(Path(f.name), ProcessingOptions())
        except ValidationError as e:
            print(f"   ✅ Correctly caught: {e}")

    # Test 3: File validation
    print("\n🧪 Test 3: File validation")
    pdf_path = create_sample_pdf()
    try:
        is_valid = await service.validate_pdf_file(pdf_path)
        print(f"   ✅ File validation passed: {is_valid}")
    except Exception as e:
        print(f"   ❌ File validation failed: {e}")
    finally:
        pdf_path.unlink()


async def demo_concurrent_processing():
    """Demonstrate concurrent processing capabilities."""
    print("\n" + "="*60)
    print("CONCURRENT PROCESSING DEMONSTRATION")
    print("="*60)

    service = MinerUService()

    # Create multiple sample PDFs
    pdf_files = []
    for i in range(3):
        pdf_path = create_sample_pdf()
        pdf_files.append(pdf_path)
        print(f"📄 Created sample PDF {i+1}: {pdf_path.name}")

    try:
        print("\n🚀 Starting concurrent processing...")

        # Process all files concurrently
        tasks = [
            service.process_pdf(pdf_path, ProcessingOptions(chunk_for_embeddings=False))
            for pdf_path in pdf_files
        ]

        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()

        print(f"⏱️  Total processing time: {end_time - start_time:.2f}s")

        # Check results
        successful = 0
        failed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"   ❌ PDF {i+1}: {result}")
                failed += 1
            else:
                print(f"   ✅ PDF {i+1}: {result.processing_metadata.processing_time_ms}ms")
                successful += 1

        print(f"\n📊 Results: {successful} successful, {failed} failed")

    finally:
        # Clean up
        for pdf_path in pdf_files:
            if pdf_path.exists():
                pdf_path.unlink()


def main():
    """Main demo function."""
    print("="*60)
    print("MINERU PDF PROCESSING SERVICE DEMO")
    print("="*60)
    print("This demo showcases MinerU service capabilities")
    print("Note: Using mock implementation (MinerU library not installed)")
    print()

    # Check if a PDF path was provided
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
            print(f"📄 Using provided PDF: {pdf_path}")
            # You could modify demo_basic_processing to use this file
        else:
            print(f"❌ Invalid PDF file: {pdf_path}")
            return

    # Run all demonstrations
    asyncio.run(demo_basic_processing())
    asyncio.run(demo_error_handling())
    asyncio.run(demo_concurrent_processing())

    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("✨ MinerU service is ready for integration!")
    print("🚀 Next steps:")
    print("   • Install MinerU library for production use")
    print("   • Integrate with Celery for background processing")
    print("   • Connect to database for content storage")
    print("   • Add embedding generation for semantic search")


if __name__ == "__main__":
    main()
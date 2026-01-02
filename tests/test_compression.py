"""
Test file compression functionality in DatabaseStorage.
"""
import pytest
import zlib
from app.core.file_storage import compress_data, decompress_data


class TestCompression:
    """Test compression and decompression functions."""
    
    def test_compress_small_data_skipped(self):
        """Small data should not be compressed (below min size)."""
        small_data = b"Hello World"  # Only 11 bytes
        result, is_compressed = compress_data(small_data)
        assert is_compressed is False
        assert result == small_data
    
    def test_compress_large_data(self):
        """Large data should be compressed."""
        # Create compressible data (repeated pattern)
        large_data = b"This is a test string that will be repeated. " * 100
        result, is_compressed = compress_data(large_data)
        assert is_compressed is True
        assert len(result) < len(large_data)
        # Verify we can decompress it
        decompressed = zlib.decompress(result)
        assert decompressed == large_data
    
    def test_compress_incompressible_data(self):
        """Random/incompressible data should not be compressed."""
        import os
        # Random data doesn't compress well
        random_data = os.urandom(2000)
        result, is_compressed = compress_data(random_data)
        # Might or might not be compressed depending on random data
        if is_compressed:
            assert len(result) < len(random_data)
        else:
            assert result == random_data
    
    def test_decompress_compressed_data(self):
        """Test decompression of compressed data."""
        original = b"Test data for decompression " * 100
        compressed = zlib.compress(original)
        result = decompress_data(compressed, is_compressed=True)
        assert result == original
    
    def test_decompress_uncompressed_data(self):
        """Test decompression passthrough for uncompressed data."""
        original = b"Uncompressed test data"
        result = decompress_data(original, is_compressed=False)
        assert result == original
    
    def test_roundtrip_compression(self):
        """Test full roundtrip: compress then decompress."""
        original = b"A" * 5000 + b"B" * 5000 + b"C" * 5000
        compressed, is_compressed = compress_data(original)
        assert is_compressed is True
        
        decompressed = decompress_data(compressed, is_compressed)
        assert decompressed == original
        
        # Verify compression ratio
        ratio = len(compressed) / len(original)
        print(f"Compression ratio: {ratio:.2%}")
        assert ratio < 0.5  # Should compress well


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


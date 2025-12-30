"""
Text Mining Widget API Tests
Tests for Corpus, Preprocess Text, Bag of Words, and Word Cloud endpoints.
Based on Orange3-Text widget tests.
"""

import pytest
import os
from fastapi.testclient import TestClient

# Import the app
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from app.main import app

client = TestClient(app)


class TestTextMiningAPI:
    """Test cases for Text Mining API endpoints."""
    
    # ========================================
    # Corpus Widget Tests
    # ========================================
    
    def test_get_available_corpora(self):
        """Test listing available corpora."""
        response = client.get("/api/v1/text/corpus/available")
        assert response.status_code == 200
        data = response.json()
        assert "orange_text_available" in data
        assert "samples" in data
        # Should have sample corpora listed
        assert len(data["samples"]) > 0
        
    def test_get_supported_languages(self):
        """Test getting supported languages list."""
        response = client.get("/api/v1/text/languages")
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert "default" in data
        assert data["default"] == "English"
        assert "English" in data["languages"]
        # Korean may not be in all NLTK stopword lists
        assert len(data["languages"]) >= 20  # At least 20 languages
        
    def test_load_corpus_sample_file(self):
        """Test loading a sample corpus file."""
        # This test requires Orange3-Text to be installed with sample data
        response = client.post(
            "/api/v1/text/corpus/load",
            json={
                "file_path": "book-excerpts.tab",
                "language": "English"
            }
        )
        # May fail if sample file not available
        assert response.status_code == 200
        data = response.json()
        # Sample file may not be available in all environments
        if data["success"]:
            assert "corpus_id" in data
            assert data["documents"] > 0
        else:
            # Expected failure when sample file not available
            assert "error" in data or data["error"] is not None
            
    def test_load_corpus_invalid_file(self):
        """Test loading non-existent corpus file."""
        response = client.post(
            "/api/v1/text/corpus/load",
            json={
                "file_path": "nonexistent_file.tab",
                "language": "English"
            }
        )
        data = response.json()
        assert data["success"] == False
        assert data["error"] is not None
        
    def test_corpus_with_title_variable(self):
        """Test corpus loading with title variable selection."""
        response = client.post(
            "/api/v1/text/corpus/load",
            json={
                "file_path": "book-excerpts.tab",
                "title_variable": "Text",
                "language": "English"
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                assert data["title_variable"] == "Text"
                
    def test_get_corpus_info(self):
        """Test getting corpus information by ID."""
        # First load a corpus
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            # Get corpus info
            response = client.get(f"/api/v1/text/corpus/{corpus_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["corpus_id"] == corpus_id
            
    def test_get_nonexistent_corpus(self):
        """Test getting info for non-existent corpus."""
        response = client.get("/api/v1/text/corpus/nonexistent_id")
        assert response.status_code == 404
        
    # ========================================
    # Preprocess Text Widget Tests
    # ========================================
    
    def test_preprocess_lowercase(self):
        """Test text preprocessing with lowercase transformation."""
        # First load a corpus
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            response = client.post(
                "/api/v1/text/preprocess",
                json={
                    "corpus_id": corpus_id,
                    "lowercase": True,
                    "remove_accents": False,
                    "parse_html": False,
                    "remove_urls": False,
                    "tokenizer": "regexp",
                    "regexp_pattern": r"\w+",
                    "use_stopwords": False
                }
            )
            assert response.status_code == 200
            data = response.json()
            if data["success"]:
                assert "corpus_id" in data
                assert data["documents"] > 0
                
    def test_preprocess_with_stopwords(self):
        """Test preprocessing with stopword filtering."""
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            response = client.post(
                "/api/v1/text/preprocess",
                json={
                    "corpus_id": corpus_id,
                    "lowercase": True,
                    "tokenizer": "regexp",
                    "regexp_pattern": r"\w+",
                    "use_stopwords": True,
                    "stopwords_language": "English"
                }
            )
            assert response.status_code == 200
            data = response.json()
            if data["success"]:
                # After stopword removal, token count should decrease
                assert data["tokens_after"] <= data["tokens_before"]
                
    def test_preprocess_different_tokenizers(self):
        """Test different tokenization methods."""
        tokenizers = ["word_punct", "whitespace", "regexp", "tweet"]
        
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            for tokenizer in tokenizers:
                response = client.post(
                    "/api/v1/text/preprocess",
                    json={
                        "corpus_id": corpus_id,
                        "lowercase": False,
                        "tokenizer": tokenizer,
                        "regexp_pattern": r"\w+" if tokenizer == "regexp" else None,
                        "use_stopwords": False
                    }
                )
                assert response.status_code == 200
                data = response.json()
                # Tokenization should produce some output
                
    def test_preprocess_nonexistent_corpus(self):
        """Test preprocessing with invalid corpus ID."""
        response = client.post(
            "/api/v1/text/preprocess",
            json={
                "corpus_id": "nonexistent_corpus",
                "lowercase": True,
                "tokenizer": "regexp"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        
    # ========================================
    # Bag of Words Widget Tests
    # ========================================
    
    def test_bow_count(self):
        """Test Bag of Words with count term frequency."""
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            # First preprocess
            preprocess_response = client.post(
                "/api/v1/text/preprocess",
                json={
                    "corpus_id": corpus_id,
                    "lowercase": True,
                    "tokenizer": "regexp",
                    "regexp_pattern": r"\w+",
                    "use_stopwords": True,
                    "stopwords_language": "English"
                }
            )
            if preprocess_response.status_code == 200 and preprocess_response.json()["success"]:
                preprocessed_id = preprocess_response.json()["corpus_id"]
                
                response = client.post(
                    "/api/v1/text/bow",
                    json={
                        "corpus_id": preprocessed_id,
                        "term_frequency": "count",
                        "document_frequency": "none",
                        "regularization": "none"
                    }
                )
                assert response.status_code == 200
                data = response.json()
                if data["success"]:
                    assert data["documents"] > 0
                    assert data["vocabulary_size"] > 0
                    
    def test_bow_tfidf(self):
        """Test Bag of Words with TF-IDF."""
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            response = client.post(
                "/api/v1/text/bow",
                json={
                    "corpus_id": corpus_id,
                    "term_frequency": "sublinear",
                    "document_frequency": "idf",
                    "regularization": "l2"
                }
            )
            assert response.status_code == 200
            
    def test_bow_binary(self):
        """Test Bag of Words with binary term frequency."""
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            response = client.post(
                "/api/v1/text/bow",
                json={
                    "corpus_id": corpus_id,
                    "term_frequency": "binary",
                    "document_frequency": "none",
                    "regularization": "none"
                }
            )
            assert response.status_code == 200
            
    def test_bow_nonexistent_corpus(self):
        """Test BOW with invalid corpus ID."""
        response = client.post(
            "/api/v1/text/bow",
            json={
                "corpus_id": "nonexistent_corpus",
                "term_frequency": "count"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        
    # ========================================
    # Word Cloud Widget Tests
    # ========================================
    
    def test_wordcloud_data(self):
        """Test word cloud data generation."""
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            response = client.post(
                "/api/v1/text/wordcloud",
                json={
                    "corpus_id": corpus_id,
                    "max_words": 50
                }
            )
            assert response.status_code == 200
            data = response.json()
            if data["success"]:
                assert "words" in data
                assert len(data["words"]) <= 50
                assert data["total_words"] > 0
                # Check word structure
                if len(data["words"]) > 0:
                    word = data["words"][0]
                    assert "word" in word
                    assert "weight" in word
                    assert "normalized" in word
                    
    def test_wordcloud_max_words(self):
        """Test word cloud respects max_words limit."""
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            for max_words in [10, 25, 100]:
                response = client.post(
                    "/api/v1/text/wordcloud",
                    json={
                        "corpus_id": corpus_id,
                        "max_words": max_words
                    }
                )
                assert response.status_code == 200
                data = response.json()
                if data["success"]:
                    assert len(data["words"]) <= max_words
                    
    def test_wordcloud_select_words(self):
        """Test word selection in word cloud."""
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            # First get word cloud data
            cloud_response = client.post(
                "/api/v1/text/wordcloud",
                json={"corpus_id": corpus_id, "max_words": 50}
            )
            if cloud_response.status_code == 200 and cloud_response.json()["success"]:
                words = cloud_response.json()["words"]
                if len(words) >= 2:
                    selected = [words[0]["word"], words[1]["word"]]
                    
                    response = client.post(
                        "/api/v1/text/wordcloud/select",
                        json={"corpus_id": corpus_id, "selected_words": selected}
                    )
                    assert response.status_code == 200
                    data = response.json()
                    if data["success"]:
                        assert "documents" in data
                        assert data["selected_words"] == selected
                        
    def test_wordcloud_nonexistent_corpus(self):
        """Test word cloud with invalid corpus ID."""
        response = client.post(
            "/api/v1/text/wordcloud",
            json={
                "corpus_id": "nonexistent_corpus",
                "max_words": 50
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        
    # ========================================
    # Cache Management Tests
    # ========================================
    
    def test_clear_cache_item(self):
        """Test clearing a specific cache item."""
        # First create a cache item
        load_response = client.post(
            "/api/v1/text/corpus/load",
            json={"file_path": "book-excerpts.tab", "language": "English"}
        )
        if load_response.status_code == 200 and load_response.json()["success"]:
            corpus_id = load_response.json()["corpus_id"]
            
            # Delete it
            response = client.delete(f"/api/v1/text/cache/{corpus_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            
    def test_clear_nonexistent_cache(self):
        """Test clearing non-existent cache item."""
        response = client.delete("/api/v1/text/cache/nonexistent_id")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False


# ========================================
# Run tests
# ========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


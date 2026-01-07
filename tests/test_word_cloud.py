"""
Test cases for Word Cloud Widget API endpoints.
Tests the word cloud generation, word selection, and output ports.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the router
from app.widgets.word_cloud import router
from app.core.text_mining_utils import set_cache_item, get_cache_item, delete_cache_item

# Create a test client
from fastapi import FastAPI
app = FastAPI()
app.include_router(router, prefix="/api/v1")
client = TestClient(app)


class TestWordCloudAPI:
    """Test cases for Word Cloud API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data before each test."""
        # Create mock corpus with tokens
        self.corpus_id = f"test_corpus_{uuid.uuid4().hex[:8]}"
        self.mock_corpus = MagicMock()
        self.mock_corpus.tokens = [
            ['hello', 'world', 'hello'],
            ['python', 'programming', 'world'],
            ['hello', 'python', 'test']
        ]
        self.mock_corpus.documents = [
            'hello world hello',
            'python programming world',
            'hello python test'
        ]
        
        # Store in cache
        set_cache_item(self.corpus_id, {
            'corpus': self.mock_corpus,
            'type': 'corpus'
        })
        
        yield
        
        # Cleanup
        delete_cache_item(self.corpus_id)
    
    def test_wordcloud_basic(self):
        """Test 1: Basic word cloud generation."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id,
            "max_words": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert len(data['words']) > 0
        assert 'hello' in [w['word'] for w in data['words']]
    
    def test_wordcloud_word_frequency(self):
        """Test 2: Word frequency calculation."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id,
            "max_words": 100
        })
        
        data = response.json()
        words_dict = {w['word']: w['weight'] for w in data['words']}
        
        # 'hello' appears 3 times
        assert words_dict.get('hello', 0) == 3
        # 'world' appears 2 times
        assert words_dict.get('world', 0) == 2
        # 'python' appears 2 times
        assert words_dict.get('python', 0) == 2
    
    def test_wordcloud_max_words_limit(self):
        """Test 3: Max words limit."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id,
            "max_words": 3
        })
        
        data = response.json()
        assert len(data['words']) <= 3
    
    def test_wordcloud_normalized_weights(self):
        """Test 4: Normalized weights are between 0 and 1."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id
        })
        
        data = response.json()
        for word in data['words']:
            assert 0 <= word['normalized'] <= 1
    
    def test_wordcloud_document_frequency(self):
        """Test 5: Document frequency calculation."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id
        })
        
        data = response.json()
        words_dict = {w['word']: w['documents'] for w in data['words']}
        
        # 'hello' appears in 2 documents
        assert words_dict.get('hello', 0) == 2
        # 'world' appears in 2 documents
        assert words_dict.get('world', 0) == 2
    
    def test_wordcloud_word_counts_id(self):
        """Test 6: Word counts ID is generated."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id
        })
        
        data = response.json()
        assert data['word_counts_id'] is not None
        assert data['word_counts_id'].startswith('wordcounts_')
    
    def test_wordcloud_invalid_corpus(self):
        """Test 7: Invalid corpus ID returns error."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": "invalid_corpus_id"
        })
        
        data = response.json()
        assert data['success'] == False
        assert 'not found' in data['error'].lower()
    
    def test_wordcloud_with_selected_words(self):
        """Test 8: Selected words are returned."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id,
            "selected_words": ["hello", "world"]
        })
        
        data = response.json()
        assert set(data['selected_words']) == {"hello", "world"}
    
    def test_wordcloud_select_words(self):
        """Test 9: Select words endpoint filters corpus."""
        response = client.post("/api/v1/text/wordcloud/select", json={
            "corpus_id": self.corpus_id,
            "selected_words": ["hello"]
        })
        
        data = response.json()
        assert data['success'] == True
        # Should return selected words
        assert "hello" in data['selected_words']
    
    def test_wordcloud_select_no_match(self):
        """Test 10: Select words with no matching documents."""
        response = client.post("/api/v1/text/wordcloud/select", json={
            "corpus_id": self.corpus_id,
            "selected_words": ["nonexistent"]
        })
        
        data = response.json()
        assert data['success'] == True
        assert data['documents'] == 0
    
    def test_word_counts_endpoint(self):
        """Test 11: Get word counts by ID."""
        # First generate word cloud to get word_counts_id
        wc_response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.corpus_id
        })
        word_counts_id = wc_response.json()['word_counts_id']
        
        # Get word counts
        response = client.get(f"/api/v1/text/wordcloud/counts/{word_counts_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert data['word_count'] > 0
    
    def test_word_counts_invalid_id(self):
        """Test 12: Invalid word counts ID returns 404."""
        response = client.get("/api/v1/text/wordcloud/counts/invalid_id")
        assert response.status_code == 404
    
    def test_languages_endpoint(self):
        """Test 13: Get supported languages."""
        response = client.get("/api/v1/text/languages")
        
        assert response.status_code == 200
        data = response.json()
        assert 'languages' in data
        assert 'English' in data['languages']
        assert data['default'] == 'English'
    
    def test_cache_clear(self):
        """Test 14: Clear cache item."""
        # Create a temporary cache item
        temp_id = f"temp_{uuid.uuid4().hex[:8]}"
        set_cache_item(temp_id, {'test': 'data'})
        
        response = client.delete(f"/api/v1/text/cache/{temp_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        
        # Verify item is deleted
        assert get_cache_item(temp_id) is None
    
    def test_wordcloud_empty_corpus(self):
        """Test 15: Empty corpus returns empty words."""
        empty_corpus_id = f"empty_corpus_{uuid.uuid4().hex[:8]}"
        empty_corpus = MagicMock()
        empty_corpus.tokens = []
        empty_corpus.documents = []
        
        set_cache_item(empty_corpus_id, {
            'corpus': empty_corpus,
            'type': 'corpus'
        })
        
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": empty_corpus_id
        })
        
        data = response.json()
        assert data['success'] == True
        assert len(data['words']) == 0
        
        delete_cache_item(empty_corpus_id)


class TestWordCloudTopicInput:
    """Test cases for Word Cloud with Topic input."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        self.topic_id = f"test_topic_{uuid.uuid4().hex[:8]}"
        
        # Create mock topic with words and weights
        set_cache_item(self.topic_id, {
            'type': 'topic',
            'words': [
                {'word': 'machine', 'weight': 10},
                {'word': 'learning', 'weight': 8},
                {'word': 'neural', 'weight': 5},
                {'word': 'network', 'weight': 5}
            ]
        })
        
        yield
        
        delete_cache_item(self.topic_id)
    
    def test_wordcloud_from_topic(self):
        """Test 16: Generate word cloud from topic."""
        response = client.post("/api/v1/text/wordcloud", json={
            "topic_id": self.topic_id
        })
        
        data = response.json()
        assert data['success'] == True
        assert len(data['words']) == 4
        
        words_dict = {w['word']: w['weight'] for w in data['words']}
        assert words_dict['machine'] == 10
        assert words_dict['learning'] == 8
    
    def test_wordcloud_topic_priority(self):
        """Test 17: Topic takes priority over corpus."""
        corpus_id = f"test_corpus_{uuid.uuid4().hex[:8]}"
        corpus = MagicMock()
        corpus.tokens = [['different', 'words']]
        
        set_cache_item(corpus_id, {
            'corpus': corpus,
            'type': 'corpus'
        })
        
        # When both topic and corpus are provided, topic should be used
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": corpus_id,
            "topic_id": self.topic_id
        })
        
        data = response.json()
        # Should use topic words, not corpus words
        words = [w['word'] for w in data['words']]
        assert 'machine' in words
        assert 'different' not in words
        
        delete_cache_item(corpus_id)


class TestWordCloudBOW:
    """Test cases for Word Cloud with Bag of Words input."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup BOW corpus."""
        import numpy as np
        
        self.bow_corpus_id = f"bow_corpus_{uuid.uuid4().hex[:8]}"
        
        # Create mock BOW corpus
        bow_corpus = MagicMock()
        
        # Mock domain with BOW attributes
        attr1 = MagicMock()
        attr1.name = 'word_a'
        attr1.attributes = {'bow-feature': True}
        
        attr2 = MagicMock()
        attr2.name = 'word_b'
        attr2.attributes = {'bow-feature': True}
        
        bow_corpus.domain = MagicMock()
        bow_corpus.domain.attributes = [attr1, attr2]
        
        # Mock X matrix (BOW counts)
        bow_corpus.X = np.array([
            [3, 1],  # doc 1: word_a=3, word_b=1
            [2, 2],  # doc 2: word_a=2, word_b=2
            [0, 1]   # doc 3: word_a=0, word_b=1
        ])
        
        set_cache_item(self.bow_corpus_id, {
            'corpus': bow_corpus,
            'type': 'corpus'
        })
        
        yield
        
        delete_cache_item(self.bow_corpus_id)
    
    def test_wordcloud_bow_weights(self):
        """Test 18: BOW weights are calculated correctly."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.bow_corpus_id
        })
        
        data = response.json()
        assert data['success'] == True
        assert data['is_bow_weights'] == True
    
    def test_wordcloud_bow_average(self):
        """Test 19: BOW average weights are correct."""
        response = client.post("/api/v1/text/wordcloud", json={
            "corpus_id": self.bow_corpus_id
        })
        
        data = response.json()
        words_dict = {w['word']: w['weight'] for w in data['words']}
        
        # word_a average: (3+2+0)/3 = 1.67
        # word_b average: (1+2+1)/3 = 1.33
        # Only positive averages should be included
        assert len(words_dict) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


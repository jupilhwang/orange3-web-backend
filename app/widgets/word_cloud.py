"""
Word Cloud Widget API endpoints.
Generate word cloud visualization from text corpus.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from app.core.text_mining_utils import (
    ORANGE_TEXT_AVAILABLE, get_cache_item, set_cache_item, delete_cache_item
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text", tags=["Text Mining - Word Cloud"])


# ============================================================================
# Models
# ============================================================================

class WordCloudRequest(BaseModel):
    """Request for word cloud data."""
    corpus_id: Optional[str] = None
    topic_id: Optional[str] = None  # Topic model input
    max_words: int = 100
    selected_words: Optional[List[str]] = None
    word_tilt: float = 0.0  # Tilt angle 0-90
    regenerate: bool = False


class WordCloudResponse(BaseModel):
    """Response with word cloud data."""
    success: bool
    words: List[Dict[str, Any]] = []
    total_words: int = 0
    word_counts_id: Optional[str] = None
    selected_words: List[str] = []
    is_bow_weights: bool = False  # True when showing BOW weights
    error: Optional[str] = None


class WordCountsResponse(BaseModel):
    """Response for Word Counts output."""
    success: bool
    data_id: Optional[str] = None
    word_count: int = 0
    total_frequency: int = 0
    words: List[Dict[str, Any]] = []
    error: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/wordcloud")
async def get_word_cloud_data(
    request: WordCloudRequest,
    x_session_id: Optional[str] = Header(None)
) -> WordCloudResponse:
    """Get word frequency data for word cloud."""
    if not ORANGE_TEXT_AVAILABLE:
        return WordCloudResponse(
            success=False,
            error="Orange3-Text is not installed"
        )
    
    # Support both corpus and topic input
    corpus = None
    topic_words = None
    
    # Option 1: Topic input
    if request.topic_id:
        cached = get_cache_item(request.topic_id)
        if cached and cached.get('type') == 'topic':
            topic_words = cached.get('words', [])
    
    # Option 2: Corpus input
    if request.corpus_id:
        cached = get_cache_item(request.corpus_id)
        if cached:
            corpus = cached.get('corpus') or cached.get('data')
    
    if corpus is None and topic_words is None:
        return WordCloudResponse(
            success=False,
            error="Corpus or Topic not found"
        )
    
    try:
        word_freq: Dict[str, int] = {}
        doc_freq: Dict[str, int] = {}  # Document frequency for Word Counts output
        is_bow_weights = False  # Track if using BOW weights
        
        # Use topic words if available
        if topic_words:
            for item in topic_words:
                word_freq[item['word']] = item.get('weight', 1)
        
        # Use corpus otherwise
        elif corpus is not None:
            # First check for BOW features (like Orange3's _bow_words)
            if hasattr(corpus, 'domain') and corpus.domain.attributes:
                bow_words = {}
                for i, attr in enumerate(corpus.domain.attributes):
                    if hasattr(attr, 'attributes') and attr.attributes.get('bow-feature', False):
                        # This is a BOW feature
                        col_values = corpus.X[:, i]
                        avg_bow = col_values.mean()
                        if avg_bow > 0:
                            bow_words[attr.name] = avg_bow
                
                if bow_words:
                    # Use BOW weights
                    word_freq = {k: v for k, v in bow_words.items()}
                    is_bow_weights = True
                    # Calculate document frequency
                    for i, attr in enumerate(corpus.domain.attributes):
                        if attr.name in word_freq:
                            col_values = corpus.X[:, i]
                            doc_freq[attr.name] = sum(1 for v in col_values if v > 0)
            
            # If no BOW, try tokens or documents
            if not word_freq:
                if hasattr(corpus, 'tokens'):
                    # Preprocessed corpus with tokens
                    for doc_tokens in corpus.tokens:
                        doc_words = set()
                        for token in doc_tokens:
                            word_freq[token] = word_freq.get(token, 0) + 1
                            doc_words.add(token)
                        for word in doc_words:
                            doc_freq[word] = doc_freq.get(word, 0) + 1
                elif hasattr(corpus, 'ngrams'):
                    # Use ngrams if available (Orange3 Corpus)
                    for doc in corpus.ngrams:
                        doc_words = set()
                        for word in doc:
                            word_freq[word] = word_freq.get(word, 0) + 1
                            doc_words.add(word)
                        for word in doc_words:
                            doc_freq[word] = doc_freq.get(word, 0) + 1
                elif hasattr(corpus, 'documents'):
                    # Raw corpus
                    for doc in corpus.documents:
                        doc_words = set()
                        for word in doc.lower().split():
                            word_freq[word] = word_freq.get(word, 0) + 1
                            doc_words.add(word)
                        for word in doc_words:
                            doc_freq[word] = doc_freq.get(word, 0) + 1
        
        # Sort by frequency and take top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:request.max_words]
        
        # Apply selection filter if provided
        selected_set = set(request.selected_words) if request.selected_words else None
        
        # Normalize weights for visualization
        max_weight = top_words[0][1] if top_words else 1
        words = [
            {
                'word': word,
                'weight': count,
                'normalized': count / max_weight,
                'selected': word in selected_set if selected_set else False,
                'documents': doc_freq.get(word, 0)
            }
            for word, count in top_words
        ]
        
        # Create Word Counts output data (Table format)
        word_counts_id = f"wordcounts_{uuid.uuid4().hex[:8]}"
        word_counts_data = [
            {'word': w['word'], 'count': w['weight'], 'documents': w['documents']}
            for w in words
        ]
        
        # Store for Word Counts output port
        set_cache_item(word_counts_id, {
            'type': 'word_counts',
            'data': word_counts_data,
            'total_words': len(word_freq),
            'source_id': request.corpus_id or request.topic_id
        })
        
        return WordCloudResponse(
            success=True,
            words=words,
            total_words=len(word_freq),
            word_counts_id=word_counts_id,
            selected_words=list(selected_set) if selected_set else [],
            is_bow_weights=is_bow_weights
        )
        
    except Exception as e:
        logger.error(f"Error generating word cloud data: {e}")
        import traceback
        traceback.print_exc()
        return WordCloudResponse(
            success=False,
            error=str(e)
        )


@router.get("/wordcloud/counts/{word_counts_id}")
async def get_word_counts(word_counts_id: str) -> WordCountsResponse:
    """Get Word Counts data for output port."""
    cached = get_cache_item(word_counts_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Word counts data not found")
    
    if cached.get('type') != 'word_counts':
        raise HTTPException(status_code=400, detail="Invalid word counts ID")
    
    data = cached.get('data', [])
    total_freq = sum(w.get('count', 0) for w in data)
    
    return WordCountsResponse(
        success=True,
        data_id=word_counts_id,
        word_count=len(data),
        total_frequency=total_freq,
        words=data
    )


class WordSelectRequest(BaseModel):
    """Request for selecting words in word cloud."""
    corpus_id: str
    selected_words: List[str]


@router.post("/wordcloud/select")
async def select_words(
    request: WordSelectRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Filter corpus by selected words."""
    cached = get_cache_item(request.corpus_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    try:
        corpus = cached.get('corpus') or cached.get('data')
        
        # Filter documents containing selected words
        selected_set = set(w.lower() for w in request.selected_words)
        selected_indices = []
        
        if hasattr(corpus, 'tokens'):
            for i, doc_tokens in enumerate(corpus.tokens):
                if any(token.lower() in selected_set for token in doc_tokens):
                    selected_indices.append(i)
        elif hasattr(corpus, 'documents'):
            for i, doc in enumerate(corpus.documents):
                if any(word in selected_set for word in doc.lower().split()):
                    selected_indices.append(i)
        
        # Create filtered corpus
        if selected_indices:
            filtered_corpus = corpus[selected_indices]
            new_corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
            set_cache_item(new_corpus_id, {
                'corpus': filtered_corpus,
                'source_id': request.corpus_id,
                'selected_words': request.selected_words
            })
            
            return {
                'success': True,
                'corpus_id': new_corpus_id,
                'documents': len(filtered_corpus),
                'selected_words': request.selected_words
            }
        else:
            return {
                'success': True,
                'corpus_id': None,
                'documents': 0,
                'selected_words': request.selected_words
            }
            
    except Exception as e:
        logger.error(f"Error selecting words: {e}")
        return {'success': False, 'error': str(e)}


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages for text processing."""
    languages = [
        "Arabic", "Azerbaijani", "Basque", "Bengali", "Catalan",
        "Chinese", "Danish", "Dutch", "English", "Finnish",
        "French", "German", "Greek", "Hebrew", "Hindi",
        "Hungarian", "Indonesian", "Irish", "Italian", "Japanese",
        "Kazakh", "Nepali", "Norwegian", "Portuguese", "Romanian",
        "Russian", "Slovene", "Spanish", "Swedish", "Tajik",
        "Turkish"
    ]
    return {'languages': languages, 'default': 'English'}


@router.delete("/cache/{item_id}")
async def clear_cache_item(item_id: str):
    """Remove an item from the text cache."""
    if delete_cache_item(item_id):
        return {'success': True, 'message': f'Removed {item_id} from cache'}
    return {'success': False, 'message': 'Item not found'}


"""
Bag of Words Widget API endpoints.
Create Bag of Words representation from text corpus.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel

from app.core.text_mining_utils import (
    ORANGE_TEXT_AVAILABLE, get_cache_item, set_cache_item
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text", tags=["Text Mining - Bag of Words"])


# ============================================================================
# Models
# ============================================================================

class BagOfWordsRequest(BaseModel):
    """Request to create Bag of Words."""
    corpus_id: str
    term_frequency: str = "count"  # count, binary, sublinear
    document_frequency: str = "none"  # none, idf, smooth_idf
    regularization: str = "none"  # none, l1, l2
    hide_bow_attributes: bool = True


class BagOfWordsResponse(BaseModel):
    """Response with Bag of Words data."""
    success: bool
    data_id: Optional[str] = None
    corpus_id: Optional[str] = None
    documents: int = 0
    vocabulary_size: int = 0
    error: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/bow")
async def create_bag_of_words(
    request: BagOfWordsRequest,
    x_session_id: Optional[str] = Header(None)
) -> BagOfWordsResponse:
    """Create Bag of Words representation."""
    if not ORANGE_TEXT_AVAILABLE:
        return BagOfWordsResponse(
            success=False,
            error="Orange3-Text is not installed"
        )
    
    cached = get_cache_item(request.corpus_id)
    if not cached:
        return BagOfWordsResponse(
            success=False,
            error="Corpus not found"
        )
    
    try:
        from orangecontrib.text.vectorization import BowVectorizer
        
        corpus = cached['corpus']
        
        # Map parameters to BowVectorizer
        wlocal = BowVectorizer.COUNT
        if request.term_frequency == 'binary':
            wlocal = BowVectorizer.BINARY
        elif request.term_frequency == 'sublinear':
            wlocal = BowVectorizer.SUBLINEAR
        
        wglobal = BowVectorizer.NONE
        if request.document_frequency == 'idf':
            wglobal = BowVectorizer.IDF
        elif request.document_frequency == 'smooth_idf':
            wglobal = BowVectorizer.SMOOTH_IDF
        
        norm = BowVectorizer.NONE
        if request.regularization == 'l1':
            norm = BowVectorizer.L1
        elif request.regularization == 'l2':
            norm = BowVectorizer.L2
        
        # Create vectorizer and transform
        vectorizer = BowVectorizer(wlocal=wlocal, wglobal=wglobal, norm=norm)
        bow_corpus = vectorizer.transform(corpus)
        
        # Get vocabulary size
        vocab_size = len(bow_corpus.domain.attributes) if bow_corpus.domain.attributes else 0
        
        # Store result
        data_id = f"bow_{uuid.uuid4().hex[:8]}"
        set_cache_item(data_id, {
            'data': bow_corpus,
            'type': 'bow',
            'source_corpus_id': request.corpus_id,
            'vocabulary_size': vocab_size
        })
        
        # Also keep the corpus for downstream widgets
        corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
        set_cache_item(corpus_id, {
            'corpus': bow_corpus,
            'type': 'bow_corpus',
            'source_id': request.corpus_id
        })
        
        return BagOfWordsResponse(
            success=True,
            data_id=data_id,
            corpus_id=corpus_id,
            documents=len(bow_corpus),
            vocabulary_size=vocab_size
        )
        
    except Exception as e:
        logger.error(f"Error creating Bag of Words: {e}")
        import traceback
        traceback.print_exc()
        return BagOfWordsResponse(
            success=False,
            error=str(e)
        )


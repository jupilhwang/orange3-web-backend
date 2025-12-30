"""
Text Mining Widget API endpoints.
Implements Corpus, Preprocess Text, Bag of Words, and Word Cloud.
"""

import logging
import uuid
import os
from typing import List, Optional, Dict, Any, Set

from fastapi import APIRouter, HTTPException, Header, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text", tags=["Text Mining"])

# Check Orange3-Text availability
try:
    from orangecontrib.text import Corpus
    from orangecontrib.text.preprocess import (
        Preprocessor, LowercaseTransformer, StripAccentsTransformer,
        HtmlTransformer, UrlRemover, RegexpTokenizer, WhitespaceTokenizer,
        WordPunctTokenizer, TweetTokenizer, SentenceTokenizer,
        StopwordsFilter, FrequencyFilter, LexiconFilter, RegexpFilter,
        Stemmer, Normalizer
    )
    from orangecontrib.text.preprocess.normalize import BaseNormalizer
    from orangecontrib.text.vectorization import BowVectorizer
    from Orange.data import Table, Domain, StringVariable, ContinuousVariable
    ORANGE_TEXT_AVAILABLE = True
    logger.info("Orange3-Text is available")
    
    # Try to import optional components
    try:
        from orangecontrib.text.preprocess import NGrams
        HAS_NGRAMS = True
    except ImportError:
        HAS_NGRAMS = False
        
    try:
        from orangecontrib.text.tag import AveragedPerceptronTagger, POSTagger
        HAS_POS_TAGGER = True
    except ImportError:
        HAS_POS_TAGGER = False
        
    try:
        from orangecontrib.text.preprocess.normalize import UDPipeLemmatizer
        HAS_UDPIPE = True
    except ImportError:
        HAS_UDPIPE = False
        
    try:
        from orangecontrib.text.topics import Topic
        HAS_TOPICS = True
    except ImportError:
        HAS_TOPICS = False
        
except ImportError as e:
    ORANGE_TEXT_AVAILABLE = False
    HAS_NGRAMS = False
    HAS_POS_TAGGER = False
    HAS_UDPIPE = False
    HAS_TOPICS = False
    logger.warning(f"Orange3-Text not available: {e}")

# In-memory cache for text data
_text_cache: Dict[str, Any] = {}


# ============================================================================
# Models
# ============================================================================

class CorpusLoadRequest(BaseModel):
    """Request to load a corpus file."""
    file_path: Optional[str] = None
    data_path: Optional[str] = None  # For Data input port (from Table)
    title_variable: Optional[str] = None
    language: str = "English"
    used_text_features: Optional[List[str]] = None  # Selected text features


class CorpusResponse(BaseModel):
    """Response with corpus information."""
    success: bool
    corpus_id: Optional[str] = None
    documents: int = 0
    text_features: List[str] = []
    meta_features: List[str] = []
    title_variable: Optional[str] = None
    language: str = "English"
    preview: Optional[List[Dict]] = None
    error: Optional[str] = None


class PreprocessRequest(BaseModel):
    """Request to preprocess text."""
    corpus_id: str
    # Transformation options
    lowercase: bool = True
    remove_accents: bool = False
    parse_html: bool = False
    remove_urls: bool = False
    # Tokenization options
    tokenizer: str = "regexp"  # word_punct, whitespace, sentence, regexp, tweet
    regexp_pattern: str = r"\w+"
    # Filtering options
    use_stopwords: bool = True
    stopwords_language: str = "English"
    custom_stopwords: List[str] = []
    min_df: Optional[float] = None  # Document frequency filter (relative 0-1)
    max_df: Optional[float] = None
    abs_min_df: Optional[int] = None  # Absolute document frequency filter
    abs_max_df: Optional[int] = None
    use_lexicon: bool = False
    lexicon_words: List[str] = []
    filter_numbers: bool = False
    include_numbers: bool = False  # Include numbers option
    filter_regexp: Optional[str] = None
    most_frequent_tokens: Optional[int] = None  # Keep only N most frequent
    # N-grams options
    use_ngrams: bool = False
    ngram_min: int = 1
    ngram_max: int = 2
    # POS Tagger options
    use_pos_tagger: bool = False
    pos_tags: List[str] = []  # e.g., ["NOUN", "VERB"]
    # Normalization
    normalization: str = "none"  # none, porter, snowball, wordnet, udpipe
    udpipe_language: Optional[str] = None  # For UDPipe lemmatizer


class PreprocessResponse(BaseModel):
    """Response with preprocessed corpus."""
    success: bool
    corpus_id: Optional[str] = None
    documents: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    ngram_info: Optional[Dict[str, Any]] = None  # N-gram statistics
    pos_info: Optional[Dict[str, Any]] = None  # POS tag statistics
    error: Optional[str] = None


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


class WordCloudRequest(BaseModel):
    """Request for word cloud data."""
    corpus_id: Optional[str] = None
    topic_id: Optional[str] = None  # Topic model input
    max_words: int = 100
    selected_words: Optional[List[str]] = None  # For filtering
    word_tilt: float = 0.0  # Tilt angle 0-90
    regenerate: bool = False  # Force regeneration


class WordCloudResponse(BaseModel):
    """Response with word cloud data."""
    success: bool
    words: List[Dict[str, Any]] = []  # [{word, weight, selected}, ...]
    total_words: int = 0
    word_counts_id: Optional[str] = None  # For Word Counts output port
    selected_words: List[str] = []  # Currently selected words
    error: Optional[str] = None


class WordCountsResponse(BaseModel):
    """Response for Word Counts output."""
    success: bool
    data_id: Optional[str] = None
    word_count: int = 0
    total_frequency: int = 0
    words: List[Dict[str, Any]] = []  # [{word, count, documents}, ...]
    error: Optional[str] = None


# ============================================================================
# Corpus Endpoints
# ============================================================================

@router.get("/corpus/available")
async def get_available_corpora():
    """Get list of available sample corpora."""
    corpora = []
    
    if ORANGE_TEXT_AVAILABLE:
        try:
            # Try to find sample corpora from Orange3-Text
            import orangecontrib.text.datasets as text_datasets
            datasets_path = os.path.dirname(text_datasets.__file__)
            
            for filename in sorted(os.listdir(datasets_path)):
                if filename.endswith('.tab') or filename.endswith('.csv'):
                    corpora.append({
                        'name': filename,
                        'path': os.path.join(datasets_path, filename),
                        'source': 'orange3-text'
                    })
        except Exception as e:
            logger.warning(f"Could not list text datasets: {e}")
    
    # Add common sample corpora
    sample_corpora = [
        {'name': 'book-excerpts.tab', 'description': 'Book excerpts with genres'},
        {'name': 'deerwester.tab', 'description': 'Deerwester toy corpus for LSI'},
        {'name': 'friends-transcripts.tab', 'description': 'Friends TV show transcripts'},
        {'name': 'andersen.tab', 'description': 'Andersen fairy tales'},
    ]
    
    return {
        'corpora': corpora,
        'samples': sample_corpora,
        'orange_text_available': ORANGE_TEXT_AVAILABLE
    }


@router.post("/corpus/load")
async def load_corpus(
    request: CorpusLoadRequest,
    x_session_id: Optional[str] = Header(None)
) -> CorpusResponse:
    """Load a corpus from file or from Data input."""
    if not ORANGE_TEXT_AVAILABLE:
        return CorpusResponse(
            success=False,
            error="Orange3-Text is not installed"
        )
    
    try:
        corpus = None
        
        # Option 1: Load from Data input (Table)
        if request.data_path:
            from app.widgets.data_utils import load_data
            table = load_data(request.data_path)
            
            if table is None:
                return CorpusResponse(
                    success=False,
                    error=f"Failed to load data from: {request.data_path}"
                )
            
            # Convert Table to Corpus
            # Find text features (StringVariable in metas or attributes)
            text_vars = []
            for var in list(table.domain.metas) + list(table.domain.attributes):
                if isinstance(var, StringVariable):
                    text_vars.append(var)
            
            if not text_vars:
                return CorpusResponse(
                    success=False,
                    error="No text features found in input data"
                )
            
            # Select used text features
            used_text_features = request.used_text_features or [text_vars[0].name]
            selected_text_vars = [v for v in text_vars if v.name in used_text_features]
            
            if not selected_text_vars:
                selected_text_vars = [text_vars[0]]
            
            # Create Corpus from Table
            corpus = Corpus.from_table(table.domain, table)
            corpus.set_text_features(selected_text_vars)
        
        # Option 2: Load from file
        elif request.file_path:
            corpus = Corpus.from_file(request.file_path)
        
        else:
            return CorpusResponse(
                success=False,
                error="Either file_path or data_path must be provided"
            )
        
        if corpus is None:
            return CorpusResponse(
                success=False,
                error=f"Failed to load corpus"
            )
        
        # Get text features
        text_features = [var.name for var in corpus.text_features] if hasattr(corpus, 'text_features') else []
        
        # Get meta features
        meta_features = [var.name for var in corpus.domain.metas] if corpus.domain.metas else []
        
        # Generate corpus ID
        corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
        
        # Store in cache
        _text_cache[corpus_id] = {
            'corpus': corpus,
            'file_path': request.file_path,
            'data_path': request.data_path,
            'title_variable': request.title_variable,
            'language': request.language,
            'used_text_features': request.used_text_features
        }
        
        # Create preview
        preview = []
        for i in range(min(5, len(corpus))):
            row = {}
            for var in corpus.domain.metas:
                try:
                    val = corpus[i, var]
                    row[var.name] = str(val) if val is not None else ""
                except:
                    row[var.name] = ""
            preview.append(row)
        
        return CorpusResponse(
            success=True,
            corpus_id=corpus_id,
            documents=len(corpus),
            text_features=text_features,
            meta_features=meta_features,
            title_variable=request.title_variable,
            language=request.language,
            preview=preview
        )
        
    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
        return CorpusResponse(
            success=False,
            error=str(e)
        )


@router.get("/corpus/{corpus_id}")
async def get_corpus_info(corpus_id: str) -> CorpusResponse:
    """Get information about a corpus."""
    if corpus_id not in _text_cache:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    cached = _text_cache[corpus_id]
    corpus = cached['corpus']
    
    text_features = [var.name for var in corpus.text_features] if hasattr(corpus, 'text_features') else []
    meta_features = [var.name for var in corpus.domain.metas] if corpus.domain.metas else []
    
    return CorpusResponse(
        success=True,
        corpus_id=corpus_id,
        documents=len(corpus),
        text_features=text_features,
        meta_features=meta_features,
        title_variable=cached.get('title_variable'),
        language=cached.get('language', 'English')
    )


# ============================================================================
# Preprocess Text Endpoints
# ============================================================================

@router.post("/preprocess")
async def preprocess_text(
    request: PreprocessRequest,
    x_session_id: Optional[str] = Header(None)
) -> PreprocessResponse:
    """Preprocess a corpus."""
    if not ORANGE_TEXT_AVAILABLE:
        return PreprocessResponse(
            success=False,
            error="Orange3-Text is not installed"
        )
    
    if request.corpus_id not in _text_cache:
        return PreprocessResponse(
            success=False,
            error="Corpus not found"
        )
    
    try:
        cached = _text_cache[request.corpus_id]
        corpus = cached['corpus']
        
        # Count tokens before preprocessing
        tokens_before = sum(len(doc.split()) for doc in corpus.documents) if hasattr(corpus, 'documents') else 0
        
        # Build preprocessor pipeline
        preprocessors = []
        ngram_info = None
        pos_info = None
        
        # Transformation
        if request.lowercase:
            preprocessors.append(LowercaseTransformer())
        if request.remove_accents:
            preprocessors.append(StripAccentsTransformer())
        if request.parse_html:
            preprocessors.append(HtmlTransformer())
        if request.remove_urls:
            preprocessors.append(UrlRemover())
        
        # Tokenization
        if request.tokenizer == 'word_punct':
            preprocessors.append(WordPunctTokenizer())
        elif request.tokenizer == 'whitespace':
            preprocessors.append(WhitespaceTokenizer())
        elif request.tokenizer == 'sentence':
            preprocessors.append(SentenceTokenizer())
        elif request.tokenizer == 'tweet':
            preprocessors.append(TweetTokenizer())
        else:  # regexp (default)
            preprocessors.append(RegexpTokenizer(pattern=request.regexp_pattern))
        
        # N-grams (after tokenization)
        if request.use_ngrams and HAS_NGRAMS:
            try:
                ngrams_processor = NGrams(ngram_range=(request.ngram_min, request.ngram_max))
                preprocessors.append(ngrams_processor)
                ngram_info = {
                    'enabled': True,
                    'range': [request.ngram_min, request.ngram_max]
                }
            except Exception as e:
                logger.warning(f"N-grams failed: {e}")
                ngram_info = {'enabled': False, 'error': str(e)}
        
        # POS Tagger (after tokenization)
        if request.use_pos_tagger and HAS_POS_TAGGER:
            try:
                pos_tagger = AveragedPerceptronTagger()
                # Note: POS tagging is applied separately in Orange3-Text
                # We'll store the info for later filtering
                pos_info = {
                    'enabled': True,
                    'tags_filter': request.pos_tags
                }
            except Exception as e:
                logger.warning(f"POS Tagger failed: {e}")
                pos_info = {'enabled': False, 'error': str(e)}
        
        # Filtering
        if request.use_stopwords:
            preprocessors.append(StopwordsFilter(language=request.stopwords_language.lower()))
        
        # Enhanced Frequency Filter
        min_df = request.min_df
        max_df = request.max_df
        
        # Support absolute document frequency
        if request.abs_min_df is not None and len(corpus) > 0:
            min_df = request.abs_min_df / len(corpus)
        if request.abs_max_df is not None and len(corpus) > 0:
            max_df = request.abs_max_df / len(corpus)
        
        if min_df is not None or max_df is not None:
            preprocessors.append(FrequencyFilter(
                min_df=min_df,
                max_df=max_df
            ))
        
        # Most frequent tokens filter
        if request.most_frequent_tokens is not None:
            try:
                preprocessors.append(FrequencyFilter(
                    keep_n=request.most_frequent_tokens
                ))
            except Exception as e:
                logger.warning(f"Most frequent filter not supported: {e}")
        
        if request.filter_regexp:
            preprocessors.append(RegexpFilter(pattern=request.filter_regexp))
        
        # Normalization / Stemming / Lemmatization
        if request.normalization == 'porter':
            preprocessors.append(Stemmer(method='porter'))
        elif request.normalization == 'snowball':
            preprocessors.append(Stemmer(method='snowball'))
        elif request.normalization == 'wordnet':
            preprocessors.append(Normalizer())
        elif request.normalization == 'udpipe' and HAS_UDPIPE:
            try:
                language = request.udpipe_language or cached.get('language', 'English')
                preprocessors.append(UDPipeLemmatizer(language=language))
            except Exception as e:
                logger.warning(f"UDPipe lemmatizer failed: {e}")
        
        # Apply preprocessors
        preprocessor = Preprocessor(preprocessors=preprocessors)
        processed_corpus = preprocessor(corpus)
        
        # Count tokens after preprocessing
        tokens_after = sum(len(doc) for doc in processed_corpus.tokens) if hasattr(processed_corpus, 'tokens') else 0
        
        # Store processed corpus
        new_corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
        _text_cache[new_corpus_id] = {
            'corpus': processed_corpus,
            'source_id': request.corpus_id,
            'preprocessed': True,
            'language': cached.get('language', 'English'),
            'ngram_info': ngram_info,
            'pos_info': pos_info
        }
        
        return PreprocessResponse(
            success=True,
            corpus_id=new_corpus_id,
            documents=len(processed_corpus),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            ngram_info=ngram_info,
            pos_info=pos_info
        )
        
    except Exception as e:
        logger.error(f"Error preprocessing corpus: {e}")
        import traceback
        traceback.print_exc()
        return PreprocessResponse(
            success=False,
            error=str(e)
        )


# ============================================================================
# Bag of Words Endpoints
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
    
    if request.corpus_id not in _text_cache:
        return BagOfWordsResponse(
            success=False,
            error="Corpus not found"
        )
    
    try:
        cached = _text_cache[request.corpus_id]
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
        _text_cache[data_id] = {
            'data': bow_corpus,
            'type': 'bow',
            'source_corpus_id': request.corpus_id,
            'vocabulary_size': vocab_size
        }
        
        # Also keep the corpus for downstream widgets
        corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
        _text_cache[corpus_id] = {
            'corpus': bow_corpus,
            'type': 'bow_corpus',
            'source_id': request.corpus_id
        }
        
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


# ============================================================================
# Word Cloud Endpoints
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
    if request.topic_id and request.topic_id in _text_cache:
        cached = _text_cache[request.topic_id]
        if cached.get('type') == 'topic':
            topic_words = cached.get('words', [])
    
    # Option 2: Corpus input
    if request.corpus_id and request.corpus_id in _text_cache:
        cached = _text_cache[request.corpus_id]
        corpus = cached.get('corpus') or cached.get('data')
    
    if corpus is None and topic_words is None:
        return WordCloudResponse(
            success=False,
            error="Corpus or Topic not found"
        )
    
    try:
        word_freq: Dict[str, int] = {}
        doc_freq: Dict[str, int] = {}  # Document frequency for Word Counts output
        
        # Use topic words if available
        if topic_words:
            for item in topic_words:
                word_freq[item['word']] = item.get('weight', 1)
        
        # Use corpus otherwise
        elif corpus is not None:
            if hasattr(corpus, 'tokens'):
                # Preprocessed corpus with tokens
                for doc_tokens in corpus.tokens:
                    doc_words = set()
                    for token in doc_tokens:
                        word_freq[token] = word_freq.get(token, 0) + 1
                        doc_words.add(token)
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
            elif hasattr(corpus, 'domain') and corpus.domain.attributes:
                # BOW corpus - use attribute names as words
                for attr in corpus.domain.attributes:
                    if hasattr(attr, 'name'):
                        # Sum column values as frequency
                        col_values = corpus.get_column(attr.name)
                        col_sum = sum(col_values)
                        doc_count = sum(1 for v in col_values if v > 0)
                        if col_sum > 0:
                            word_freq[attr.name] = int(col_sum)
                            doc_freq[attr.name] = doc_count
        
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
        _text_cache[word_counts_id] = {
            'type': 'word_counts',
            'data': word_counts_data,
            'total_words': len(word_freq),
            'source_id': request.corpus_id or request.topic_id
        }
        
        return WordCloudResponse(
            success=True,
            words=words,
            total_words=len(word_freq),
            word_counts_id=word_counts_id,
            selected_words=list(selected_set) if selected_set else []
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
    if word_counts_id not in _text_cache:
        raise HTTPException(status_code=404, detail="Word counts data not found")
    
    cached = _text_cache[word_counts_id]
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


@router.post("/wordcloud/select")
async def select_words(
    corpus_id: str,
    selected_words: List[str],
    x_session_id: Optional[str] = Header(None)
):
    """Filter corpus by selected words."""
    if corpus_id not in _text_cache:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    try:
        cached = _text_cache[corpus_id]
        corpus = cached.get('corpus') or cached.get('data')
        
        # Filter documents containing selected words
        selected_set = set(w.lower() for w in selected_words)
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
            _text_cache[new_corpus_id] = {
                'corpus': filtered_corpus,
                'source_id': corpus_id,
                'selected_words': selected_words
            }
            
            return {
                'success': True,
                'corpus_id': new_corpus_id,
                'documents': len(filtered_corpus),
                'selected_words': selected_words
            }
        else:
            return {
                'success': True,
                'corpus_id': None,
                'documents': 0,
                'selected_words': selected_words
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
    if item_id in _text_cache:
        del _text_cache[item_id]
        return {'success': True, 'message': f'Removed {item_id} from cache'}
    return {'success': False, 'message': 'Item not found'}


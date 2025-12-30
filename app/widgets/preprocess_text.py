"""
Preprocess Text Widget API endpoints.
Text preprocessing with tokenization, filtering, and normalization.
Supports ordered preprocessor pipeline (Orange3 style).
"""

import logging
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Header
from pydantic import BaseModel

from .text_mining_utils import (
    ORANGE_TEXT_AVAILABLE, HAS_NGRAMS, HAS_POS_TAGGER, HAS_UDPIPE,
    get_cache_item, set_cache_item
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text", tags=["Text Mining - Preprocess"])


# ============================================================================
# Models
# ============================================================================

class PreprocessorSettings(BaseModel):
    """Settings for a single preprocessor."""
    # Transformation
    lowercase: Optional[bool] = None
    removeAccents: Optional[bool] = None
    parseHtml: Optional[bool] = None
    removeUrls: Optional[bool] = None
    
    # Tokenization
    tokenizer: Optional[str] = None
    regexpPattern: Optional[str] = None
    
    # Normalization
    method: Optional[str] = None
    snowballLanguage: Optional[str] = None
    udpipeLanguage: Optional[str] = None
    useUdpipeTokenizer: Optional[bool] = None
    
    # Filtering
    useStopwords: Optional[bool] = None
    stopwordsLanguage: Optional[str] = None
    useLexicon: Optional[bool] = None
    filterNumbers: Optional[bool] = None
    includeNumbers: Optional[bool] = None
    filterRegexp: Optional[str] = None
    useDocFreq: Optional[bool] = None
    docFreqType: Optional[str] = None
    docFreqRelMin: Optional[float] = None
    docFreqRelMax: Optional[float] = None
    docFreqAbsMin: Optional[int] = None
    docFreqAbsMax: Optional[int] = None
    useMostFreq: Optional[bool] = None
    mostFreqN: Optional[int] = None
    
    # N-grams
    ngramMin: Optional[int] = None
    ngramMax: Optional[int] = None
    
    # POS Tagger
    posTags: Optional[str] = None


class PreprocessorItem(BaseModel):
    """A single preprocessor in the pipeline."""
    type: str  # transformation, tokenization, normalization, filtering, ngrams, pos
    settings: PreprocessorSettings


class PreprocessRequest(BaseModel):
    """Request to preprocess text with ordered preprocessors."""
    corpus_id: str
    preprocessors: List[PreprocessorItem]  # Ordered list of preprocessors


class PreprocessResponse(BaseModel):
    """Response with preprocessed corpus."""
    success: bool
    corpus_id: Optional[str] = None
    documents: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    types_before: int = 0
    types_after: int = 0
    preview_tokens: List[List[str]] = []
    ngram_info: Optional[Dict[str, Any]] = None
    pos_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============================================================================
# Preprocessor Builders
# ============================================================================

def build_transformation_preprocessors(settings: PreprocessorSettings):
    """Build transformation preprocessors."""
    from orangecontrib.text.preprocess import (
        LowercaseTransformer, StripAccentsTransformer,
        HtmlTransformer, UrlRemover
    )
    
    preprocessors = []
    if settings.lowercase:
        preprocessors.append(LowercaseTransformer())
    if settings.removeAccents:
        preprocessors.append(StripAccentsTransformer())
    if settings.parseHtml:
        preprocessors.append(HtmlTransformer())
    if settings.removeUrls:
        preprocessors.append(UrlRemover())
    
    return preprocessors


def build_tokenization_preprocessors(settings: PreprocessorSettings):
    """Build tokenization preprocessors."""
    from orangecontrib.text.preprocess import (
        WordPunctTokenizer, WhitespaceTokenizer,
        RegexpTokenizer, TweetTokenizer
    )
    
    # PunktSentenceTokenizer is optional
    try:
        from orangecontrib.text.preprocess import PunktSentenceTokenizer
        SentenceTokenizer = PunktSentenceTokenizer
    except ImportError:
        SentenceTokenizer = None
    
    tokenizer = settings.tokenizer or 'regexp'
    pattern = settings.regexpPattern or r'\w+'
    
    if tokenizer == 'word_punct':
        return [WordPunctTokenizer()]
    elif tokenizer == 'whitespace':
        return [WhitespaceTokenizer()]
    elif tokenizer == 'sentence':
        if SentenceTokenizer:
            return [SentenceTokenizer()]
        else:
            return [RegexpTokenizer(pattern=pattern)]
    elif tokenizer == 'tweet':
        return [TweetTokenizer()]
    else:  # regexp (default)
        return [RegexpTokenizer(pattern=pattern)]


def build_normalization_preprocessors(settings: PreprocessorSettings):
    """Build normalization preprocessors."""
    from orangecontrib.text.preprocess import PorterStemmer, SnowballStemmer
    
    try:
        from orangecontrib.text.preprocess import WordNetLemmatizer
    except ImportError:
        WordNetLemmatizer = None
    
    method = settings.method or 'none'
    
    if method == 'porter':
        return [PorterStemmer()]
    elif method == 'snowball':
        lang = settings.snowballLanguage or 'en'
        return [SnowballStemmer(language=lang)]
    elif method == 'wordnet':
        if WordNetLemmatizer:
            return [WordNetLemmatizer()]
        else:
            logger.warning("WordNetLemmatizer not available")
            return []
    elif method == 'udpipe':
        try:
            from orangecontrib.text.preprocess import UDPipeLemmatizer
            lang = settings.udpipeLanguage or 'English'
            return [UDPipeLemmatizer(language=lang)]
        except ImportError:
            logger.warning("UDPipeLemmatizer not available")
            return []
    else:
        return []


def build_filtering_preprocessors(settings: PreprocessorSettings, n_docs: int):
    """Build filtering preprocessors."""
    from orangecontrib.text.preprocess import (
        StopwordsFilter, FrequencyFilter, RegexpFilter
    )
    
    # Language code mapping
    lang_map = {
        'english': 'en', 'german': 'de', 'french': 'fr', 'spanish': 'es',
        'italian': 'it', 'dutch': 'nl', 'portuguese': 'pt', 'russian': 'ru',
        'danish': 'da', 'finnish': 'fi', 'hungarian': 'hu', 'norwegian': 'no',
        'swedish': 'sv', 'turkish': 'tr'
    }
    
    preprocessors = []
    
    # Stopwords
    if settings.useStopwords:
        lang_code = lang_map.get((settings.stopwordsLanguage or 'English').lower(), 'en')
        preprocessors.append(StopwordsFilter(language=lang_code))
    
    # Regexp filter
    if settings.filterRegexp:
        preprocessors.append(RegexpFilter(pattern=settings.filterRegexp))
    
    # Document Frequency filter
    if settings.useDocFreq:
        if settings.docFreqType == 'relative':
            preprocessors.append(FrequencyFilter(
                min_df=settings.docFreqRelMin or 0.1,
                max_df=settings.docFreqRelMax or 0.9
            ))
        else:  # absolute
            min_df = (settings.docFreqAbsMin or 1) / max(n_docs, 1)
            max_df = (settings.docFreqAbsMax or 10) / max(n_docs, 1)
            preprocessors.append(FrequencyFilter(min_df=min_df, max_df=max_df))
    
    # Most Frequent filter
    if settings.useMostFreq:
        try:
            from orangecontrib.text.preprocess import MostFrequentTokensFilter
            preprocessors.append(MostFrequentTokensFilter(keep_n=settings.mostFreqN or 100))
        except ImportError:
            logger.warning("MostFrequentTokensFilter not available")
    
    return preprocessors


def build_ngrams_preprocessors(settings: PreprocessorSettings):
    """Build N-grams preprocessors."""
    if not HAS_NGRAMS:
        return []
    
    try:
        from orangecontrib.text.preprocess import NGrams
        min_n = settings.ngramMin or 1
        max_n = settings.ngramMax or 2
        # Note: parameter is 'ngrams_range' not 'ngram_range'
        return [NGrams(ngrams_range=(min_n, max_n))]
    except ImportError:
        logger.warning("NGrams not available")
        return []


def build_pos_preprocessors(settings: PreprocessorSettings):
    """Build POS tagger preprocessors."""
    if not HAS_POS_TAGGER:
        return []
    
    try:
        from orangecontrib.text.tag import AveragedPerceptronTagger
        # POS tagger itself - filtering would be separate
        return [AveragedPerceptronTagger()]
    except ImportError:
        logger.warning("POS Tagger not available")
        return []


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/preprocess")
async def preprocess_text(
    request: PreprocessRequest,
    x_session_id: Optional[str] = Header(None)
) -> PreprocessResponse:
    """Preprocess a corpus with ordered preprocessors."""
    if not ORANGE_TEXT_AVAILABLE:
        return PreprocessResponse(
            success=False,
            error="Orange3-Text is not installed"
        )
    
    cached = get_cache_item(request.corpus_id)
    if not cached:
        return PreprocessResponse(
            success=False,
            error="Corpus not found"
        )
    
    try:
        from orangecontrib.text.preprocess import PreprocessorList
        
        corpus = cached['corpus']
        n_docs = len(corpus) if hasattr(corpus, '__len__') else 0
        
        # Count tokens and types before preprocessing
        tokens_before = 0
        types_before_set = set()
        if hasattr(corpus, 'documents'):
            for doc in corpus.documents:
                words = doc.split()
                tokens_before += len(words)
                types_before_set.update(w.lower() for w in words)
        types_before = len(types_before_set)
        
        # Build preprocessor pipeline in order
        all_preprocessors = []
        ngram_info = None
        pos_info = None
        
        for pp_item in request.preprocessors:
            pp_type = pp_item.type
            settings = pp_item.settings
            
            if pp_type == 'transformation':
                all_preprocessors.extend(build_transformation_preprocessors(settings))
            elif pp_type == 'tokenization':
                all_preprocessors.extend(build_tokenization_preprocessors(settings))
            elif pp_type == 'normalization':
                all_preprocessors.extend(build_normalization_preprocessors(settings))
            elif pp_type == 'filtering':
                all_preprocessors.extend(build_filtering_preprocessors(settings, n_docs))
            elif pp_type == 'ngrams':
                ngram_pps = build_ngrams_preprocessors(settings)
                if ngram_pps:
                    all_preprocessors.extend(ngram_pps)
                    ngram_info = {
                        'enabled': True,
                        'range': [settings.ngramMin or 1, settings.ngramMax or 2]
                    }
            elif pp_type == 'pos':
                pos_pps = build_pos_preprocessors(settings)
                if pos_pps:
                    all_preprocessors.extend(pos_pps)
                    pos_info = {
                        'enabled': True,
                        'tags_filter': settings.posTags
                    }
        
        logger.info(f"Built preprocessor pipeline with {len(all_preprocessors)} preprocessors")
        
        # Apply preprocessors
        if all_preprocessors:
            preprocessor = PreprocessorList(all_preprocessors)
            processed_corpus = preprocessor(corpus)
        else:
            processed_corpus = corpus
        
        # Count tokens and types after preprocessing
        tokens_after = 0
        types_after_set = set()
        preview_tokens = []
        
        if hasattr(processed_corpus, 'tokens'):
            for i, doc_tokens in enumerate(processed_corpus.tokens):
                tokens_after += len(doc_tokens)
                types_after_set.update(doc_tokens)
                # Get first document's first 5 tokens for preview (Orange3 style)
                if i == 0:
                    preview_tokens = [list(doc_tokens[:5])]
        
        types_after = len(types_after_set)
        
        # Store processed corpus
        new_corpus_id = f"corpus_{uuid.uuid4().hex[:8]}"
        set_cache_item(new_corpus_id, {
            'corpus': processed_corpus,
            'source_id': request.corpus_id,
            'preprocessed': True,
            'language': cached.get('language', 'English'),
            'ngram_info': ngram_info,
            'pos_info': pos_info
        })
        
        return PreprocessResponse(
            success=True,
            corpus_id=new_corpus_id,
            documents=len(processed_corpus),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            types_before=types_before,
            types_after=types_after,
            preview_tokens=preview_tokens,
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

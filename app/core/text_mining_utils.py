"""
Text Mining Utilities - Shared imports and cache for text mining widgets.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# In-memory cache for text data (shared across text mining widgets)
_text_cache: Dict[str, Any] = {}

# Check Orange3-Text availability
ORANGE_TEXT_AVAILABLE = False
HAS_NGRAMS = False
HAS_POS_TAGGER = False
HAS_UDPIPE = False
HAS_TOPICS = False
HAS_SENTENCE_TOKENIZER = False

try:
    from orangecontrib.text import Corpus
    from orangecontrib.text.preprocess import (
        Preprocessor, LowercaseTransformer, StripAccentsTransformer,
        HtmlTransformer, UrlRemover, RegexpTokenizer, WhitespaceTokenizer,
        WordPunctTokenizer, TweetTokenizer,
        StopwordsFilter, FrequencyFilter, LexiconFilter, RegexpFilter,
        PorterStemmer, SnowballStemmer, NGrams
    )
    from orangecontrib.text.preprocess.normalize import BaseNormalizer
    from orangecontrib.text.vectorization import BowVectorizer
    from Orange.data import Table, Domain, StringVariable, ContinuousVariable
    ORANGE_TEXT_AVAILABLE = True
    HAS_NGRAMS = True
    logger.info("Orange3-Text is available")
    
    # Alias for backward compatibility
    Stemmer = PorterStemmer
    
    # Try to import optional components
    try:
        from orangecontrib.text.preprocess import PunktSentenceTokenizer
        SentenceTokenizer = PunktSentenceTokenizer
        HAS_SENTENCE_TOKENIZER = True
    except ImportError:
        SentenceTokenizer = None
        HAS_SENTENCE_TOKENIZER = False
        
    try:
        from orangecontrib.text.tag import AveragedPerceptronTagger, POSTagger
        HAS_POS_TAGGER = True
    except ImportError:
        HAS_POS_TAGGER = False
        
    try:
        from orangecontrib.text.preprocess import UDPipeLemmatizer
        HAS_UDPIPE = True
    except ImportError:
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
    logger.warning(f"Orange3-Text not available: {e}")


def get_text_cache() -> Dict[str, Any]:
    """Get the shared text cache."""
    return _text_cache


def set_cache_item(key: str, value: Any) -> None:
    """Set an item in the text cache."""
    _text_cache[key] = value


def get_cache_item(key: str) -> Any:
    """Get an item from the text cache."""
    return _text_cache.get(key)


def delete_cache_item(key: str) -> bool:
    """Delete an item from the text cache."""
    if key in _text_cache:
        del _text_cache[key]
        return True
    return False


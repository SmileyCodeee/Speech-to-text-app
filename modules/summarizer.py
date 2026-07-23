"""
summarizer.py
--------------
Produces a short summary of a (potentially long) transcript.

Two modes are supported:

1. "extractive" (default, always available, works for ANY language):
   A frequency-based extractive summarizer (a lightweight variant of the
   classic Luhn/TextRank family of algorithms) that scores sentences by
   the importance of the words they contain and picks the top-scoring
   sentences, in original order. No model download needed, so this keeps
   the app fully offline and language-independent, matching the project's
   privacy/offline goals.

2. "abstractive" (optional, requires `transformers` + `torch` + an
   internet connection the first time to download the model):
   Uses a Hugging Face summarization pipeline (e.g. facebook/bart-large-cnn
   for English, or a multilingual model such as csebuetnlp/mT5_multilingual_XLSum)
   to generate a fluent, rewritten summary.
   Reference: https://huggingface.co/docs/transformers/main_classes/pipelines
"""

import re
from collections import Counter
from typing import List, Optional

from .text_processor import split_sentences

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "and", "or", "but",
    "if", "then", "so", "as", "it", "this", "that", "to", "of", "in", "on",
    "for", "with", "like", "i", "we", "you", "there", "here", "now", "not",
}

def summarize_extractive(text: str, num_sentences: int = 5) -> str:
    """Language-agnostic extractive summarization (no external model)."""
    sentences = split_sentences(text)

    # Drop filler/disfluency fragments (very short, mostly stopwords) so
    # they can't dominate scoring just for being short and repetitive.
    candidates = [s for s in sentences if len(re.findall(r"\w+", s)) >= 5]
    if not candidates:
        candidates = sentences

    if len(candidates) <= 2:
        return " ".join(candidates)

    target = max(1, min(num_sentences, round(len(candidates) * 0.4)))

    # Build frequency table excluding stopwords, so common filler words
    # don't inflate the score of short, low-content fragments.
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_freq

    scores = []
    for idx, sentence in enumerate(candidates):
        sentence_words = [w for w in re.findall(r"\w+", sentence.lower()) if w not in _STOPWORDS]
        if not sentence_words:
            continue
        score = sum(freq.get(w, 0) for w in sentence_words) / len(sentence_words)
        scores.append((idx, score, sentence))

    top = sorted(scores, key=lambda t: -t[1])[:target]
    top_in_order = sorted(top, key=lambda t: t[0])
    return " ".join(s for _, _, s in top_in_order)

_ABSTRACTIVE_PIPELINE = None


def summarize_abstractive(text: str, model_name: str = "facebook/bart-large-cnn") -> str:
    global _ABSTRACTIVE_PIPELINE
    if _ABSTRACTIVE_PIPELINE is None:
        from transformers import pipeline
        _ABSTRACTIVE_PIPELINE = pipeline("summarization", model=model_name)

    max_chars = 3000
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [text]

    summaries = []
    try:
        for chunk in chunks:
            if not chunk.strip():
                continue
            out = _ABSTRACTIVE_PIPELINE(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(out[0]["summary_text"])
    except Exception:
        # The cached pipeline's underlying HTTP client may be dead
        # (e.g. interrupted by a Streamlit rerun mid-download) — drop it
        # so the next call rebuilds a fresh one instead of reusing it.
        _ABSTRACTIVE_PIPELINE = None
        raise

    return " ".join(summaries)

def summarize(
    text: str,
    method: str = "extractive",
    num_sentences: int = 5,
    model_name: Optional[str] = None,
) -> str:
    if not text.strip():
        return ""
    if method == "abstractive":
        try:
            return summarize_abstractive(text, model_name or "facebook/bart-large-cnn")
        except Exception as exc:
            # Graceful fallback keeps the app usable even if transformers/
            # torch isn't installed or the model can't be downloaded
            # (e.g. no internet access).
            fallback = summarize_extractive(text, num_sentences)
            return f"[Abstractive summarizer unavailable ({exc}); showing extractive summary]\n{fallback}"
    return summarize_extractive(text, num_sentences)
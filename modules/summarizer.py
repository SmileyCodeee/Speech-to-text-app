"""
summarizer.py
--------------
Produces a short summary of a (potentially long) transcript.

Two modes are supported:

1. "extractive" (default, always available, works for ANY language):
   A graph-based TextRank summarizer: sentences are scored by how
   similar (in shared vocabulary) they are to every other sentence, then
   ranked using the same principle as PageRank - a sentence that echoes
   the themes of many other sentences scores higher than one that's just
   locally word-dense. This is noticeably more accurate than plain
   frequency scoring, which can be fooled by short fragments reusing a
   few common words. No model download needed, so this keeps the app
   fully offline and language-independent.

2. "abstractive" (optional, requires `transformers` + `torch` + an
   internet connection the first time to download the model):
   Loads facebook/bart-large-cnn (or another seq2seq model) directly via
   AutoTokenizer/AutoModelForSeq2SeqLM and runs generation manually,
   bypassing transformers' `pipeline("summarization")` task wrapper —
   some transformers installs/versions report "Unknown task summarization"
   from the pipeline registry even though the underlying model/tokenizer
   classes work fine, so calling them directly is more robust.
   Reference: https://huggingface.co/docs/transformers/model_doc/bart
"""

import re
from collections import Counter
from typing import List, Optional

import numpy as np

from .text_processor import split_sentences

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "and", "or", "but",
    "if", "then", "so", "as", "it", "this", "that", "to", "of", "in", "on",
    "for", "with", "like", "i", "we", "you", "there", "here", "now", "not",
    "just", "will", "can", "our", "your", "be", "do", "does", "did",
}


def _sentence_vector(sentence: str, vocab_index: dict) -> np.ndarray:
    """Bag-of-words vector for one sentence, stopwords excluded."""
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    words = [w for w in re.findall(r"\w+", sentence.lower()) if w not in _STOPWORDS]
    for w in words:
        if w in vocab_index:
            vec[vocab_index[w]] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _textrank_scores(sentences: List[str], damping: float = 0.85, iterations: int = 30) -> np.ndarray:
    """
    Score sentences by graph centrality (TextRank): build a similarity
    matrix between all sentence pairs (cosine similarity over word
    vectors), then run PageRank-style iteration so a sentence's score
    reflects how strongly it connects to the rest of the transcript,
    not just its own word density.
    """
    n = len(sentences)
    vocab = sorted({
        w for s in sentences
        for w in re.findall(r"\w+", s.lower())
        if w not in _STOPWORDS
    })
    vocab_index = {w: i for i, w in enumerate(vocab)}

    vectors = np.stack([_sentence_vector(s, vocab_index) for s in sentences])

    sim = vectors @ vectors.T
    np.fill_diagonal(sim, 0.0)

    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    scores = np.full(n, 1.0 / n, dtype=np.float32)
    for _ in range(iterations):
        scores = (1 - damping) / n + damping * (transition.T @ scores)

    return scores


def summarize_extractive(text: str, num_sentences: int = 5) -> str:
    """Graph-based (TextRank) extractive summarization - no external model."""
    sentences = split_sentences(text)

    # Drop filler/disfluency fragments (very short, mostly stopwords) so
    # they can't enter the graph and dilute genuinely content-bearing
    # sentences' connections to each other.
    candidates = [s for s in sentences if len(re.findall(r"\w+", s)) >= 5]
    if not candidates:
        candidates = sentences

    if len(candidates) <= 2:
        return " ".join(candidates)

    target = max(1, min(num_sentences, round(len(candidates) * 0.4)))

    scores = _textrank_scores(candidates)

    ranked = sorted(range(len(candidates)), key=lambda i: -scores[i])[:target]
    ranked_in_order = sorted(ranked)
    return " ".join(candidates[i] for i in ranked_in_order)


_ABSTRACTIVE_MODEL = None
_ABSTRACTIVE_TOKENIZER = None
_ABSTRACTIVE_MODEL_NAME = None


def summarize_abstractive(text: str, model_name: str = "facebook/bart-large-cnn") -> str:
    """
    Abstractive summarization via Hugging Face Transformers, loaded
    directly (not through `pipeline(...)`) for robustness across
    transformers versions.
    """
    global _ABSTRACTIVE_MODEL, _ABSTRACTIVE_TOKENIZER, _ABSTRACTIVE_MODEL_NAME

    if _ABSTRACTIVE_MODEL is None or _ABSTRACTIVE_MODEL_NAME != model_name:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _ABSTRACTIVE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _ABSTRACTIVE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _ABSTRACTIVE_MODEL_NAME = model_name

    # Overlapping chunks (200-char overlap) so a sentence split across a
    # chunk boundary isn't lost from both halves' context.
    max_chars = 3000
    overlap = 200
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + max_chars])
        i += max_chars - overlap
    if not chunks:
        chunks = [text]

    summaries = []
    try:
        for chunk in chunks:
            if not chunk.strip():
                continue
            inputs = _ABSTRACTIVE_TOKENIZER(
                chunk, return_tensors="pt", truncation=True, max_length=1024
            )
            output_ids = _ABSTRACTIVE_MODEL.generate(
                **inputs,
                max_length=130,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                do_sample=False,
            )
            summary = _ABSTRACTIVE_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
            summaries.append(summary)
    except Exception:
        _ABSTRACTIVE_MODEL = None
        _ABSTRACTIVE_TOKENIZER = None
        _ABSTRACTIVE_MODEL_NAME = None
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
            fallback = summarize_extractive(text, num_sentences)
            return f"[Abstractive summarizer unavailable ({exc}); showing extractive summary]\n{fallback}"
    return summarize_extractive(text, num_sentences)
"""
text_processor.py
------------------
Turns a raw transcript (plain text or Whisper segments with timestamps)
into structured notes: paragraphs / topic sections / bullet points.

Approach (language-agnostic, no extra ML model required so it works for
any language Whisper can transcribe):
  1. Sentence segmentation using punctuation-aware regex rules.
  2. Paragraph/topic breaks inferred from pauses between Whisper segments
     (a gap of PAUSE_THRESHOLD seconds or more usually signals a new
     thought/topic) - falls back to a fixed sentence-count window when
     timestamps aren't available (e.g. plain pasted text).
  3. Optional bullet-point view, one bullet per sentence, useful for quick
     scanning.

Keyword extraction uses YAKE (Yet Another Keyword Extractor), a
lightweight, unsupervised, statistics-based, MULTILINGUAL keyword
extractor that needs no training data or internet access.
Reference: https://github.com/LIAAD/yake
"""

import re
from dataclasses import dataclass
from typing import List, Optional

try:
    import yake
except ImportError:  # pragma: no cover - optional dependency at import time
    yake = None

PAUSE_THRESHOLD = 2.0          # seconds - gap that triggers a new paragraph
SENTENCES_PER_PARAGRAPH = 4    # fallback when no timestamps are available

# Basic sentence-final punctuation for major scripts (Latin, Devanagari,
# Arabic, CJK use different terminal punctuation; regex covers the common
# ones so the tool degrades gracefully across languages).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।؟。！？])\s+")


@dataclass
class StructuredNotes:
    paragraphs: List[str]
    bullets: List[str]
    raw_text: str


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # If punctuation-based splitting produced only one chunk (common with
    # unpunctuated Whisper output from casual speech), fall back to
    # breaking the text into fixed-size word windows so downstream
    # summarization has multiple units to score instead of one giant blob.
    if len(sentences) <= 1:
        words = text.split()
        window = 12  # words per pseudo-sentence
        sentences = [
            " ".join(words[i:i + window])
            for i in range(0, len(words), window)
        ]
        sentences = [s for s in sentences if s.strip()]

    return sentences


def structure_from_segments(segments) -> StructuredNotes:
    """Build paragraphs using pause gaps between Whisper segments."""
    if not segments:
        return StructuredNotes(paragraphs=[], bullets=[], raw_text="")

    paragraphs = []
    current = [segments[0].text.strip()]

    for prev, curr in zip(segments, segments[1:]):
        gap = curr.start - prev.end
        if gap >= PAUSE_THRESHOLD:
            paragraphs.append(" ".join(current).strip())
            current = [curr.text.strip()]
        else:
            current.append(curr.text.strip())
    if current:
        paragraphs.append(" ".join(current).strip())

    paragraphs = [p for p in paragraphs if p]
    raw_text = " ".join(s.text.strip() for s in segments)
    bullets = split_sentences(raw_text)

    return StructuredNotes(paragraphs=paragraphs, bullets=bullets, raw_text=raw_text)


def structure_from_text(text: str) -> StructuredNotes:
    """Fallback structuring for plain text with no timestamp info."""
    sentences = split_sentences(text)
    paragraphs = [
        " ".join(sentences[i:i + SENTENCES_PER_PARAGRAPH])
        for i in range(0, len(sentences), SENTENCES_PER_PARAGRAPH)
    ]
    return StructuredNotes(paragraphs=paragraphs, bullets=sentences, raw_text=text)


def extract_keywords(text: str, language_code: Optional[str] = "en", top_n: int = 10) -> List[str]:
    """
    Extract top_n keywords/keyphrases from text using YAKE.
    language_code should be a 2-letter ISO code; YAKE supports dozens of
    languages and falls back reasonably well even if the exact code isn't
    in its curated stopword list.
    """
    if not text.strip():
        return []
    if yake is None:
        return _fallback_keyword_frequency(text, top_n)

    lang = (language_code or "en")[:2]
    try:
        extractor = yake.KeywordExtractor(lan=lang, n=2, top=top_n, dedupLim=0.9)
        keywords = extractor.extract_keywords(text)
    except Exception:
        # Unsupported language for YAKE's stopword corpus -> use English
        # rules as a generic fallback rather than failing the whole app.
        extractor = yake.KeywordExtractor(lan="en", n=2, top=top_n, dedupLim=0.9)
        keywords = extractor.extract_keywords(text)

    # yake returns (keyword, score) with LOWER score = more relevant
    keywords.sort(key=lambda kv: kv[1])
    return [kw for kw, _ in keywords[:top_n]]


def _fallback_keyword_frequency(text: str, top_n: int) -> List[str]:
    """Very simple frequency-based fallback if YAKE isn't installed."""
    words = re.findall(r"\w+", text.lower())
    stop = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "to",
             "of", "in", "on", "for", "with", "that", "this", "it", "as"}
    freq = {}
    for w in words:
        if len(w) < 3 or w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    return [w for w, _ in ranked[:top_n]]

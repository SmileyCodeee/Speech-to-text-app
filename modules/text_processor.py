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

After YAKE extraction, a comprehensive post-filter removes generic words
(such as common verbs, fillers, and overly broad nouns) that YAKE's own
stopword list may miss. Single-word keywords that are already part of a
multi-word keyphrase are also deduplicated — e.g., if "machine learning"
is returned, "machine" and "learning" as standalone keywords are removed
since the phrase is more informative.

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


# ──────────────────────────────────────────────────────────
# Comprehensive keyword blacklist — words that should NEVER
# appear as standalone keywords because they're too generic,
# non-topical, or are speech fillers. YAKE's own stopword
# lists cover basic function words but miss many common verbs,
# adjectives, and speech fillers that slip through as "keywords".
#
# Multi-word keyphrases are checked differently: a phrase is
# removed only if ALL its component words are in the blacklist.
# So "machine learning" survives even if "learning" is listed,
# because "machine" isn't — the phrase carries topical meaning.
# ──────────────────────────────────────────────────────────
_KEYWORD_BLACKLIST = {
    # ── Articles & determiners ──
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "all", "each", "every",
    "both", "few", "more", "most", "other", "another", "such", "no",

    # ── Prepositions ──
    "about", "above", "across", "after", "against", "along", "among",
    "around", "at", "before", "behind", "below", "between", "beyond",
    "by", "down", "from", "in", "into", "near", "of", "off", "on",
    "out", "over", "past", "through", "to", "under", "up", "upon",
    "with", "within", "without",

    # ── Pronouns ──
    "i", "me", "myself", "we", "us", "ourselves", "you", "yourself",
    "he", "him", "himself", "she", "her", "herself", "it", "itself",
    "they", "them", "themselves", "who", "whom", "whose", "which",
    "what", "where", "when", "why", "how",

    # ── Auxiliary/modal verbs ──
    "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must",

    # ── Common non-topical verbs ──
    "get", "got", "make", "made", "take", "took", "come", "came", "go",
    "going", "gone", "say", "said", "tell", "told", "know", "knew",
    "think", "thought", "see", "saw", "find", "found", "give", "gave",
    "use", "used", "want", "look", "looked", "work", "worked", "call",
    "called", "try", "tried", "ask", "asked", "need", "feel", "felt",
    "become", "became", "leave", "left", "put", "mean", "meant", "keep",
    "kept", "let", "begin", "began", "seem", "seemed", "help", "helped",
    "show", "showed", "hear", "heard", "play", "run", "ran", "move",
    "moved", "live", "lived", "bring", "brought", "happen", "happened",
    "write", "written", "sit", "stand", "lose", "lost", "pay", "paid",
    "meet", "met", "include", "continue", "change", "lead", "led",
    "understand", "watch", "follow", "stop", "create", "speak", "read",
    "allow", "add", "spend", "grow", "open", "walk", "win", "offer",
    "remember", "consider", "appear", "buy", "wait", "serve", "die",
    "send", "sent", "expect", "build", "stay", "fall", "cut", "reach",
    "remain", "suggest", "raise", "pass", "sell", "require", "report",
    "decide", "pull", "set", "turn",

    # ── Common non-topical adjectives/adverbs ──
    "also", "just", "very", "really", "often", "however", "too",
    "usually", "already", "always", "sometimes", "never", "soon",
    "still", "ever", "again", "much", "many", "less", "better", "best",
    "well", "bad", "same", "different", "important", "possible",
    "necessary", "able", "available", "likely", "actual", "basic",
    "clear", "close", "common", "complete", "current", "easy", "enough",
    "fair", "free", "full", "good", "great", "hard", "high", "low",
    "major", "new", "old", "real", "recent", "right", "small", "special",
    "strong", "sure", "true", "whole", "long", "short", "big", "little",
    "first", "last", "next", "own", "main", "simple", "single", "general",
    "specific", "natural", "normal", "typical", "certain", "obvious",
    "essential", "significant", "relevant", "effective", "successful",
    "positive", "negative", "initial", "final", "primary", "overall",
    "rather", "quite", "pretty", "fairly", "somewhat", "maybe", "perhaps",
    "probably", "definitely", "certainly", "absolutely", "totally",
    "completely", "exactly",

    # ── Generic nouns (too broad to carry topical meaning as keywords) ──
    "thing", "things", "something", "anything", "nothing", "everything",
    "part", "point", "fact", "reason", "way", "case", "area", "side",
    "end", "place", "person", "people", "group", "number", "world",
    "life", "year", "day", "week", "month", "hand", "water", "home",
    "room", "word", "idea", "kind", "sort", "class", "category",
    "problem", "question", "answer", "result", "effect", "situation",
    "condition", "standard", "value", "role", "function", "feature",
    "aspect", "detail", "element", "component", "factor", "issue",
    "item", "section", "step", "phase",

    # ── Speech fillers & discourse markers ──
    "like", "okay", "yeah", "yes", "no", "um", "uh", "hmm", "actually",
    "basically", "literally", "seriously", "honestly", "obviously",
    "essentially", "generally", "simply", "merely", "so", "then",
    "there", "here", "now",

    # ── Conjunctions ──
    "and", "or", "but", "nor", "yet", "although", "though", "because",
    "since", "unless", "until", "while", "whether", "either", "neither",
    "besides", "moreover", "nevertheless", "otherwise", "therefore",
    "thus", "hence", "instead", "despite",

    # ── Gerund/progressive forms ──
    "getting", "doing", "having", "being", "looking", "talking", "trying",
    "using", "working", "making", "taking", "coming", "giving", "keeping",
    "putting", "running", "turning", "moving", "living", "showing",
    "feeling", "leaving", "calling", "asking", "telling", "knowing",
    "thinking", "seeing", "finding", "wanting", "saying",
}


def _filter_keywords(keywords: List[str], blacklist: set) -> List[str]:
    """
    Post-filter YAKE keywords to remove generic/useless terms.

    Rules:
      1. Remove keywords where ALL component words are in the blacklist.
         ("really important" → both words blacklisted → removed)
         But "machine learning" survives because "machine" isn't
         blacklisted, even though "learning" is.
      2. Remove keywords that are pure numbers.
      3. Remove keywords shorter than 3 characters total.
    """
    result = []
    for kw in keywords:
        kw_words = kw.lower().split()

        # Remove if every component word is blacklisted
        if all(w in blacklist for w in kw_words):
            continue

        # Remove pure numbers
        if all(w.isdigit() for w in kw_words):
            continue

        # Remove very short keywords
        if len(kw.strip()) < 3:
            continue

        result.append(kw)

    return result


def _dedup_keywords(keywords: List[str]) -> List[str]:
    """
    Remove single-word keywords that are already part of a multi-word
    keyphrase. For example, if both "machine learning" and "machine"
    are in the list, remove "machine" because the phrase is more
    informative and already covers that word.
    """
    multi_word = [kw for kw in keywords if len(kw.split()) > 1]
    single_word = [kw for kw in keywords if len(kw.split()) == 1]

    # Collect all words that appear inside multi-word keyphrases
    covered_words = set()
    for mw in multi_word:
        for w in mw.lower().split():
            covered_words.add(w)

    # Keep single-word keywords only if they're not covered by a phrase
    kept_singles = [sw for sw in single_word if sw.lower() not in covered_words]

    # Return multi-word keywords first (more informative), then singles
    return multi_word + kept_singles


def extract_keywords(text: str, language_code: Optional[str] = "en", top_n: int = 10) -> List[str]:
    """
    Extract top_n keywords/keyphrases from text using YAKE, then
    post-filter to remove generic/useless terms and deduplicate
    single-word keywords covered by multi-word phrases.

    YAKE settings tuned for better topical accuracy:
      - n=3: captures up to 3-word phrases (e.g. "machine learning",
        "natural language processing"), which are more informative
        than single words.
      - dedupLim=0.6: stricter deduplication — removes near-duplicate
        keywords that differ by only one word.
      - windowsSize=2: larger context window for co-occurrence
        statistics, improving phrase quality.
      - top=top_n*3: over-extract from YAKE so the post-filter has
        more candidates to choose from.

    language_code should be a 2-letter ISO code; YAKE supports dozens
    of languages and falls back reasonably well even if the exact code
    isn't in its curated stopword list.
    """
    if not text.strip():
        return []
    if yake is None:
        return _fallback_keyword_frequency(text, top_n)

    lang = (language_code or "en")[:2]

    # Over-extract from YAKE so the post-filter has plenty of candidates.
    # Request 3x the desired count; after filtering we'll still have enough.
    extract_count = top_n * 3

    try:
        extractor = yake.KeywordExtractor(
            lan=lang,
            n=3,            # up to 3-word keyphrases
            top=extract_count,
            dedupLim=0.6,   # stricter dedup (was 0.9)
            windowsSize=2,  # larger context window (was default 1)
        )
        raw_keywords = extractor.extract_keywords(text)
    except Exception:
        # Unsupported language for YAKE's stopword corpus -> use English
        # rules as a generic fallback rather than failing the whole app.
        extractor = yake.KeywordExtractor(
            lan="en", n=3, top=extract_count, dedupLim=0.6, windowsSize=2
        )
        raw_keywords = extractor.extract_keywords(text)

    # yake returns (keyword, score) with LOWER score = more relevant
    raw_keywords.sort(key=lambda kv: kv[1])
    kw_list = [kw for kw, _ in raw_keywords[:extract_count]]

    # ── Post-filter: remove generic/useless keywords ──
    kw_list = _filter_keywords(kw_list, _KEYWORD_BLACKLIST)

    # ── Dedup: remove single-word keywords covered by phrases ──
    kw_list = _dedup_keywords(kw_list)

    return kw_list[:top_n]


def _fallback_keyword_frequency(text: str, top_n: int) -> List[str]:
    """
    Frequency-based fallback if YAKE isn't installed. Uses an expanded
    stopword list and TF-IDF-like scoring: words appearing in many
    sentences are downweighted (too common to be distinctive keywords).
    """
    words = re.findall(r"\w+", text.lower())

    # Expanded stopword set for the fallback path
    stop = _KEYWORD_BLACKLIST | {
        "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
        "don", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
    }

    # Count word frequencies
    freq = {}
    for w in words:
        if len(w) < 3 or w in stop or w.isdigit():
            continue
        freq[w] = freq.get(w, 0) + 1

    # Simple IDF-like penalty: count how many "sentences" (roughly
    # newline-or-period-delimited chunks) each word appears in.
    # Words in almost every chunk are too common to be keywords.
    chunks = re.split(r"[.\n!?]", text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
    if chunks:
        chunk_freq = {}
        for w in freq:
            chunk_freq[w] = sum(1 for c in chunks if w in c.lower())
        # Penalize words appearing in >50% of chunks
        threshold = len(chunks) * 0.5
        scored = {}
        for w, count in freq.items():
            penalty = 1.0
            if chunk_freq.get(w, 0) > threshold:
                penalty = 0.3  # heavy penalty for overly common words
            scored[w] = count * penalty
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    else:
        ranked = sorted(freq.items(), key=lambda kv: -kv[1])

    return [w for w, _ in ranked[:top_n]]
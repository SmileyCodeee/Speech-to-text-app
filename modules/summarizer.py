"""
summarizer.py
--------------
Produces a short summary of a (potentially long) transcript.

Two modes are supported:

1. "extractive" (default, always available, works for ANY language):
   A graph-based TextRank summarizer with TF-IDF weighting: sentences are
   represented as TF-IDF vectors (not raw word counts), so words that
   appear in every sentence (generic filler) get downweighted while
   distinctive topical words get upweighted. Sentences are scored by graph
   centrality (TextRank/PageRank), plus a position bias that boosts the
   first and last sentences (topic sentences and conclusions). This produces
   summaries focused on main points rather than filler.

   No model download needed, so this keeps the app fully offline and
   language-independent.

2. "abstractive" (optional, requires `transformers` + `torch` + an
   internet connection the first time to download the model):
   Loads facebook/bart-large-cnn (or another seq2seq model) directly via
   AutoTokenizer/AutoModelForSeq2SeqLM and runs generation manually.

   After the model is downloaded once, it is cached locally and works
   offline. If the model isn't cached and the user is offline, it
   gracefully falls back to extractive summarization instead of hanging.

   Reference: https://huggingface.co/docs/transformers/model_doc/bart
"""

import re
import socket
from collections import Counter
from typing import List, Optional

import numpy as np

from .text_processor import split_sentences

# ──────────────────────────────────────────────────────────
# Expanded stopword list — ~250 entries covering:
#   articles/determiners, prepositions, pronouns, conjunctions,
#   auxiliary/modal verbs, common non-topical verbs,
#   common non-topical adjectives/adverbs, speech fillers,
#   and generic nouns that rarely carry topical meaning.
# ──────────────────────────────────────────────────────────
_STOPWORDS = {
    # ── Articles & determiners ──
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "all", "each", "every",
    "both", "few", "more", "most", "other", "another", "such", "no", "not",
    "only", "same", "so", "than", "too", "very", "just",

    # ── Prepositions ──
    "about", "above", "across", "after", "against", "along", "among",
    "around", "at", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "by", "down", "from", "in", "into", "near",
    "of", "off", "on", "out", "over", "past", "through", "to", "under",
    "up", "upon", "with", "within", "without",

    # ── Pronouns ──
    "i", "me", "myself", "we", "us", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "who", "whom", "whose", "which", "what",
    "where", "when", "why", "how",

    # ── Conjunctions & connecting words ──
    "and", "or", "but", "nor", "if", "then", "so", "as", "yet", "although",
    "though", "because", "since", "unless", "until", "while", "whether",
    "either", "neither", "besides", "furthermore", "moreover",
    "nevertheless", "nonetheless", "otherwise", "therefore", "thus",
    "hence", "consequently", "meanwhile", "instead", "despite", "except",
    "rather", "regarding",

    # ── Auxiliary & modal verbs ──
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "need", "dare",
    "ought", "used",

    # ── Common non-topical verbs ──
    "get", "got", "make", "made", "take", "took", "come", "came", "go",
    "going", "gone", "say", "said", "tell", "told", "know", "knew",
    "think", "thought", "see", "saw", "find", "found", "give", "gave",
    "use", "used", "want", "look", "looked", "work", "worked", "call",
    "called", "try", "tried", "ask", "asked", "feel", "felt", "become",
    "became", "leave", "left", "put", "mean", "meant", "keep", "kept",
    "let", "begin", "began", "seem", "seemed", "help", "helped", "show",
    "showed", "hear", "heard", "play", "played", "run", "ran", "move",
    "moved", "live", "lived", "bring", "brought", "happen", "happened",
    "write", "written", "sit", "sat", "stand", "stood", "lose", "lost",
    "pay", "paid", "meet", "met", "include", "included", "continue",
    "continued", "change", "changed", "lead", "led", "understand",
    "understood", "watch", "watched", "follow", "followed", "stop",
    "stopped", "create", "created", "speak", "spoken", "read", "allow",
    "allowed", "add", "added", "spend", "spent", "grow", "grew", "open",
    "opened", "walk", "walked", "win", "won", "offer", "offered",
    "remember", "remembered", "consider", "considered", "appear",
    "appeared", "buy", "bought", "wait", "waited", "serve", "served",
    "die", "died", "send", "sent", "expect", "expected", "build",
    "built", "stay", "stayed", "fall", "fell", "cut", "reach", "reached",
    "remain", "remained", "suggest", "suggested", "raise", "raised",
    "pass", "passed", "sell", "sold", "require", "required", "report",
    "reported", "decide", "decided", "pull", "pulled", "set", "setting",
    "turn", "turned",

    # ── Common non-topical adjectives/adverbs ──
    "also", "really", "often", "however", "usually", "already", "always",
    "sometimes", "never", "soon", "still", "ever", "again", "much", "many",
    "less", "least", "better", "best", "well", "bad", "different",
    "important", "possible", "necessary", "able", "available", "likely",
    "actual", "basic", "clear", "close", "common", "complete", "current",
    "easy", "enough", "fair", "free", "full", "good", "great", "hard",
    "high", "low", "major", "new", "old", "real", "recent", "right",
    "small", "special", "strong", "sure", "true", "whole", "long", "short",
    "big", "little", "first", "last", "next", "previous", "own", "main",
    "simple", "single", "general", "specific", "natural", "normal",
    "typical", "certain", "obvious", "essential", "significant", "relevant",
    "effective", "successful", "positive", "negative", "initial", "final",
    "primary", "overall", "rather", "quite", "pretty", "fairly",
    "somewhat", "maybe", "perhaps", "probably", "definitely", "certainly",
    "absolutely", "totally", "completely", "exactly",

    # ── Generic nouns (too broad to carry topical meaning) ──
    "thing", "things", "something", "anything", "nothing", "everything",
    "part", "point", "fact", "reason", "way", "case", "area", "side",
    "end", "place", "person", "people", "group", "number", "world",
    "life", "year", "day", "week", "month", "hand", "water", "home",
    "room", "word", "idea", "kind", "sort", "class", "category",

    # ── Speech fillers & discourse markers ──
    "like", "well", "okay", "yeah", "yes", "no", "um", "uh", "hmm",
    "actually", "basically", "literally", "seriously", "honestly",
    "essentially", "simply", "merely", "here", "there", "now", "then",

    # ── Gerund/progressive forms of common verbs ──
    "getting", "doing", "having", "being", "looking", "talking", "trying",
    "using", "working", "making", "taking", "coming", "giving", "keeping",
    "putting", "running", "turning", "moving", "living", "showing",
    "feeling", "leaving", "calling", "asking", "telling", "knowing",
    "thinking", "seeing", "finding", "wanting", "saying",

    # ── Negation contractions (stem form) ──
    "don", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
    "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
}


def _is_online(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    """
    Quick connectivity check — attempts a TCP connection to the given host.
    Returns True if the connection succeeds within the timeout, False
    otherwise. Used to avoid a long hang when the abstractive model isn't
    cached locally and the user is offline.
    """
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (OSError, socket.timeout):
        return False


def _compute_idf(sentences: List[str], vocab_index: dict) -> np.ndarray:
    """
    Compute smoothed inverse document frequency (IDF) for each vocabulary
    word across all candidate sentences.

    IDF = log((N+1) / (df+1)) + 1

    Words appearing in many sentences get low IDF (too common to be
    discriminative), while words appearing in few sentences get high IDF
    (distinctive, likely topical). This is the key improvement over raw
    word counts — filler words like "also", "really", "said" that slip
    past the stopword list still get downweighted because they appear in
    almost every sentence.
    """
    n = len(sentences)
    doc_freq = np.zeros(len(vocab_index), dtype=np.float32)

    for sent in sentences:
        unique_words = set(
            w for w in re.findall(r"\w+", sent.lower())
            if w not in _STOPWORDS
        )
        for w in unique_words:
            if w in vocab_index:
                doc_freq[vocab_index[w]] += 1.0

    # Smoothed IDF: log((N+1)/(df+1)) + 1  (the +1 prevents zero IDF
    # and ensures even universally-appearing words still have a small weight)
    idf = np.log((n + 1.0) / (doc_freq + 1.0)) + 1.0
    return idf


def _sentence_vector(sentence: str, vocab_index: dict, idf: np.ndarray) -> np.ndarray:
    """
    TF-IDF weighted vector for one sentence, stopwords excluded.

    Each word's contribution is its term frequency (count in this sentence)
    multiplied by its IDF weight. This means:
    - A word that appears in this sentence but rarely in other sentences
      (high IDF) contributes strongly — it's distinctive and topical.
    - A word that appears in this sentence AND in most other sentences
      (low IDF) contributes weakly — it's generic filler.

    The vector is L2-normalized so cosine similarity works correctly.
    """
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    words = [w for w in re.findall(r"\w+", sentence.lower()) if w not in _STOPWORDS]
    for w in words:
        if w in vocab_index:
            vec[vocab_index[w]] += idf[vocab_index[w]]  # TF * IDF
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _textrank_scores(
    sentences: List[str],
    damping: float = 0.85,
    iterations: int = 30,
) -> np.ndarray:
    """
    Score sentences by graph centrality (TextRank) using TF-IDF-weighted
    similarity vectors.

    Build a similarity matrix between all sentence pairs (cosine similarity
    over TF-IDF vectors), then run PageRank-style iteration so a sentence's
    score reflects how strongly it connects to the rest of the transcript
    via distinctive topical words — not via shared filler.
    """
    n = len(sentences)
    vocab = sorted({
        w for s in sentences
        for w in re.findall(r"\w+", s.lower())
        if w not in _STOPWORDS
    })
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # Compute IDF weights across all sentences
    idf = _compute_idf(sentences, vocab_index)

    # Build TF-IDF weighted sentence vectors
    vectors = np.stack([_sentence_vector(s, vocab_index, idf) for s in sentences])

    # Cosine similarity matrix (vectors are already L2-normalized)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, 0.0)

    # Build transition matrix for PageRank iteration
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    scores = np.full(n, 1.0 / n, dtype=np.float32)
    for _ in range(iterations):
        scores = (1 - damping) / n + damping * (transition.T @ scores)

    return scores


def summarize_extractive(text: str, num_sentences: int = 5) -> str:
    """
    Graph-based (TextRank) extractive summarization with TF-IDF weighting
    and position bias. No external model needed — works offline for any
    language.
    """
    sentences = split_sentences(text)

    # Filter out filler/disfluency fragments. Require at least 4 content
    # words (non-stopwords), not just 5 total words — a 6-word sentence
    # where 5 are stopwords is not content-bearing.
    candidates = [
        s for s in sentences
        if len([w for w in re.findall(r"\w+", s.lower()) if w not in _STOPWORDS]) >= 4
    ]
    if not candidates:
        # Relaxed fallback: accept sentences with >= 3 total words
        candidates = [s for s in sentences if len(re.findall(r"\w+", s)) >= 3]
    if not candidates:
        candidates = sentences

    if len(candidates) <= 2:
        return " ".join(candidates)

    target = max(1, min(num_sentences, round(len(candidates) * 0.4)))

    # TextRank scores (now TF-IDF weighted)
    scores = _textrank_scores(candidates)

    # ── Position bias ──
    # In lectures/meetings/articles, the first sentence is usually the
    # topic sentence (states the main point), and the last sentence is
    # often a conclusion/summary. Give them a boost so they're more
    # likely to appear in the final summary.
    position_bias = np.zeros(len(candidates), dtype=np.float32)
    for i in range(len(candidates)):
        if i == 0:
            position_bias[i] = 0.20   # first sentence: strong boost
        elif i == 1:
            position_bias[i] = 0.10
        elif i == 2:
            position_bias[i] = 0.05
        if i >= len(candidates) - 1 and len(candidates) > 3:
            position_bias[i] += 0.05   # last sentence: small boost

    final_scores = scores + position_bias

    # Select top-scoring sentences, preserving their original order
    ranked = sorted(range(len(candidates)), key=lambda i: -final_scores[i])[:target]
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

    Model loading strategy:
      1. Try loading from local cache first (local_files_only=True).
         This works offline if the model was downloaded in a previous
         session.
      2. If not cached, do a quick connectivity check before attempting
         to download — avoids a 30-60 second hang on a failed connection
         when the user is offline.
      3. If online, download normally (first time only; cached after).
    """
    global _ABSTRACTIVE_MODEL, _ABSTRACTIVE_TOKENIZER, _ABSTRACTIVE_MODEL_NAME

    if _ABSTRACTIVE_MODEL is None or _ABSTRACTIVE_MODEL_NAME != model_name:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        # Step 1: Try loading from local cache — works offline if the
        # model was previously downloaded.
        try:
            _ABSTRACTIVE_TOKENIZER = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            _ABSTRACTIVE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, local_files_only=True
            )
        except (OSError, ValueError):
            # Step 2: Model not in local cache. Check connectivity before
            # attempting a download so we don't hang for 30-60 seconds.
            if not _is_online():
                raise RuntimeError(
                    f"Abstractive model '{model_name}' is not cached locally "
                    "and you appear to be offline. Either switch to Extractive "
                    "summarization, or connect to the internet once to download "
                    "the model (after that it works offline)."
                )
            # Step 3: Online — download the model.
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
            return (
                f"⚠️ Abstractive summarizer unavailable ({exc}). "
                f"Showing extractive summary instead:\n\n{fallback}"
            )
    return summarize_extractive(text, num_sentences)
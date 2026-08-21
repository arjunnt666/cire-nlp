"""
CIRE — Contextual Intent Resolution Engine
============================================
Version: 1.0.0
License: MIT

A zero-dependency, domain-agnostic NLP framework for constrained
conversational agents. Swap DOMAIN_SCHEMA with your own topic data.
"""

import re
import random
import logging
import hashlib
import time
import sys
import json
from difflib import SequenceMatcher
from functools import lru_cache
from collections import defaultdict, Counter
from typing import Optional

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger("cire")


# ══════════════════════════════════════════════════════════════════════════════
#  CIRE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CIRE_VERSION = "1.0.0"
CIRE_BUILD   = "stable"

# Fuzzy match threshold — raise for stricter matching, lower for more tolerance
FSM_THRESHOLD          = 0.80

# Domain Recency Weighting bonus applied to last-resolved domain on ties
DRW_TIEBREAK_BONUS     = 2

# Minimum word length to qualify for fuzzy matching (avoids noise on short words)
FSM_MIN_WORD_LENGTH    = 4

# Maximum response text length (characters)
MAX_RESPONSE_LENGTH    = 1024

# Intent class priority order (higher index = evaluated first)
INTENT_PRIORITY = [
    "info", "rules", "tip", "advanced_tip",
    "edge", "mistakes", "variants", "ranking"
]


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN SCHEMA
#  ─────────────────────────────────────────────────────────────────────────────
#  Replace the entries below with your own domain data.
#  Each domain key maps to a dict with:
#    keywords    : list[str]  — primary surface forms
#    synonyms    : list[str]  — alternate surface forms (lower weight)
#    info        : str        — general overview (default intent)
#    rules       : str        — procedural / definitional content
#    tip         : str        — actionable recommendations
#    advanced_tip: str        — expert-level content
#    edge        : str        — statistical / analytical content
#    mistakes    : str        — common failure patterns
#    variants    : str        — sub-topic taxonomy
#    ranking     : str        — ordered comparison content  (optional)
#    emoji       : str        — display emoji               (optional)
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_SCHEMA: dict[str, dict] = {

    # ── EXAMPLE DOMAIN A ─────────────────────────────────────────────────────
    "domain_a": {
        "keywords":     ["topic a", "subject a", "area a"],
        "synonyms":     ["a topic", "a subject"],
        "info":         "General overview of domain A.",
        "rules":        "How domain A works procedurally.",
        "tip":          "Actionable advice for domain A.",
        "advanced_tip": "Expert-level insight for domain A.",
        "edge":         "Statistical context for domain A.",
        "mistakes":     "Common errors in domain A.",
        "variants":     "Sub-types within domain A.",
        "emoji":        "🔵"
    },

    # ── EXAMPLE DOMAIN B ─────────────────────────────────────────────────────
    "domain_b": {
        "keywords":     ["topic b", "subject b", "area b"],
        "synonyms":     ["b topic", "b subject"],
        "info":         "General overview of domain B.",
        "rules":        "How domain B works procedurally.",
        "tip":          "Actionable advice for domain B.",
        "advanced_tip": "Expert-level insight for domain B.",
        "edge":         "Statistical context for domain B.",
        "mistakes":     "Common errors in domain B.",
        "variants":     "Sub-types within domain B.",
        "emoji":        "🟢"
    },

}


# ══════════════════════════════════════════════════════════════════════════════
#  STATIC CONTENT — replace with your own strings
# ══════════════════════════════════════════════════════════════════════════════

WELCOME_TEXT = (
    "Welcome. I can help you with: {topics}. "
    "Say a topic name to get started. "
    "Say 'what can you do' for a full overview."
)

HELP_TEXT = (
    "Available topics: {topics}. "
    "For each topic you can ask: info, rules, tip, strategy, mistakes, or variants. "
    "Example: 'rules for topic a' or 'mistakes in topic b'."
)

CAPABILITIES_TEXT = (
    "I am a CIRE-powered conversational agent. "
    "I can answer questions across {count} domains. "
    "Say a topic name to begin, or say 'help' for a command list."
)

FALLBACK_TEXT = (
    "I didn't quite catch that. "
    "Try saying a topic name directly, or say 'what can you do'."
)

GOODBYE_PHRASES = [
    "Goodbye. Come back anytime.",
    "See you later.",
    "Bye. Stay curious.",
    "Until next time."
]

RANDOM_TIPS = [
    "Start with the basics before diving into advanced content.",
    "Asking about mistakes in a topic is often the fastest way to learn.",
    "Use the 'variants' command to discover sub-topics you didn't know existed.",
    "The 'edge' command gives you the statistical picture for any topic.",
    "Combine 'advanced tip' with 'mistakes' for a complete picture."
]

RANDOM_FACTS = [
    "CIRE resolves most intents in under 2 milliseconds with no ML model required.",
    "The fuzzy surface matcher operates at word level, not character level, reducing false positives.",
    "Domain Recency Weighting enables natural conversational flow without explicit topic re-declaration.",
    "The Contextual Button Coherence System verifies every suggested action before showing it.",
    "CIRE's Intent Stratification Layer evaluates 8 intent classes in strict priority order."
]


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 0 — LEXICAL SURFACE NORMALIZATION (LSN)
# ══════════════════════════════════════════════════════════════════════════════

# Homoglyph substitution table — maps visually similar chars to canonical form
_HOMOGLYPH_TABLE: dict[str, str] = {
    "\u0430": "a",  # Cyrillic а → Latin a
    "\u0435": "e",  # Cyrillic е → Latin e
    "\u043e": "o",  # Cyrillic о → Latin o
    "\u0440": "p",  # Cyrillic р → Latin p
    "\u0441": "c",  # Cyrillic с → Latin c
    "\u0445": "x",  # Cyrillic х → Latin x
}

# Contraction expansion table
_CONTRACTIONS: dict[str, str] = {
    "what's":  "what is",
    "don't":   "do not",
    "doesn't": "does not",
    "i'm":     "i am",
    "can't":   "cannot",
    "won't":   "will not",
    "isn't":   "is not",
}


def _apply_homoglyphs(text: str) -> str:
    """Substitute known homoglyphs to canonical Latin form."""
    return "".join(_HOMOGLYPH_TABLE.get(ch, ch) for ch in text)


def _expand_contractions(text: str) -> str:
    """Expand English contractions to full forms."""
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def normalize(text: str) -> str:
    """
    LSN — Lexical Surface Normalization.
    Produces a canonical surface form from raw user input.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = _expand_contractions(text)
    text = _apply_homoglyphs(text)
    text = re.sub(r"[^\w\s]", " ", text)      # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def _bounded(haystack: str, needle: str) -> bool:
    """True if needle appears as a whole word/phrase in haystack."""
    if not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", haystack) is not None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 0 — META INTENT RESOLVER (MIR)
# ══════════════════════════════════════════════════════════════════════════════

# Multi-surface trigger sets for each meta intent
_MIR_TRIGGERS: dict[str, list[str]] = {
    "what_can_you_do": [
        "what can you do", "what do you do", "what are you",
        "what can you help with", "capabilities", "your features",
        "how do i use this", "show me options", "what do you know",
        "how does this work", "tell me about yourself", "what is this"
    ],
    "help": [
        "help", "help me", "i need help", "i am lost", "what do i say",
        "command list", "list commands", "what commands", "what can i say",
        "how do i", "instructions", "guide", "show commands"
    ],
    "goodbye": [
        "goodbye", "bye", "quit", "exit", "stop", "done", "close",
        "i am done", "that is all", "end", "finish", "farewell"
    ],
    "fact": [
        "fact", "give me a fact", "random fact", "interesting",
        "tell me something", "surprise me", "did you know"
    ],
    "quick_tip": [
        "tip", "quick tip", "random tip", "give me a tip",
        "any advice", "advice", "suggestion"
    ]
}

# Pre-compute flattened lookup for O(1) exact match
_MIR_EXACT: dict[str, str] = {
    phrase: intent
    for intent, phrases in _MIR_TRIGGERS.items()
    for phrase in phrases
}


def detect_meta_intent(text: str) -> Optional[str]:
    """
    MIR — Meta Intent Resolver (Layer 0).
    Intercepts global commands before domain resolution.
    Returns meta intent string or None if no match.
    """
    text = normalize(text)

    # Exact lookup first — O(1)
    if text in _MIR_EXACT:
        return _MIR_EXACT[text]

    # Phrase scan with word boundaries so "tip" does not fire inside "topic"
    for phrase, intent in _MIR_EXACT.items():
        if _bounded(text, phrase):
            return intent

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — DOMAIN TOPIC SCORER (DTS)
# ══════════════════════════════════════════════════════════════════════════════

# Weight constants for DTS scoring
_WEIGHT_EXACT_KEYWORD  = 10
_WEIGHT_EXACT_SYNONYM  = 8
_WEIGHT_FUZZY_WORD     = 5
_WEIGHT_PREFIX         = 3


@lru_cache(maxsize=512)
def _fuzzy_pair(word: str, keyword: str) -> float:
    """Cached pairwise fuzzy ratio between two strings."""
    return SequenceMatcher(None, word, keyword).ratio()


def fuzzy_match(word: str, keyword: str, threshold: float = FSM_THRESHOLD) -> bool:
    """
    FSM — Fuzzy Surface Matching.
    Word-level edit-distance check with configurable threshold.
    """
    if word == keyword:
        return True
    if len(word) < FSM_MIN_WORD_LENGTH or len(keyword) < FSM_MIN_WORD_LENGTH:
        return False
    return _fuzzy_pair(word, keyword) >= threshold


def _score_domain(text_norm: str, words: list[str], data: dict) -> int:
    """Compute DTS score for a single domain against the normalized utterance."""
    score = 0
    keywords = data.get("keywords", [])
    synonyms = data.get("synonyms", [])

    for kw in keywords:
        kw_norm = normalize(kw)
        if _bounded(text_norm, kw_norm):
            score += _WEIGHT_EXACT_KEYWORD
            continue
        if kw_norm and words[0][:len(kw_norm)] == kw_norm[:len(words[0])]:
            score += _WEIGHT_PREFIX
            continue
        for word in words:
            if len(word) >= FSM_MIN_WORD_LENGTH and fuzzy_match(word, kw_norm):
                score += _WEIGHT_FUZZY_WORD
                break

    for syn in synonyms:
        syn_norm = normalize(syn)
        if _bounded(text_norm, syn_norm):
            score += _WEIGHT_EXACT_SYNONYM
            continue
        for word in words:
            if len(word) >= FSM_MIN_WORD_LENGTH and fuzzy_match(word, syn_norm):
                score += _WEIGHT_FUZZY_WORD
                break

    return score


def detect_topic(text: str, last_topic: Optional[str] = None) -> Optional[str]:
    """
    DTS — Domain Topic Scorer (Layer 1).
    Scores all registered domains and returns the best match.
    Applies Domain Recency Weighting (DRW) on ties.
    """
    text_norm = normalize(text)
    words = text_norm.split()

    scores: dict[str, int] = {}

    for domain, data in DOMAIN_SCHEMA.items():
        s = _score_domain(text_norm, words, data)
        if s > 0:
            scores[domain] = s

    if not scores:
        return None

    max_score = max(scores.values())
    top_domains = [d for d, s in scores.items() if s == max_score]

    # Apply DRW tiebreaker
    if len(top_domains) > 1 and last_topic in top_domains:
        return last_topic

    return top_domains[0]


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — INTENT STRATIFICATION LAYER (ISL)
# ══════════════════════════════════════════════════════════════════════════════

# Intent trigger map — evaluated in INTENT_PRIORITY order
_INTENT_TRIGGERS: dict[str, list[str]] = {
    "rules":        ["rules", "how to", "how do i", "explain", "what is", "how does", "what are"],
    "tip":          ["tip", "strategy", "advice", "best way", "how to win", "recommend", "suggest"],
    "advanced_tip": ["advanced", "expert", "professional", "deep dive", "detailed", "in depth", "pro tip"],
    "edge":         ["probability", "percent", "math", "statistics", "odds", "house edge", "expected value", "ev"],
    "mistakes":     ["mistake", "error", "avoid", "wrong", "bad", "common mistake", "do not", "don't"],
    "variants":     ["types", "kinds", "versions", "variants", "variations", "different kinds", "what kinds"],
    "ranking":      ["best", "worst", "compare", "rank", "ranking", "top", "vs", "versus", "which is better"],
}


def detect_intent(text: str) -> str:
    """
    ISL — Intent Stratification Layer (Layer 2).
    Classifies the utterance into one of 8 intent classes.
    Returns 'info' as default if no triggers match.
    """
    text_norm = normalize(text)

    for intent in reversed(INTENT_PRIORITY):   # higher priority checked last, wins
        triggers = _INTENT_TRIGGERS.get(intent, [])
        for trigger in triggers:
            if _bounded(text_norm, trigger):
                return intent

    return "info"


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — CONTEXTUAL FALLBACK SYNTHESIZER (CFS)
# ══════════════════════════════════════════════════════════════════════════════

def build_fallback(last_topic: Optional[str] = None) -> str:
    """
    CFS — Contextual Fallback Synthesizer (Layer 3).
    Generates context-aware fallback anchored to last resolved domain.
    """
    if last_topic and last_topic in DOMAIN_SCHEMA:
        topic_name = last_topic.replace("_", " ").title()
        return (
            f"I didn't catch that. We were talking about {topic_name}. "
            f"You can ask about its rules, tips, mistakes, or variants. "
            f"Or say 'what can you do' to see all topics."
        )
    return FALLBACK_TEXT


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXTUAL BUTTON COHERENCE SYSTEM (CBCS)
# ══════════════════════════════════════════════════════════════════════════════

def verify_button(label: str) -> bool:
    """
    CBCS verification — returns True if the label resolves via LRP.
    Used to filter suggestion sets before showing them to the user.
    """
    meta = detect_meta_intent(label)
    if meta:
        return True
    topic = detect_topic(label)
    return topic is not None


def filter_buttons(candidates: list[str]) -> list[str]:
    """
    CBCS — Contextual Button Coherence System.
    Filters candidate button labels to only those verifiably resolvable.
    """
    return [label for label in candidates if verify_button(label)]


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _get_follow_up(intent: str) -> str:
    """Return a contextual follow-up prompt based on current intent."""
    follow_ups = {
        "info":         "Want to hear tips or rules?",
        "rules":        "Want strategy advice for this topic?",
        "tip":          "Want to know common mistakes to avoid?",
        "advanced_tip": "Want the statistical breakdown?",
        "edge":         "Want to hear the strategy to improve your odds?",
        "mistakes":     "Want advanced tips now?",
        "variants":     "Want rules or tips for a specific variant?",
        "ranking":      "Want detailed tips on the top option?",
    }
    return follow_ups.get(intent, "")


def build_response(
    user_text: str,
    session_state: Optional[dict] = None
) -> tuple[str, bool, Optional[str]]:
    """
    Full LRP execution — Layered Resolution Pipeline.

    Args:
        user_text:     Raw utterance from user.
        session_state: Optional dict with keys: last_topic, message_count.

    Returns:
        (response_text, end_session, resolved_topic)
    """
    state        = session_state or {}
    last_topic   = state.get("last_topic")
    text         = normalize(user_text)

    # ── Layer 0: Meta Intent Resolver ─────────────────────────────────────
    meta = detect_meta_intent(text)

    if meta == "what_can_you_do":
        count = len(DOMAIN_SCHEMA)
        return CAPABILITIES_TEXT.format(count=count), False, last_topic

    if meta == "help":
        topics = ", ".join(k.replace("_", " ") for k in DOMAIN_SCHEMA)
        return HELP_TEXT.format(topics=topics), False, last_topic

    if meta == "goodbye":
        return random.choice(GOODBYE_PHRASES), True, None

    if meta == "fact":
        return random.choice(RANDOM_FACTS), False, last_topic

    if meta == "quick_tip":
        return random.choice(RANDOM_TIPS), False, last_topic

    # ── Layer 1: Domain Topic Scorer ──────────────────────────────────────
    topic = detect_topic(text, last_topic=last_topic)

    # ── Layer 2: Intent Stratifier ────────────────────────────────────────
    intent = detect_intent(text)

    if topic:
        data    = DOMAIN_SCHEMA[topic]
        emoji   = data.get("emoji", "")
        content = data.get(intent) or data.get("info") or ""

        # Truncate if needed
        if len(content) > MAX_RESPONSE_LENGTH:
            content = content[:MAX_RESPONSE_LENGTH - 3] + "..."

        follow_up = _get_follow_up(intent)
        response  = f"{emoji} {content}".strip()
        if follow_up:
            response += f" {follow_up}"

        return response, False, topic

    # ── Layer 3: Contextual Fallback Synthesizer ──────────────────────────
    return build_fallback(last_topic), False, last_topic


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class CIRESession:
    """
    Lightweight session state manager.
    Tracks resolved topic, message count, and resolution history.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id    = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.last_topic    = None
        self.message_count = 0
        self.history: list[dict] = []
        self._domain_freq  = Counter()

    def process(self, user_text: str) -> tuple[str, bool]:
        """Process an utterance and update session state."""
        self.message_count += 1

        response, end, resolved_topic = build_response(
            user_text,
            session_state={"last_topic": self.last_topic}
        )

        if resolved_topic:
            self.last_topic = resolved_topic
            self._domain_freq[resolved_topic] += 1

        self.history.append({
            "turn":    self.message_count,
            "input":   user_text,
            "topic":   resolved_topic,
            "intent":  detect_intent(normalize(user_text)),
            "output":  response,
        })

        logger.debug(
            f"[{self.session_id}] turn={self.message_count} "
            f"topic={resolved_topic} end={end}"
        )

        return response, end

    def top_domains(self, n: int = 3) -> list[tuple[str, int]]:
        """Return the n most frequently resolved domains in this session."""
        return self._domain_freq.most_common(n)

    def reset(self) -> None:
        """Reset session state while preserving session ID."""
        self.last_topic    = None
        self.message_count = 0
        self.history.clear()
        self._domain_freq.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  CIRE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

class CIREDiagnostics:
    """
    Runtime diagnostics and introspection utilities.
    Useful during development and domain schema authoring.
    """

    @staticmethod
    def score_all_domains(text: str) -> dict[str, int]:
        """Return DTS scores for all domains against the given utterance."""
        text_norm = normalize(text)
        words     = text_norm.split()
        return {
            domain: _score_domain(text_norm, words, data)
            for domain, data in DOMAIN_SCHEMA.items()
        }

    @staticmethod
    def explain_resolution(text: str) -> dict:
        """Full resolution trace for a given utterance."""
        text_norm = normalize(text)
        meta      = detect_meta_intent(text_norm)
        topic     = detect_topic(text_norm)
        intent    = detect_intent(text_norm)
        scores    = CIREDiagnostics.score_all_domains(text_norm)
        return {
            "raw_input":     text,
            "normalized":    text_norm,
            "meta_intent":   meta,
            "resolved_topic": topic,
            "intent_class":  intent,
            "domain_scores": scores,
        }

    @staticmethod
    def validate_schema() -> list[str]:
        """
        Validate all domain schemas.
        Returns list of warnings for missing or empty required fields.
        """
        required = ["keywords", "info"]
        recommended = ["tip", "rules", "mistakes"]
        warnings = []

        for domain, data in DOMAIN_SCHEMA.items():
            for field in required:
                if field not in data or not data[field]:
                    warnings.append(f"[{domain}] Missing required field: '{field}'")
            for field in recommended:
                if field not in data or not data[field]:
                    warnings.append(f"[{domain}] Missing recommended field: '{field}'")

        return warnings

    @staticmethod
    def verify_all_buttons(button_sets: list[list[str]]) -> dict[str, bool]:
        """Run CBCS verification on a flat list of button label sets."""
        results = {}
        for btn_set in button_sets:
            for label in btn_set:
                results[label] = verify_button(label)
        return results

    @staticmethod
    def resolution_stats(utterances: list[str]) -> dict:
        """Batch resolution stats over a list of test utterances."""
        resolved   = 0
        meta_hits  = 0
        fallbacks  = 0
        intent_dist = Counter()

        for utt in utterances:
            meta  = detect_meta_intent(utt)
            topic = detect_topic(utt)
            intent = detect_intent(utt)
            intent_dist[intent] += 1

            if meta:
                meta_hits += 1
            elif topic:
                resolved += 1
            else:
                fallbacks += 1

        total = len(utterances)
        return {
            "total":          total,
            "meta_hits":      meta_hits,
            "domain_resolved": resolved,
            "fallbacks":      fallbacks,
            "resolution_rate": round((resolved + meta_hits) / total * 100, 1) if total else 0,
            "intent_distribution": dict(intent_dist),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CIRE REGISTRY — Dynamic domain registration at runtime
# ══════════════════════════════════════════════════════════════════════════════

class CIRERegistry:
    """
    Runtime domain registry.
    Allows registering, updating, and removing domains without restart.
    """

    _domains: dict[str, dict] = {}

    @classmethod
    def register(cls, domain_key: str, schema: dict) -> None:
        """Register or overwrite a domain schema."""
        if "keywords" not in schema or not schema["keywords"]:
            raise ValueError(f"Domain '{domain_key}' must have at least one keyword.")
        cls._domains[domain_key] = schema
        DOMAIN_SCHEMA[domain_key] = schema
        _fuzzy_pair.cache_clear()
        logger.info(f"[Registry] Registered domain: {domain_key}")

    @classmethod
    def unregister(cls, domain_key: str) -> None:
        """Remove a domain from the registry."""
        cls._domains.pop(domain_key, None)
        DOMAIN_SCHEMA.pop(domain_key, None)
        _fuzzy_pair.cache_clear()
        logger.info(f"[Registry] Unregistered domain: {domain_key}")

    @classmethod
    def list_domains(cls) -> list[str]:
        """Return list of all registered domain keys."""
        return list(DOMAIN_SCHEMA.keys())

    @classmethod
    def get_schema(cls, domain_key: str) -> Optional[dict]:
        """Return the schema for a specific domain."""
        return DOMAIN_SCHEMA.get(domain_key)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_welcome() -> str:
    """Return the welcome message with current topic list."""
    topics = ", ".join(k.replace("_", " ") for k in DOMAIN_SCHEMA)
    return WELCOME_TEXT.format(topics=topics)


def get_help() -> str:
    """Return the full help text with current topic list."""
    topics = ", ".join(k.replace("_", " ") for k in DOMAIN_SCHEMA)
    return HELP_TEXT.format(topics=topics)


def resolve(
    text: str,
    last_topic: Optional[str] = None
) -> dict:
    """
    Top-level resolution entry point.

    Args:
        text:       Raw user utterance.
        last_topic: Last resolved domain key for DRW (optional).

    Returns:
        dict with keys: response, end_session, topic, intent, meta
    """
    text_norm = normalize(text)
    meta      = detect_meta_intent(text_norm)
    topic     = detect_topic(text_norm, last_topic=last_topic)
    intent    = detect_intent(text_norm)

    response, end, resolved = build_response(
        text,
        session_state={"last_topic": last_topic}
    )

    return {
        "response":    response,
        "end_session": end,
        "topic":       resolved,
        "intent":      intent,
        "meta":        meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_smoke() -> None:
    print(f"CIRE v{CIRE_VERSION} ({CIRE_BUILD}) — smoke test")
    print("-" * 60)

    diag = CIREDiagnostics()
    warnings = diag.validate_schema()
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  Schema validation passed")

    test_phrases = [
        "what can you do",
        "help",
        "goodbye",
        "topic a",
        "rules for topic b",
        "mistakes in topic a",
        "xyz unknown phrase",
    ]

    print()
    for phrase in test_phrases:
        result = resolve(phrase)
        print(f"  '{phrase}'")
        print(f"    meta={result['meta']} topic={result['topic']} intent={result['intent']}")
        print(f"    → {result['response'][:80]}...")
        print()


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args == ["--smoke"]:
        run_smoke()
        return 0
    if args[0] in ("-h", "--help"):
        print("usage: python cire_engine.py [utterance]")
        print("       python cire_engine.py --smoke")
        return 0
    result = resolve(" ".join(args))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

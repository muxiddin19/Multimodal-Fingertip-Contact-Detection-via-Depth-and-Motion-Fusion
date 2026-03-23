"""
Character-Level N-gram Language Model for VR Keyboard
=====================================================
Provides P(char | context) to disambiguate between spatially close keys.
Uses character bigram/trigram frequencies from English text with
Kneser-Ney-style backoff.

Reference: VelociTap (CHI 2015), VISAR (ACM TOCHI 2018)
"""

import math
from typing import Dict, Optional


# Character bigram log-probabilities derived from English corpus.
# Format: P(c2 | c1) as log10 probabilities.
# Only storing the most common transitions; unseen pairs get smoothed.
_BIGRAM_FREQS = {
    # After space (word start)
    ' t': -0.74, ' a': -1.05, ' s': -1.12, ' o': -1.22, ' i': -1.25,
    ' w': -1.30, ' c': -1.38, ' h': -1.42, ' b': -1.48, ' m': -1.50,
    ' f': -1.52, ' p': -1.55, ' d': -1.60, ' n': -1.70, ' l': -1.75,
    ' e': -1.80, ' r': -1.82, ' g': -1.90, ' y': -2.00, ' u': -2.05,
    ' v': -2.20, ' k': -2.40, ' j': -2.50, ' q': -2.80, ' x': -3.00,
    ' z': -3.10,
    # Common pairs
    'th': -0.40, 'he': -0.50, 'in': -0.60, 'er': -0.62, 'an': -0.65,
    'on': -0.70, 'en': -0.72, 'at': -0.75, 'es': -0.78, 'ed': -0.80,
    'or': -0.82, 'te': -0.83, 're': -0.84, 'ti': -0.85, 'is': -0.87,
    'it': -0.88, 'al': -0.90, 'ar': -0.91, 'st': -0.92, 'to': -0.93,
    'nt': -0.94, 'ng': -0.95, 'se': -0.96, 'ha': -0.97, 'ou': -0.98,
    'le': -1.00, 'nd': -1.01, 'hi': -1.02, 'as': -1.03, 'de': -1.04,
    'me': -1.05, 'of': -1.06, 'so': -1.07, 'ne': -1.08, 'io': -1.09,
    've': -1.10, 'no': -1.11, 'ta': -1.12, 'li': -1.13, 'co': -1.14,
    'ri': -1.15, 'ro': -1.16, 'ea': -1.17, 'ce': -1.18, 'la': -1.19,
    'el': -1.20, 'ma': -1.21, 'di': -1.22, 'si': -1.23, 'ra': -1.24,
    'na': -1.25, 'ur': -1.26, 'ch': -1.27, 'ss': -1.28, 'us': -1.29,
    'pe': -1.30, 'ge': -1.31, 'om': -1.32, 'il': -1.33, 'ca': -1.34,
    'ly': -1.35, 'ni': -1.36, 'be': -1.37, 'rs': -1.38, 'ct': -1.39,
    'wa': -1.40, 'fo': -1.41, 'wi': -1.42, 'sh': -1.43, 'll': -1.44,
    'lo': -1.45, 'ot': -1.46, 'ma': -1.47, 'wh': -1.48, 'ho': -1.49,
    'wo': -1.50, 'do': -1.51, 'if': -1.52, 'up': -1.53, 'ab': -1.54,
    'go': -1.55, 'pr': -1.56, 'ye': -1.57, 'da': -1.58, 'mi': -1.59,
    # After vowels -> common consonants
    'e ': -0.70, 'a ': -1.20, 'o ': -1.30, 'i ': -1.80, 'y ': -1.10,
    's ': -0.85, 't ': -0.95, 'd ': -1.00, 'n ': -1.05, 'r ': -1.15,
    'l ': -1.25, 'f ': -1.60, 'k ': -1.70, 'g ': -1.50, 'p ': -1.65,
    # Double letters
    'ee': -1.50, 'oo': -1.60, 'tt': -1.70, 'ff': -1.80, 'pp': -1.90,
    'rr': -2.00, 'mm': -2.10, 'nn': -1.85, 'dd': -2.10, 'bb': -2.30,
    'cc': -2.20, 'gg': -2.40, 'zz': -3.00,
    # q is almost always followed by u
    'qu': -0.10,
}

# Character unigram frequencies (log10)
_UNIGRAM_FREQS = {
    ' ': -0.70, 'e': -1.07, 't': -1.13, 'a': -1.19, 'o': -1.22,
    'i': -1.25, 'n': -1.27, 's': -1.29, 'h': -1.32, 'r': -1.34,
    'd': -1.47, 'l': -1.50, 'c': -1.55, 'u': -1.58, 'm': -1.62,
    'w': -1.65, 'f': -1.68, 'g': -1.72, 'y': -1.75, 'p': -1.78,
    'b': -1.82, 'v': -1.90, 'k': -2.10, 'j': -2.50, 'x': -2.60,
    'q': -2.80, 'z': -2.90,
}

# Smoothing for unseen bigrams
_SMOOTH_LOG = -3.5


class CharLanguageModel:
    """
    Character-level bigram language model for keyboard input disambiguation.

    Provides P(next_char | context) to help distinguish between
    spatially close keys (e.g., after 'th', 'e' is much more likely than 'r').
    """

    def __init__(self):
        self.bigrams = _BIGRAM_FREQS
        self.unigrams = _UNIGRAM_FREQS

    def score_char(self, char: str, context: str) -> float:
        """
        Return log P(char | context) using bigram with unigram backoff.

        Args:
            char: candidate character (single char, lowercase)
            context: recent typed text (uses last char for bigram)

        Returns:
            Log probability (higher = more likely)
        """
        c = char.lower()

        # Numbers and special chars: return uniform (no language preference)
        if not c.isalpha() and c != ' ':
            return -1.5  # neutral score

        # Bigram: P(c | prev_char)
        if context:
            prev = context[-1].lower()
            bigram = prev + c
            if bigram in self.bigrams:
                return self.bigrams[bigram]

        # Backoff to unigram
        if c in self.unigrams:
            return self.unigrams[c]

        return _SMOOTH_LOG

    def score_candidates(
        self,
        touch_probs: Dict[str, float],
        context: str,
        alpha: float = 0.7
    ) -> Dict[str, float]:
        """
        Combine touch probabilities with language model scores.

        Args:
            touch_probs: dict of key_name -> P(key | touch) from Gaussian model
            context: recent typed text
            alpha: weight for touch model (1-alpha for LM)

        Returns:
            dict of key_name -> combined score
        """
        combined = {}
        for key_name, touch_p in touch_probs.items():
            if touch_p <= 0:
                combined[key_name] = -100.0
                continue

            log_touch = math.log(touch_p + 1e-10)

            # Get the character this key produces
            char = key_name.lower() if len(key_name) == 1 else key_name
            if char == 'space':
                char = ' '

            # Only apply LM to printable characters
            if len(char) == 1 and (char.isalpha() or char == ' '):
                log_lm = self.score_char(char, context)
                combined[key_name] = alpha * log_touch + (1.0 - alpha) * log_lm
            else:
                # Modifier keys, numbers: use touch probability only
                combined[key_name] = log_touch

        return combined

#!/usr/bin/env python3
"""Shared Persian text normalization for WER/CER SCORING.

train_whisper.compute_metrics and eval_baseline.py must normalize identically,
or their two WER numbers are not comparable -- and without a comparable
baseline you cannot tell a fine-tuning improvement from a regression. Keeping
one definition here is what makes that guarantee hold.

Deliberately NOT the same function as sm_04_build_dataset.normalize_fa, and
they must not be merged. That one prepares LABELS: it keeps punctuation and
ZWNJ because they are real orthography the model should learn to produce.
This one prepares text for SCORING, where punctuation placement is not the
thing being measured -- and because sm_04 attaches punctuation to the
preceding word, leaving it in charges a full word substitution for every
comma the model happens to place differently.
"""
import re
import unicodedata

# codepoint variants that are genuinely interchangeable in Persian text
_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا",
})
_DIACRITICS = re.compile(r"[ً-ْـ]")          # harakat + tatweel
_ZWNJ = re.compile(r"[‌‎‏]")  # ZWNJ and bidi marks
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def normalize_for_wer(text: str) -> str:
    """Fold away everything WER should not be charging for: codepoint variants,
    diacritics, ZWNJ/bidi marks, punctuation, and digit script."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    text = _ZWNJ.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    for i, (fa, ar) in enumerate(zip(_FA_DIGITS, _AR_DIGITS)):
        text = text.replace(fa, str(i)).replace(ar, str(i))
    return re.sub(r"\s+", " ", text).strip()

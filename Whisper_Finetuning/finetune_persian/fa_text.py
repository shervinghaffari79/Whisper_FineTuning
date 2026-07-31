#!/usr/bin/env python3
"""Shared Persian text normalization.

Two deliberately different functions, kept in one module so their contrast
stays visible instead of drifting apart in separate files -- they must NOT be
merged:

  normalize_for_wer -- for SCORING. train_whisper.compute_metrics and
  eval_baseline.py must normalize identically, or their two WER numbers are
  not comparable, and without a comparable baseline you cannot tell a
  fine-tuning improvement from a regression. Strips punctuation, since
  dataset builders attach punctuation to the preceding word, and leaving it
  in would charge a full word substitution for every comma the model happens
  to place differently -- burying the recognition signal this metric exists
  to expose.

  normalize_fa -- for LABELS (dataset-building scripts like
  hf_build_dataset.py). Keeps punctuation and ZWNJ, which are real
  orthography the model should learn to produce.
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


# Deliberately NOT the same normalization as normalize_for_wer above, and they
# must not be merged. This one prepares LABELS (dataset-building scripts like
# hf_build_dataset.py): it keeps punctuation and ZWNJ, which are real
# orthography the model should learn to produce. normalize_for_wer prepares
# text for SCORING, where punctuation placement is not the thing being
# measured. Own regexes, not shared with normalize_for_wer's -- kept separate
# on purpose so a future edit to one can't silently change the other.
_ARABIC_TO_PERSIAN_LABEL = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه"})
_DIACRITICS_LABEL = re.compile(r"[ً-ْـ]")


def normalize_fa(text: str) -> str:
    """Light normalization only -- keep punctuation and ZWNJ, which are real
    orthography the model should learn to produce. Unify the Arabic/Persian
    codepoint variants that are genuinely interchangeable."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_TO_PERSIAN_LABEL)
    text = _DIACRITICS_LABEL.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()

"""
Classroom-oriented text fixes applied BEFORE VieNeu's own normalizer.

The stock normalizer reads most school text well (2 + 3 = 5, 5 cm, 15%, 8h30,
"Bài 3:" ...). These rules cover what it reads differently from a Vietnamese
primary-school teacher:

    12 : 4 = 3   -> 12 chia 4 = 3          (it dropped the division sign)
    12 ÷ 4, 3 × 4 -> chia / nhân
    1/2, 3/4     -> 1 phần 2, 3 phần 4      (it said "một trên hai")
    3,14         -> 3 phẩy 14               (it said "ba phẩy một bốn")
    IV, XII      -> 4, 12                   (it spelled standalone Roman numerals)
    1,2,3        -> 1, 2, 3                 (a list, not "một phẩy hai phẩy ba")

Dates with a year (12/11/2026), times (8:30) and digit groups (1.000.000) are
left to the stock normalizer.
"""
from __future__ import annotations

import re

_NUM_LIST = re.compile(r"(?<![\d,.])\d+(?:,\d+){2,}(?![\d,])")       # 1,2,3 -> 1, 2, 3
_DIV_SPACED = re.compile(r"(?<=\d)\s+:\s+(?=\d)")                  # 12 : 4
_DIV_BEFORE_EQ = re.compile(r"(?<=\d):(?=\d+\s*=)")                  # 12:4 = 3
_DIV_SIGN = re.compile(r"\s*÷\s*")
_MUL_SIGN = re.compile(r"\s*[×✕]\s*")
_FRACTION = re.compile(r"(?<![\d/.,])(\d{1,3})/(\d{1,3})(?![\d/])")  # not part of a date like 12/11/2026
_DECIMAL = re.compile(r"(?<![\d,.])(\d+),([1-9]\d{0,2})(?![\d,])")   # 3,14 ; not 1,05 or 1,2,3
_ROMAN = {"II": 2, "III": 3, "IV": 4, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "XI": 11, "XII": 12,
          "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
          "XXI": 21}
_ROMAN_RE = re.compile(r"(?<![\w])(" + "|".join(sorted(_ROMAN, key=len, reverse=True)) + r")(?![\w])")


def normalize_edu(text: str) -> str:
    text = _DIV_SIGN.sub(" chia ", text)
    text = _MUL_SIGN.sub(" nhân ", text)
    text = _DIV_SPACED.sub(" chia ", text)
    text = _DIV_BEFORE_EQ.sub(" chia ", text)
    text = _NUM_LIST.sub(lambda m: m.group(0).replace(",", ", "), text)
    text = _FRACTION.sub(r"\1 phần \2", text)
    text = _DECIMAL.sub(r"\1 phẩy \2", text)
    text = _ROMAN_RE.sub(lambda m: str(_ROMAN[m.group(1)]), text)
    return re.sub(r"[ \t]{2,}", " ", text)

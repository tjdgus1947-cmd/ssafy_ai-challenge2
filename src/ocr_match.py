"""
OCR 텍스트 ↔ 보기 4개 매칭.

두 가지로 쓴다.
  1) 프롬프트 보조  : build_ocr_block() 로 OCR 결과를 프롬프트에 끼워 넣는다
  2) 독립 신호      : option_scores() 로 보기별 점수를 뽑아 VLM 확률과 앙상블한다

노트북에 그대로 복사해 넣어도 되고, import 해서 써도 된다.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

LETTERS = ["a", "b", "c", "d"]

# 한글/영문/숫자만 남긴다. OCR이 흔히 틀리는 구두점·공백 차이를 없애기 위함
_KEEP = re.compile(r"[^0-9a-z가-힣]")


def norm(s: str) -> str:
    return _KEEP.sub("", str(s).lower())


def digits(s: str) -> str:
    return re.sub(r"\D", "", str(s))


def _best_window_ratio(needle: str, hay: str) -> float:
    """hay 안에서 needle 과 가장 비슷한 구간의 유사도(0~1)."""
    if not needle or not hay:
        return 0.0
    if needle in hay:
        return 1.0
    n = len(needle)
    if n > len(hay):
        return SequenceMatcher(None, needle, hay).ratio()
    best = 0.0
    # 창을 needle 길이의 1.5배까지 보며 훑는다
    step = max(1, n // 4)
    for i in range(0, len(hay) - n + 1, step):
        for w in (n, int(n * 1.5)):
            seg = hay[i:i + w]
            if not seg:
                continue
            r = SequenceMatcher(None, needle, seg).ratio()
            if r > best:
                best = r
                if best >= 0.999:
                    return 1.0
    return best


def option_scores(ocr_text: str, options: list[str]) -> list[float]:
    """
    보기 4개 각각이 OCR 텍스트에 얼마나 나타나는지 0~1 점수.

    숫자가 든 보기는 숫자열 일치를 우선한다 — 가격·전화번호·날짜에서
    '1,800원' vs '1800' 같은 표기 차이를 흡수하기 위함.
    """
    hay = norm(ocr_text)
    hay_d = digits(ocr_text)
    out = []
    for o in options:
        o_n = norm(o)
        o_d = digits(o)
        s = _best_window_ratio(o_n, hay)
        if len(o_d) >= 3:                      # 숫자가 충분히 길면 숫자 매칭도 본다
            s = max(s, _best_window_ratio(o_d, hay_d))
        out.append(round(s, 4))
    return out


def ocr_prior(ocr_text: str, options: list[str], margin: float = 0.12
              ) -> tuple[list[float], bool]:
    """
    보기 점수를 확률처럼 정규화하고, '믿을 만한가'를 함께 반환한다.
    1등과 2등의 차이가 margin 이상일 때만 신뢰한다.
    """
    s = option_scores(ocr_text, options)
    order = sorted(s, reverse=True)
    confident = (order[0] >= 0.60) and (order[0] - order[1] >= margin)
    tot = sum(s)
    p = [x / tot for x in s] if tot > 0 else [0.25] * 4
    return p, confident


def build_ocr_block(ocr_text: str, options: list[str], max_chars: int = 400) -> str:
    """프롬프트에 끼울 OCR 블록. 비어 있으면 빈 문자열(프롬프트 오염 방지)."""
    t = re.sub(r"\s+", " ", str(ocr_text or "")).strip()
    if not t:
        return ""
    return f"[이미지에서 추출한 글자]\n{t[:max_chars]}\n"

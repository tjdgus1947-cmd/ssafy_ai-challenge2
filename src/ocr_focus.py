"""
OCR 박스 -> 질문과 관련된 영역 crop (focus panel)

왜 필요한가
-----------
원본은 짧은 변 720px인데 간판 글자는 그 안에서 수십 px밖에 안 된다.
전체 이미지를 1024로 키우면 토큰이 2배가 되지만 글자 영역의 실효 해상도는
겨우 1.4배 오른다. 대신 **글자가 있는 영역만 잘라 확대하면**
그 부분의 실효 해상도가 3~5배가 되면서 토큰은 오히려 줄어든다.

VLM에 [원본 이미지] + [focus crop] 두 장을 같이 넣는다.
Qwen2.5-VL 은 다중 이미지 입력을 지원한다.

crop 영역을 고르는 방법 (질문/보기와의 관련도 순)
  1. 보기 4개 중 하나와 문자열이 가장 잘 맞는 OCR 조각  <- 가장 강한 단서
  2. 질문에 나오는 고유명사('오봉집' 등)와 맞는 조각
  3. 아무것도 안 맞으면 crop 하지 않는다 (None 반환) — 엉뚱한 데를 확대하면 해롭다
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from PIL import Image

LETTERS = ["a", "b", "c", "d"]
_KEEP = re.compile(r"[^0-9a-z가-힣]")


def norm(s: str) -> str:
    return _KEEP.sub("", str(s).lower())


def parse_frags(ocr_json: str) -> list[dict]:
    """build_ocr_cache.py 가 저장한 ocr_json 컬럼을 파싱."""
    try:
        v = json.loads(ocr_json) if isinstance(ocr_json, str) else (ocr_json or [])
        return [f for f in v if isinstance(f, dict) and "b" in f]
    except (json.JSONDecodeError, TypeError):
        return []


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _quoted_terms(question: str) -> list[str]:
    """질문 속 따옴표 안의 고유명사. '오봉집', 'EUTHYMOL' 같은 것."""
    return re.findall(r"['\"‘’“”]([^'\"‘’“”]{2,20})['\"‘’“”]", str(question))


def pick_focus_boxes(frags: list[dict], question: str, options: list[str],
                     top_k: int = 3, min_score: float = 0.45) -> list[dict]:
    """관련도 높은 OCR 조각 top_k 개를 고른다."""
    if not frags:
        return []
    opt_n = [norm(o) for o in options]
    terms = [norm(t) for t in _quoted_terms(question)]

    scored = []
    for f in frags:
        t = norm(f.get("t", ""))
        if not t:
            continue
        s_opt = max((_ratio(t, o) for o in opt_n), default=0.0)
        s_term = max((_ratio(t, q) for q in terms), default=0.0)
        # 보기 일치를 더 신뢰한다
        score = max(s_opt, 0.8 * s_term)
        if score >= min_score:
            scored.append((score, f))

    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:top_k]]


def union_box(boxes: list[list[int]]) -> list[int]:
    xs1 = [b[0] for b in boxes]; ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]; ys2 = [b[3] for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _center(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def cluster_near_best(picked: list[dict], w: int, h: int,
                      max_dist_ratio: float = 0.22) -> list[list[int]]:
    """
    1등 조각 주변에 있는 것만 합친다.

    멀리 떨어진 조각까지 union 하면 crop 이 원본만큼 넓어져 확대 이득이 사라진다
    (이 함수가 없으면 실효 확대율이 1배 아래로 내려가는 경우가 생긴다).
    """
    if not picked:
        return []
    best = picked[0]["b"]
    bx, by = _center(best)
    lim = max_dist_ratio * max(w, h)
    out = [best]
    for f in picked[1:]:
        cx, cy = _center(f["b"])
        if abs(cx - bx) <= lim and abs(cy - by) <= lim:
            out.append(f["b"])
    return out


def expand_clip(box: list[int], w: int, h: int,
                margin: float = 0.35, min_side: int = 96) -> list[int]:
    """박스에 여유를 주고 이미지 밖으로 나가지 않게 자른다."""
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    mx, my = bw * margin, bh * margin
    x1, y1 = x1 - mx, y1 - my
    x2, y2 = x2 + mx, y2 + my

    # 너무 얇은 박스는 최소 크기까지 키운다 (한 줄짜리 간판 글씨 대비)
    if x2 - x1 < min_side:
        cx = (x1 + x2) / 2; x1, x2 = cx - min_side / 2, cx + min_side / 2
    if y2 - y1 < min_side:
        cy = (y1 + y2) / 2; y1, y2 = cy - min_side / 2, cy + min_side / 2

    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1: x1, x2 = 0, w
    if y2 <= y1: y1, y2 = 0, h
    return [x1, y1, x2, y2]


def _focus_box(ocr_json, question, options, w, h):
    """crop 할 최종 박스. 없으면 None."""
    picked = pick_focus_boxes(parse_frags(ocr_json), question, options)
    if not picked:
        return None
    boxes = cluster_near_best(picked, w, h)
    return expand_clip(union_box(boxes), w, h)


def make_focus_crop(img: Image.Image, ocr_json: str, question: str,
                    options: list[str], target_px: int = 640,
                    max_area_ratio: float = 0.55, min_zoom: float = 1.5):
    """
    focus crop 이미지를 만든다. 확대 이득이 없으면 None 을 돌려준다.

    target_px      : crop 의 긴 변을 이 크기로 확대한다
    max_area_ratio : crop 이 원본의 이 비율보다 크면 사실상 전체라 의미 없음
    min_zoom       : 이 배율 미만이면 crop 해봐야 이득이 없으므로 포기
    """
    w, h = img.size
    box = _focus_box(ocr_json, question, options, w, h)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    if cw * ch > max_area_ratio * w * h:
        return None
    scale = target_px / max(cw, ch)
    if scale < min_zoom:
        return None

    crop = img.crop((x1, y1, x2, y2))
    return crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))),
                       Image.LANCZOS)


def effective_zoom(img_size, ocr_json, question, options,
                   target_px: int = 640, max_area_ratio: float = 0.55,
                   min_zoom: float = 1.5):
    """crop 이 실효 해상도를 몇 배 올리는지 (디버깅·검증용). 이득 없으면 None."""
    w, h = img_size
    box = _focus_box(ocr_json, question, options, w, h)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    if cw * ch > max_area_ratio * w * h:
        return None
    z = target_px / max(cw, ch)
    return round(z, 2) if z >= min_zoom else None

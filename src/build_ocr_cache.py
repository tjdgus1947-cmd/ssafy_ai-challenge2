#!/usr/bin/env python3
"""
SSAFY 16기 2회차 AI 챌린지 — OCR 캐시 생성기

질문의 90%가 '이미지 속 글자 읽기'이고, 보기의 76%가 이미지에 그대로 찍혀 있을
법한 문자열이다. 한국어 OCR을 한 번 돌려 캐시해 두면
  1) 프롬프트에 OCR 결과를 같이 넣어 VLM의 판독을 보조하고
  2) OCR 텍스트와 보기 4개를 문자열 매칭해 독립적인 예측 신호를 만들 수 있다.

한 번 만들어 두면 모든 실험이 재사용한다. 학습이 도는 동안 다른 팀원이 돌리면 된다.

사용법
------
    # 4명이 나눠 돌리기 (각자 shard 0,1,2,3)
    python build_ocr_cache.py --data-dir ./data --split train --shard 0 --num-shards 4
    python build_ocr_cache.py --data-dir ./data --split test  --shard 0 --num-shards 4

    # 합치기
    python build_ocr_cache.py --data-dir ./data --split train --merge

엔진
----
    easyocr : 설치가 쉽고 안정적. pip install easyocr
    paddle  : 한국어 간판 정확도가 더 좋은 편. 설치가 까다로움.
              pip install paddlepaddle-gpu paddleocr

중간 저장하므로 끊겨도 같은 명령을 다시 실행하면 이어서 간다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

MAX_CHARS = 600          # 프롬프트 폭주 방지: 이미지당 OCR 텍스트 상한
SAVE_EVERY = 100


# ────────────────────────────── 엔진 ──────────────────────────────
def _poly_to_xyxy(poly) -> list[int]:
    """4점 다각형 -> [x1, y1, x2, y2] 정수 박스."""
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, gpu: bool = True):
        import easyocr
        self.reader = easyocr.Reader(["ko", "en"], gpu=gpu)

    def read(self, path: str) -> list[dict]:
        out = self.reader.readtext(path, detail=1, paragraph=False)
        frags = []
        for box, text, conf in out:
            frags.append({"t": str(text),
                          "c": round(float(conf), 3),
                          "b": _poly_to_xyxy(box)})
        return frags


class PaddleEngine:
    name = "paddle"

    def __init__(self, gpu: bool = True):
        from paddleocr import PaddleOCR
        self.reader = PaddleOCR(use_angle_cls=True, lang="korean",
                                use_gpu=gpu, show_log=False)

    def read(self, path: str) -> list[dict]:
        res = self.reader.ocr(path, cls=True)
        frags = []
        for page in (res or []):
            for line in (page or []):
                # line = [box(4점), (text, conf)]
                try:
                    frags.append({"t": str(line[1][0]),
                                  "c": round(float(line[1][1]), 3),
                                  "b": _poly_to_xyxy(line[0])})
                except (IndexError, TypeError):
                    continue
        return frags


def make_engine(name: str, gpu: bool):
    if name == "paddle":
        return PaddleEngine(gpu=gpu)
    return EasyOCREngine(gpu=gpu)


# ────────────────────────────── 본체 ──────────────────────────────
def clean_fragments(frags: list[dict], min_conf: float) -> tuple[str, str]:
    """
    신뢰도 낮은 조각을 버리고
      - 프롬프트에 넣을 한 줄 텍스트
      - bbox 까지 담은 JSON 문자열  (crop 생성에 쓴다)
    을 만든다.
    """
    keep = []
    for f in frags:
        t = re.sub(r"\s+", " ", str(f.get("t", ""))).strip()
        if not t or float(f.get("c", 0)) < min_conf:
            continue
        keep.append({"t": t, "c": f.get("c", 0), "b": f.get("b", [0, 0, 0, 0])})

    text = " | ".join(k["t"] for k in keep)[:MAX_CHARS]
    boxes = json.dumps(keep, ensure_ascii=False, separators=(",", ":"))
    return text, boxes


def run_shard(args) -> None:
    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / f"{args.split}.csv")
    ids = df["id"].astype(str).tolist()
    paths = [str(data_dir / p) for p in df["path"].astype(str)]

    # shard 분할
    sel = [i for i in range(len(df)) if i % args.num_shards == args.shard]
    out_path = Path(args.out_dir) / f"ocr_{args.split}_shard{args.shard}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if out_path.exists():
        prev = pd.read_csv(out_path).astype({"id": str})
        done = {r["id"]: r for r in prev.to_dict("records")}
        print(f"이어서 진행: {len(done)}건 완료됨")

    todo = [i for i in sel if ids[i] not in done]
    print(f"[{args.split}] shard {args.shard}/{args.num_shards} — "
          f"할당 {len(sel)}건, 남은 {len(todo)}건")
    if not todo:
        print("할 일 없음")
        return

    engine = make_engine(args.engine, gpu=not args.cpu)
    print(f"엔진: {engine.name}")

    rows = list(done.values())
    t0 = time.time()
    for n, i in enumerate(todo, start=1):
        try:
            frags = engine.read(paths[i])
            text, boxes = clean_fragments(frags, args.min_conf)
            nbox = len(frags)
        except Exception as e:                      # 한 장 실패로 전체가 멈추지 않게
            print(f"  [실패] {ids[i]}: {e}", file=sys.stderr)
            text, boxes, nbox = "", "[]", -1

        rows.append({"id": ids[i], "ocr_text": text, "n_box": nbox, "ocr_json": boxes})

        if n % SAVE_EVERY == 0 or n == len(todo):
            pd.DataFrame(rows).to_csv(out_path, index=False)
            rate = n / max(1e-9, time.time() - t0)
            eta = (len(todo) - n) / max(1e-9, rate) / 60
            print(f"  {n}/{len(todo)}  {rate:.1f}장/초  남은 {eta:.0f}분", flush=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("저장:", out_path)


def merge(args) -> None:
    out_dir = Path(args.out_dir)
    parts = sorted(out_dir.glob(f"ocr_{args.split}_shard*.csv"))
    if not parts:
        raise SystemExit(f"{out_dir} 에 ocr_{args.split}_shard*.csv 가 없다")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.astype({"id": str}).drop_duplicates("id", keep="last")

    src = pd.read_csv(Path(args.data_dir) / f"{args.split}.csv").astype({"id": str})
    missing = set(src["id"]) - set(df["id"])
    dst = out_dir / f"ocr_{args.split}.csv"
    df.to_csv(dst, index=False)

    df["ocr_text"] = df["ocr_text"].fillna("")
    print(f"합침: {len(parts)}개 shard -> {len(df)}건  ({dst})")
    print(f"  누락        : {len(missing)}건")
    print(f"  빈 텍스트   : {(df['ocr_text'].str.len() == 0).mean():.1%}")
    print(f"  평균 글자수 : {df['ocr_text'].str.len().mean():.0f}")
    if missing:
        print("  누락 id 예시:", list(missing)[:5])


def main() -> None:
    p = argparse.ArgumentParser(description="한국어 OCR 캐시 생성")
    p.add_argument("--data-dir", default="./data", help="train.csv 와 train/ 이 있는 폴더")
    p.add_argument("--out-dir", default="./ocr", help="결과 저장 폴더")
    p.add_argument("--split", default="train", choices=["train", "test", "dev"])
    p.add_argument("--engine", default="easyocr", choices=["easyocr", "paddle"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--min-conf", type=float, default=0.30,
                   help="이 신뢰도 미만의 조각은 버린다")
    p.add_argument("--cpu", action="store_true", help="GPU 없이 실행")
    p.add_argument("--merge", action="store_true", help="shard 파일들을 합치기만 한다")
    args = p.parse_args()

    if args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()

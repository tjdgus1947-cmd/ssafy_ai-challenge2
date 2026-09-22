# 외부 자료 출처

대회 규칙상 **외부 데이터·모델을 쓰면 링크를 함께 제출해야 한다.** 쓰는 즉시 여기에 적는다.

> 규칙 확인 사항
> - 외부 데이터 사용 가능. 단 **공개적으로 사용 가능한 데이터**여야 한다
> - 허깅페이스 사전학습 모델 사용 가능
> - 데이터 증강은 train 및 dev 에 한해 허용
> - **API 호출로 추론하는 것은 금지** (GPT-4V, Gemini 등 — 실격 사유)
> - 전처리 목적의 모델 사용 가능 (객체 탐지, OCR 등)

라이선스는 최종 제출 전에 각 모델 카드에서 다시 확인한다.

---

## 사전학습 모델 (VQA 본체)

| 이름 | 링크 | 사용 | 역할 |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct | 비교만 | 공식 베이스라인, 제로샷 기준점 (0.860) |
| Qwen2.5-VL-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | **사용** | 제로샷 앙상블 멤버 (다른 계열로 다양성 제공) |
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | **사용** | 주력 베이스. LoRA 학습 (768 / 1024) |
| Qwen3-VL-32B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct | **사용** | LoRA 학습 앙상블 멤버 |
| Qwen3-VL-4B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | 검토 | 경량 앙상블 후보 |

모든 모델은 bitsandbytes 4bit(nf4)로 로드하고 LoRA(r=16)만 학습했다. 가중치 원본은 수정하지 않았다.

---

## 전처리 모델 · 라이브러리

| 이름 | 링크 | 라이선스 | 용도 | 사용 |
|---|---|---|---|---|
| EasyOCR | https://github.com/JaidedAI/EasyOCR | Apache-2.0 | 한국어 OCR (텍스트 + 박스) → 매칭 점수 | **사용** |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0 | 한국어 OCR 대안 (품질 비교) | 검토 |
| transformers | https://github.com/huggingface/transformers | Apache-2.0 | 모델 로딩·추론 | 사용 |
| PEFT | https://github.com/huggingface/peft | Apache-2.0 | LoRA 학습 | 사용 |
| bitsandbytes | https://github.com/bitsandbytes-foundation/bitsandbytes | MIT | 4bit 양자화 | 사용 |

---

## 외부 데이터셋

사용하지 않음. 학습·검증 모두 대회 train.csv만 사용했다.
(dev.csv는 라벨 신뢰도 문제로 사용하지 않음 — `EDA.md` 참고)

---

## 참고한 공개 솔루션

| 이름 | 링크 | 참고한 내용 |
|---|---|---|
| SSAFY 15기 2회차 1위 (MinjuJangg) | https://github.com/MinjuJangg/ai_chall | choice_ce 손실, 보기 순서 셔플·TTA, 앙상블 |
| SSAFY 14기 (junhyung8795) | https://github.com/junhyung8795/SSAFY_AI_Challenge | softmax 기반 추론, LoRA 하이퍼파라미터 |

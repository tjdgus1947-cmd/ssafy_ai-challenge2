# 외부 자료 출처

대회 규칙상 **외부 데이터·모델을 쓰면 링크를 함께 제출해야 한다.**
쓰는 즉시 여기에 적는다. 마지막 날 몰아서 정리하면 뭘 썼는지 기억나지 않는다.

> 규칙 확인 사항
> - 외부 데이터 사용 가능. 단 **공개적으로 사용 가능한 데이터**여야 한다
> - 허깅페이스 사전학습 모델 사용 가능
> - 데이터 증강은 train 및 dev 에 한해 허용
> - **API 호출로 추론하는 것은 금지** (GPT-4V, Gemini 등 — 실격 사유)
> - 전처리 목적의 모델 사용 가능 (객체 탐지, OCR 등)

---

## 사전학습 모델

| 이름 | 링크 | 라이선스 | 용도 | 사용 여부 |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct | Apache-2.0 | VQA 본체 (공식 베이스라인) | 사용 |
| Qwen2.5-VL-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | Apache-2.0 | VQA 본체 (확장) | 검토 |
| VARCO-VISION-2.0-1.7B-OCR | https://huggingface.co/NCSOFT/VARCO-VISION-2.0-1.7B-OCR | cc-by-nc-4.0 | 한국어 OCR 추출 | 검토 |
| VARCO-VISION-2.0-14B | https://huggingface.co/NCSOFT/VARCO-VISION-2.0-14B | cc-by-nc-4.0 | VQA 본체 (한국어 특화) | 검토 |

---

## 라이브러리 / 도구

| 이름 | 링크 | 라이선스 | 용도 | 사용 여부 |
|---|---|---|---|---|
| EasyOCR | https://github.com/JaidedAI/EasyOCR | Apache-2.0 | 한국어 OCR 추출 (텍스트 + bbox) | 사용 |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0 | 한국어 OCR (대안) | 검토 |
| transformers | https://github.com/huggingface/transformers | Apache-2.0 | 모델 로딩·학습 | 사용 |
| PEFT (LoRA) | https://github.com/huggingface/peft | Apache-2.0 | 파라미터 효율 파인튜닝 | 사용 |
| bitsandbytes | https://github.com/bitsandbytes-foundation/bitsandbytes | MIT | 4bit 양자화 | 사용 |

---

## 외부 데이터셋

현재 사용 없음. 쓰게 되면 아래에 추가한다.

| 이름 | 링크 | 라이선스·공개 여부 | 용도 | 사용 여부 |
|---|---|---|---|---|
| | | | | |

> 주의: AI Hub 등 로그인·승인이 필요한 데이터는 "공개적으로 사용 가능"에
> 해당하는지 애매하다. 쓰려면 먼저 디스커션으로 확인할 것.

---

## 참고한 공개 솔루션

| 이름 | 링크 | 참고한 내용 |
|---|---|---|
| SSAFY 15기 2회차 1위 (MinjuJangg) | https://github.com/MinjuJangg/ai_chall | choice_ce 손실, 보기 순서 셔플·TTA, margin 기반 앙상블 |
| SSAFY 14기 (junhyung8795) | https://github.com/junhyung8795/SSAFY_AI_Challenge | softmax 기반 추론, LoRA 하이퍼파라미터 |

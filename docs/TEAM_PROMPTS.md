# 역할별 LLM 프롬프트

각자 쓰는 LLM(Claude, ChatGPT 등) **새 채팅**에
**[공통 컨텍스트] + [내 역할 프롬프트]** 를 통째로 붙여넣고 시작하세요.
그러면 LLM이 우리 팀이 지금까지 알아낸 걸 전부 아는 상태로 도와줍니다.

사용 팁
- 노트북 셀 **출력은 그대로 복사해서** 붙여넣기 (요약하지 말 것)
- 에러가 나면 **Traceback 전체**를 붙여넣기
- LLM이 준 코드는 **본인 사본 노트북**(`TEMPLATE` 사본에 `_이름` 을 붙인 것)에서만 실행. TEMPLATE은 건드리지 않기
- 결과 숫자는 단톡 + `docs/EXPERIMENTS.md` 에 기록

---

## [공통 컨텍스트] — 모든 역할이 맨 앞에 붙여넣기

```
너는 SSAFY 16기 2회차 AI 챌린지에 참가한 우리 팀(4명)의 기술 조력자야.
나는 팀원 중 한 명이고, 아래 컨텍스트를 전부 이해한 상태로 내 역할을 도와줘.

# 대회
- 캐글 대회. 기간 2026-09-21 ~ 09-28. 평가는 Accuracy.
- 입력: 이미지 + 한국어 질문 + 보기 a/b/c/d. 출력: a/b/c/d 중 하나.
- 제출: id,answer 형식 CSV. 팀당 하루 20회(팀 전체 공유).
- 규칙:
  * API 호출로 추론하는 방식은 금지(실격). GPT/Gemini/Claude API로 문제를 푸는 코드는 절대 제안하지 마.
  * 허깅페이스 사전학습 모델 사용 가능. 반드시 VQA 모델이 본체여야 함.
  * 외부 데이터·모델 사용 가능하지만 링크를 함께 제출해야 함. 공개 데이터여야 함.
  * 전처리 목적의 모델(OCR, 객체탐지 등) 사용 가능.
  * 코드와 결과는 재현 가능해야 함(SEED 고정).

# 데이터 (train 전수 분석 결과)
- train 6,714 / test 6,714 / dev 2,683건. 컬럼: id, path, question, a, b, c, d, answer
- dev는 정답이 없고 answer1~5(교육생 5명 응답)만 있음. 최다응답 일치 인원이 최대 3명(4명 이상 0건)이라
  사실상 "라벨 기준 미달로 폐기된 문항"이다. 기본적으로 쓰지 않는다.
- 이미지: 짧은 변 720px (720x960 등). 장당 약 106KB.
- 정답 분포 a/b/c/d 거의 균등.
- 질문 유형 비중: 상호·이름 25.8% / 문구·판독 21.4% / 가격·할인 13.7% / 기타 10.9% /
  위치·주소 8.2% / 번호·연락처 5.7% / 메뉴·상품 4.9% / 시간·일정 4.8% / 부정형 4.1% /
  색상 0.4% / 카운팅 0.2%
  → 거의 전부 "사진 속 글자 읽기(OCR)" 문제다.
- 보기 유사도: 보기 4개 중 가장 비슷한 두 개의 편집거리가 1글자인 문항 25.9%, 2글자 이하 44.8%.
  예) 1,800원 / 1,300원 / 1,200원 / 1,700원
  → 해상도(글자를 얼마나 선명하게 보여주느냐)가 가장 큰 레버다.

# 환경
- Google Colab Pro+ 팀 공유 계정 1개를 4명이 같이 씀. 컴퓨팅 단위도 공유.
  A100은 동시에 2명까지만. 다 쓰면 반드시 "런타임 연결 해제".
- 드라이브 루트: /content/drive/MyDrive
- 데이터: 3번 셀이 드라이브의 ssafy-16-2-ai.zip(암호 있음)을 /content/data 로 푼다(약 40초).
- 저장 경로: OUT_DIR = /content/drive/MyDrive/ssafy_ai/<WHO>  (WHO=본인 이름, 팀원끼리 덮어쓰기 방지)

# 노트북 (드라이브 Colab Notebooks의 TEMPLATE 사본, 위에서 아래로 실행하면 제출 파일까지 나옴)
- 0번 설정: 실험은 여기 변수만 바꾼다.
  WHO, MODEL_SIZE("3b"/"7b"), LOSS_MODE("choice_ce"/"mask"), SHUFFLE_CHOICES,
  IMAGE_SIZE(기본 768), MIN_PIXELS, MAX_PIXELS=IMAGE_SIZE**2, TRAIN_N, VALID_N=500,
  EPOCHS, BATCH=1, GRAD_ACCUM=8, LR=1e-4, LORA_R=16, EVAL_BATCH=4, SEED=42, N_TTA,
  ZIP_PW, DATA_DIR, OUT_DIR, TAG(실험 이름 자동 생성)
- 1번 설치(설치 후 세션 재시작 필요), 2번 임포트(GPU 없으면 assert로 멈춤)
- 3번 데이터 → train_all, valid_df(홀드아웃 500), fit_df(학습용), test_df 생성. path는 절대경로.
- 4번 EDA: qtype(question) 함수(위 11개 유형 분류), min_pair_diff(row)
- 5번 모델: Qwen2.5-VL (3B 또는 7B), 4bit 양자화 + LoRA. processor, tok, base_model, model 생성
- 6번 프롬프트: SYSTEM_INSTRUCT, build_mc_prompt, build_messages(question, opts, img, answer),
  LETTERS=["a","b","c","d"], LETTER_IDS, LETTER_T, IM_END_ID
- 7번 추론 함수 (generate 대신 마지막 위치 logit에서 a/b/c/d 4개만 뽑아 softmax):
  predict_probs(df, batch_size=None, desc, rotate=0, save_path=None) -> [N,4] 확률 텐서
  probs_to_letters(probs), acc_of(probs, df), evaluate(df, name, tta=0) -> (acc, probs),
  report_by_qtype(df, probs, name) -> 유형별 정확도 출력 + correct/qtype 컬럼이 붙은 DataFrame 반환
- 8번 제로샷 기준점 (학습 없이 홀드아웃 평가)
- 9번 Dataset/Collator: 학습 시 보기 순서 셔플, choice_ce용 ans_pos/choice_labels 생성
- 10번 파인튜닝: LOSS_MODE="choice_ce"면 정답 위치 직전 logit에서 a/b/c/d 4개만 뽑아 4-class CE
- 11번 평가 + TTA(보기 순서를 회전시켜 여러 번 추론 후 확률 합산)
- 12번 test 추론 + 제출 파일 + 제출 전 체크
- 부록 A: 세션이 끊겼을 때 저장된 LoRA 어댑터만 다시 불러오기

# 지금까지 결과 (09/21 14시 기준)
- 제로샷(학습 없음, IMAGE_SIZE 768):
  3B  홀드아웃 0.8600 / 리더보드 0.86267
  7B  홀드아웃 0.9120 / 리더보드 0.91957   → 베이스 모델 7B로 확정 (0번 셀 MODEL_SIZE="7b")
- 홀드아웃 500건이 리더보드를 잘 예측함(2회 모두 차이 +0.3~0.8%p, 순위 일치).
  단 오차가 ±1.3%p라 1%p 안쪽 차이는 "비슷함"으로 본다.
- 7B 제로샷 유형별(홀드아웃): 상호·이름 0.945(128) / 문구·판독 0.941(101) / 기타 0.923(52) /
  시간·일정 0.909(33) / 가격·할인 0.892(83) / 부정형 0.889(9) / 위치·주소 0.872(39) /
  번호·연락처 0.871(31) / 메뉴·상품 0.783(23)
- 3B→7B에서 추론형(위치·주소 +15, 부정형)이 가장 많이 올랐고, 메뉴·상품은 그대로(0.783).
  남은 오답은 대부분 읽기형: 가격·할인 > 상호·이름 > 문구·판독 > 메뉴·상품 > 위치·주소
- 속도(A100, EVAL_BATCH=4): 7B 추론이 3B와 거의 같음(홀드아웃 2분40초, test 36분).
  병목은 GPU가 아니라 이미지 로딩·전처리(CPU).

# 팀 역할
① 모델 선택  ② 학습 레시피  ③ OCR 채널  ④ 분석·제출

# 너에게 바라는 방식
- 코드는 노트북에 그대로 붙여넣을 수 있는 "셀 단위"로 줘. 이미 있는 변수·함수(위 목록)를 최대한 재사용해.
- 확실하지 않은 건 추측하지 말고, 먼저 확인하는 코드를 줘.
- 내가 셀 출력을 붙여넣으면 해석해주고 "다음에 할 한 가지"를 알려줘.
- 한 번에 변수 하나만 바꾸는 실험 설계를 지켜줘.
- GPU 컴퓨팅 단위가 공유 자원이라는 걸 고려해서 불필요하게 오래 걸리는 제안은 피해줘.
```

---

## [① 모델 선택] 프롬프트

```
# 내 역할: ① 모델 선택 (A100 사용)
목표: 어떤 베이스 모델로 갈지 결정한다. 리더보드 상위권과의 차이가 모델에서 오는지를 가려낸다.
나는 TEMPLATE 노트북 관리자이기도 하다.

## 오늘 할 일
1) (완료) 7B 제로샷 홀드아웃 0.9120 (3B 0.8600 대비 +5.2%p) → 베이스 7B 확정
2) (선택) Qwen2.5-VL-32B 제로샷 홀드아웃 — 크기 효과가 더 남았는지 확인.
   다만 남은 오답이 대부분 읽기형이라 32B보다 해상도·OCR·파인튜닝이 더 효과적일 가능성이 크다.
   32B는 용량이 60GB 이상이라 다운로드·로딩 시간과 디스크를 먼저 확인할 것.
3) 다른 계열 "도전자" 모델 탐색 (별도 사본 노트북에서만)
   - 후보: Qwen3-VL 계열, NCSOFT/VARCO-VISION-2.0 (한국어 특화, LLaVA-OneVision 구조), InternVL
   - 학습까지 가지 말고 "제로샷 홀드아웃"만 먼저 잰다.
   - 포팅 시 바뀌는 것: 모델 로딩 클래스, processor 인자(Qwen의 min_pixels/max_pixels는 다른 모델엔 없음),
     채팅 템플릿, a/b/c/d의 토큰 id(LETTER_IDS), 답이 나오기 직전 위치.
     predict_probs의 "마지막 위치 logit에서 a/b/c/d 4개만 보기" 방식이 성립하려면,
     생성 프롬프트가 정답 글자 바로 앞에서 끝나야 한다. 이걸 먼저 검증해줘.
   - 갈아탈 기준: 도전자 제로샷이 Qwen2.5-VL-7B 제로샷(0.9120)보다 확실히(2~3%p 이상) 높을 때만 학습 포팅.
4) TEMPLATE 관리: 팀원이 찾은 개선을 모아서 TEMPLATE에 반영하고 버전 번호를 올린다.
   TEMPLATE의 ZIP_PW(암호)는 깃허브에 올리지 않는다.

## 주의
- 모델 교체는 코드가 가장 잘 꼬이는 작업. 메인 TEMPLATE은 직접 실험하지 말고 사본에서만.
- 모델을 바꾸면 5번 셀(모델 로드)부터 다시 실행해야 한다.
- 7B는 VRAM을 더 쓰므로 OOM이 나면 EVAL_BATCH를 줄인다.

## 오늘 17:30까지 산출물
- (완료) 3B vs 7B 제로샷 표와 결정
- TEMPLATE을 7B 기본값으로 업데이트해 팀에 배포
- (가능하면) 32B 또는 도전자 모델 1개의 제로샷 홀드아웃 점수

다음으로 무엇을 확인하는 게 팀에 가장 도움이 될지 같이 판단해줘.
```

---

## [② 학습 레시피] 프롬프트

```
# 내 역할: ② 학습 레시피 (A100 사용)
목표: 파인튜닝 설정 중 무엇이 점수를 올리는지 하나씩 확인한다. 내 실험 기록이 그대로 발표 자료가 된다.
베이스는 7B로 확정됐다(0번 셀 MODEL_SIZE="7b"). 7B 추론 속도가 3B와 거의 같아서 레시피 실험도 7B로 바로 한다.
7B 학습이 너무 느리면 그때만 3B로 레시피를 찾고 7B에 적용한다(같은 Qwen2.5-VL 계열이라 결론이 거의 옮겨간다).

## 실험 순서
0) 학습 전 빠른 확인 (학습 없이, 5분씩): 7B 제로샷을 IMAGE_SIZE 512와 1024로 (768은 이미 0.9120)
   - 0번 셀 IMAGE_SIZE 변경 → 0번 재실행 → 5번(모델·processor 재생성) → 8번
   - 1024는 EVAL_BATCH=2
1) 학습 실험은 전부 TRAIN_N=2000, EPOCHS=1 (한 번에 30~40분 예상). 이긴 설정만 나중에 TRAIN_N=6000.
2) 해상도: 512 / 768 / 1024 (LOSS_MODE="choice_ce", SHUFFLE_CHOICES=True 고정)
3) 이긴 해상도에서 LOSS_MODE: "choice_ce" vs "mask"
4) 그다음 SHUFFLE_CHOICES: True vs False
5) 최고 설정으로 TRAIN_N=6000 학습 → test 추론 → 제출 파일을 ④에게 전달

## 매 학습 실험마다 실행할 셀
0번(설정 변경 후 재실행) → 5번(모델 다시 로드) → 9번 → 10번 → 11번의 첫 셀(평가)
⚠️ 실험 사이에 5번을 반드시 다시 실행. 안 하면 이전 실험의 LoRA 위에 이어서 학습돼 비교가 오염된다.

## 학습 중 확인할 것
- 첫 loss: 7B 제로샷이 이미 91%라 choice_ce loss는 0.3~0.5 근처에서 시작하는 게 정상.
  1.39(=ln4, 무작위) 근처면 뭔가 어긋난 것 → 멈추고 확인.
- 몇 백 스텝 만에 0.05 아래로 떨어지면 과적합 의심.
- tqdm 속도로 총 소요시간 계산해서 기록.

## 제출 파일 만들기 (11번의 TTA 셀을 건너뛴 경우)
12번 셀은 11번 TTA 셀의 변수(hold)를 참조하므로, TTA를 안 돌렸으면 아래를 대신 쓴다:
    probs = predict_probs(test_df, desc="test", save_path=f"{OUT_DIR}/test_{TAG}_r0.pt")
    sub_name = f"sub_{TAG}_hold{ft_acc:.4f}.csv"
    submission = pd.DataFrame({"id": test_df["id"], "answer": probs_to_letters(probs)})
    submission.to_csv(f"/content/{sub_name}", index=False)
    submission.to_csv(f"{OUT_DIR}/{sub_name}", index=False)
    print(sub_name)
그다음 12번의 "제출 전 체크" 셀 실행.

## 기록 형식 (단톡 + EXPERIMENTS.md)
TAG / TRAIN_N / 홀드아웃 정확도 / 제로샷 대비 증감 / 유형별 표 / 학습 소요시간 / 첫 loss·마지막 loss

## 오늘 17:30까지 산출물
- 해상도 3개 비교 결과(홀드아웃 + 유형별)
- 가능하면 choice_ce vs mask 1회

첫 실험 설정을 같이 점검해주고, 셀 출력을 붙여넣으면 해석해줘.
```

---

## [③ OCR 채널] 프롬프트

```
# 내 역할: ③ OCR 채널 (T4 사용, A100 쓰지 않음)
목표: 사진 속 글자를 텍스트로 뽑아 캐시해 둔다. 오답 기여 1위인 가격·할인(한 글자 차이 숫자) 문제를 잡을 무기.
노트북 학습과 완전히 독립적으로 돌아간다.

## 가진 도구 (레포 src/, 드라이브 MyDrive/ssafy_ai/src/ 에 올려서 사용)
- build_ocr_cache.py : EasyOCR(기본) 또는 PaddleOCR로 이미지별 글자 추출.
  인자: --data-dir --out-dir --split {train,test,dev} --engine {easyocr,paddle}
        --shard N --num-shards M --min-conf 0.30 --merge
  출력 CSV 컬럼: id, ocr_text(프롬프트용 한 줄, 최대 600자), n_box, ocr_json(글자별 bbox 포함 JSON)
  중간 저장되므로 끊겨도 같은 명령 재실행하면 이어서 간다.
- ocr_match.py : option_scores(ocr_text, options) → 보기 4개별 0~1 점수
                 ocr_prior(ocr_text, options) → (정규화 확률, confident 여부)
                 build_ocr_block(ocr_text, options) → 프롬프트에 끼울 텍스트
- ocr_focus.py : make_focus_crop(img, ocr_json, question, options) → 글자 영역 확대 crop (내일 이후 사용)

## 순서
1) 노트북 0~4번 셀까지 실행해 데이터 풀기 (런타임 T4). 5번(모델)은 필요 없다.
2) 20장 테스트 (shard를 잘게 쪼개면 소량만 돈다):
     !pip install -q easyocr
     !python /content/drive/MyDrive/ssafy_ai/src/build_ocr_cache.py \
        --data-dir /content/data --out-dir /content/drive/MyDrive/ssafy_ai/ocr \
        --split train --shard 0 --num-shards 300
   - 결과 CSV를 열어 가격·전화번호가 실제 사진과 맞게 읽혔는지 5개 이상 눈으로 확인
   - 로그의 "장/초" 속도 기록 → 전체 13,428장 소요시간 계산
3) 괜찮으면 train, test 전체 실행 (④와 --num-shards 2 로 나눠도 됨) → 끝나면 --merge
4) OCR이 쓸모 있는지 train으로 측정 (정답이 있으므로 바로 잴 수 있음):
   - 전체 정확도, confident 비율, confident 구간 정확도
   - 유형별로도(특히 가격·할인, 번호·연락처, 상호·이름) → qtype 함수 재사용
   confident 구간 정확도가 95% 이상이면 강력한 앙상블 신호다.
5) (내일) OCR 텍스트를 프롬프트에 넣는 실험은 본인 사본에서 하고 결과를 ①에게 전달해 TEMPLATE 반영.

## 주의
- --out-dir 은 반드시 드라이브. 로컬(/content)에 두면 세션 끊길 때 사라진다.
- 모의 실험에선 confident 정확도 97.9%가 나왔지만 실제 OCR 품질은 아직 모른다. 그걸 재는 게 내 일이다.
- 한국어 간판 OCR 품질이 나쁘면 --engine paddle 또는 한국어 특화 OCR 모델
  (NCSOFT/VARCO-VISION-2.0-1.7B-OCR, 긴 변 2304px 업스케일 권장)을 비교 후보로 검토.
  쓰면 docs/EXTERNAL_SOURCES.md 에 링크 기록.

## 오늘 17:30까지 산출물
- 20장 품질 확인 결과 + 속도
- 가능하면 train OCR 완료 + 위 정확도 3개 숫자(전체/유형별)

먼저 20장 테스트를 같이 진행하고, 결과 CSV를 붙여넣으면 품질을 판단해줘.
```

---

## [④ 분석·제출] 프롬프트

```
# 내 역할: ④ 분석·제출 (T4 사용)
목표: 모델이 "왜" 틀리는지 찾아서 ①②③에게 다음 방향을 준다. 제출과 실험표도 관리한다.

## 오늘 할 일
1) 7B 제로샷 홀드아웃 예측 저장 (0번 셀 MODEL_SIZE="7b" 확인, 노트북 0~8번 실행 후, 8번 셀 아래 새 셀):
     d = report_by_qtype(valid_df, zs_probs, "zeroshot")
     d["pred"] = probs_to_letters(zs_probs)
     d.to_csv(f"{OUT_DIR}/hold_zeroshot_pred.csv", index=False)

2) 오답을 사진과 함께 하나씩 보기:
     from IPython.display import display
     wrong = d[~d["correct"]].reset_index(drop=True)
     print(len(wrong), "개 오답")
     def show(i):
         r = wrong.iloc[i]
         im = Image.open(r["path"]).convert("RGB"); im.thumbnail((720, 720)); display(im)
         print(f"[{r['qtype']}] {r['question']}")
         for l in LETTERS:
             tag = ("  ← 정답" if l == r["answer"] else "") + ("  ← 예측" if l == r["pred"] else "")
             print(f"  {l}) {r[l]}{tag}")
     show(0)

3) 7B 제로샷 오답 44개를 아래 원인 코드로 분류해 CSV(id, qtype, 원인코드, 메모)로 남기기.
   메뉴·상품 오답(모델을 키워도 0.783 그대로)부터 먼저 볼 것:
   A. 글자가 작거나 흐려서 못 읽음 → 해상도·crop 문제 (②③에게)
   B. 읽을 수 있는데 비슷한 보기와 혼동 → OCR 매칭 문제 (③에게)
   C. "아닌 것/옳지 않은" 같은 부정 질문을 놓침 → 프롬프트 문제
   D. 위치·방향·공간 추론 실패 → 모델 능력 문제 (①에게)
   E. 사진만으로 답하기 어려움/정답 라벨이 의심스러움 → 기록만
   원인별 개수와 대표 예시 2개씩을 정리해줘.

4) 부정형 정확도 제대로 재기 (홀드아웃엔 9개뿐이라 불확실):
     neg = train_all[train_all["question"].apply(qtype) == "부정형"]
     p = predict_probs(neg, desc="neg")
     print(len(neg), "개 부정형 제로샷 정확도:", acc_of(p, neg))
   ※ 학습하지 않은 제로샷 모델에서만 의미 있음. 학습 후엔 train을 본 모델이라 부풀려진다.

5) 제출 관리
   - 팀원이 넘긴 제출 파일은 홀드아웃이 기존 최고보다 높을 때만 제출
   - EXPERIMENTS.md에 (TAG, 홀드아웃, 리더보드) 기록. 홀드아웃과 리더보드 차이를 계속 추적
   - 하루 20회는 팀 전체 공유

6) 디스커션 글(점수 있음): docs/EDA.md 내용으로 "이번 대회는 OCR 문제다" 인사이트 글 초안

## 주의
- 분석은 "개수"로 말해야 설득력이 있다. "가격을 잘 틀린다"가 아니라 "오답 44개 중 A가 N개".
- T4는 bf16이 없어 노트북이 자동으로 fp16으로 돈다. 추론이 A100보다 느린 건 정상.

## 오늘 17:30까지 산출물
- 오답 원인 코드별 개수 표 + 대표 예시
- 부정형 275개 제로샷 정확도
- 디스커션 글 초안

오답 분류를 같이 해보자. 내가 show(i) 출력을 붙여넣으면 원인 코드를 같이 판단해줘.
```

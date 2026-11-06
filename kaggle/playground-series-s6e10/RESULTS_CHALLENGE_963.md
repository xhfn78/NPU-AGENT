# 다른 조합을 사용한 0.963 목표 실험

**2026-10-28 목표 확인:** 사용자의 목표는 Kaggle 리더보드 0.963입니다.
최근 사용자 제공 리더보드 점수는 0.96068이며 제출 파일명은 확인되지 않았습니다.
아래 2026-10-27 결과는 로컬 검증 기록입니다. 이 수치로 리더보드 목표 달성을 판정하지 않습니다.

2026-10-27 실행. **전체 검증 데이터에서 ROC-AUC 0.963에는 아직 도달하지 못했습니다.**
아래 점수는 로컬 검증 점수입니다. Kaggle에는 제출하지 않았으며 리더보드 점수는 알 수 없습니다.

## 비교 조건

- 대회 train 699,635행 중 seed=42로 층화 분할한 139,927행을 동일하게 평가했습니다.
- 기존 기준 조합은 LightGBM 75% + 이전 CatBoost/XGBoost 조합 25%의 logit 평균입니다.
- 기존 기준: seed42 **0.960446358**, seed2026 **0.961928894**.
- 경쟁 후보 선택에 seed42를 반복 사용했습니다. seed2026도 앞선 실험에서 사용한 추가 분할입니다.
  따라서 두 분할 모두 독립적인 최종 테스트로 해석하지 않습니다.
- 원본 CSV를 바꾸지 않았습니다. 대회 타깃 인코딩은 해당 학습 부분 안에서 교차 계산합니다.
- 원본 데이터 기반 특징은 공개 항공사 만족도 데이터의 레이블만 사용합니다.
  대회 test의 숨겨진 정답이나 다른 참가자의 예측 파일은 사용하지 않았습니다.

## 최종 선택과 추가 분할 확인

기존 조합 80%와 Keras Attention 20%의 **확률 평균**을 선택했습니다.

| 평가 | 기존 기준 | 새 조합 | 차이 |
|---|---:|---:|---:|
| seed42, 모델 선택용 | 0.960446358 | **0.960497232** | +0.000050874 |
| seed2026, 설정을 고정한 추가 확인 | 0.961928894 | **0.961986166** | +0.000057272 |

추가 확인에서 Attention 단독은 0.961368496이었습니다. 가중치와 21 epoch는
추가 분할 평가 전에 고정했습니다. 두 분할의 개선 폭은 작으며 반복한 탐색을 감안하면
Kaggle에서도 반드시 개선된다고 보장할 수 없습니다. 목표 달성 여부는 두 분할 모두 false입니다.

전체 699,635행 재학습을 완료하고 제출 산출물을 `outputs/challenge_963/`에 저장했습니다.
이전 모델의 전체 학습 예측은 동일한 데이터 해시를 확인하고 재사용합니다.
Attention은 전체 학습 데이터를 사용해 21 epoch로 다시 학습했습니다.

최종 검사를 통과했습니다. CSV는 299,844행이며 ID 순서와 고유성이 일치하고 결측값은 0개입니다.
확률 범위는 0.010978605~0.992903753, 저장된 구성 모델의 예측을 다시 결합한 값과
CSV의 최대 오차는 1.11e-16입니다. Keras H5 재로딩 예측 오차는 0이었습니다.
대회 train/test 및 공개 원본 ZIP의 해시도 앞선 실험 기록과 일치했습니다.

제출 파일 SHA-256:
`f21de3a51aaa4b7c8ab1b6c3dc12a927ca5cc4096caf96cdd2bc9d5cc3b9dd8a`.

주요 최종 산출물:

- `outputs/challenge_963/submission_challenge.csv`: 새 제출 파일, `id,satisfaction`.
- `outputs/challenge_963/confirmation.json`: 설정을 고정한 추가 분할 AUC.
- `outputs/challenge_963/verification.json`: ID·확률·CSV 내보내기 확인.
- `outputs/challenge_963/attention_full/model.h5`: 전체 학습한 Keras 모델.
- `outputs/challenge_963/attention_full/metrics.json`: 21 epoch와 H5 재로딩 오차.
- `outputs/challenge_963/full_preprocessing.joblib`: 특징 인코딩·범주 매핑.
- `outputs/challenge_963/attention_full/dense_preprocessing.joblib`: 신경망 전처리.

기존 조합을 구성하는 트리 모델과 원본 보조 모델은 이전 출력 폴더의 파일을 이용합니다.
H5 모델의 사용자 정의 층은 `keras_attention.py`에 등록되어 있습니다.

## 새로 완료한 후보

| 후보 | seed42 전체 검증 ROC-AUC |
|---|---:|
| 기존 기준 조합 | 0.960446358 |
| 여행 유형별 LightGBM 전문 모델 | 0.960266779 |
| 고객 충성도별 LightGBM 전문 모델 | 0.960282001 |
| 좌석 등급별 LightGBM 전문 모델 | 0.960142834 |
| 원본 이웃 13개 특징 + LightGBM, 63 leaf | 0.960166050 |
| 원본 이웃 13개 특징 + LightGBM, 31 leaf | 0.960149934 |
| 원본 모델 확률을 정답에 10% 혼합 | 0.960218455 |
| 원본 모델 확률을 정답에 30% 혼합 | 0.960000590 |
| Keras Attention, 21 epoch | 0.959799393 |
| TabICLv2, 학습 문맥 100,000행, 1 member | 0.958985024 |
| TabICLv2, 학습 문맥 50,000행, 4 member | 0.958808019 |
| TabICLv2, 학습 문맥 400,000행, 1 member | 0.959245821 |
| TabICLv2 미세조정, 문맥 100,000행 | 0.959009308 |
| 더 큰 LightGBM, 127 leaf | 0.960271324 |
| 더 큰 LightGBM, 255 leaf | 0.960268014 |
| 무작위 분할을 추가한 LightGBM Extra Trees | 0.960309229 |
| 기존 기준 80% + Attention 20%, 확률 평균 | **0.960497232** |

단독 모델 점수가 낮아도 오류가 다른지 확인하기 위해 확률 평균, logit 평균,
순위 평균을 비교했습니다. 단독/2개/3개 후보의 선택 내역은
`outputs/challenge_963/selection.json`에 있습니다.
이 점수에 맞춘 추가 가중치 탐색은 목표 달성의 증거가 아닙니다.
추가 트리 실험을 포함한 선택은 `selection_with_trees.json`에 별도로 저장해
확인 중인 기존 조합의 선택 기록을 보존했습니다.
추가한 3개 트리 후보를 포함해도 선택된 모델과 가중치는 동일했습니다.

## 스태킹과 잔차 교정

LightGBM, CatBoost, Keras MLP, 이전 0.960 조합의 예측 네 개를 사용했습니다.
각 예측의 확률, logit, p(1-p)에 좌석 등급·여행 유형·고객 유형 조건을 추가하고,
규제 로지스틱 회귀로 결합 가중치를 학습합니다.

- 5-fold 메타 교차 예측 최고: **0.960521092**, 조건 포함, C=0.1.
- 조건을 제외한 메타 교차 예측: 0.960504266, C=0.01.
- LightGBM으로 기존 logit의 잔차를 교정한 메타 교차 예측: 0.960482347, 100 trees.

메타 교차 예측은 각 행의 정답을 그 행의 메타 모델 학습에서 제외합니다.
다만 기본 후보와 C의 선택에 이 분할을 사용했으므로 이 수치는 탐색용입니다.

별도 스태킹 확인은 seed2026의 학습 부분을 다시 나눕니다.
기본 모델은 447,766행으로 학습한 뒤 111,942행을 예측하고, 그 예측과 정답으로
메타 모델을 학습합니다. 이후 559,708행으로 기본 모델을 학습해 외부 검증
139,927행을 예측합니다. 외부 검증의 정답은 기본·메타 모델 학습에 사용하지 않습니다.
설정과 epoch/트리 수는 확인 전에 고정합니다.

이 방식의 실제 추가 분할 AUC는 **0.961884735**였습니다.
기존 기준 0.961928894보다 **0.000044159 낮아** 제출 모델로 채택하지 않았습니다.
탐색용 메타 교차 예측 점수만 보고 개선됐다고 결론 내리지 않았습니다.
기록은 `outputs/challenge_963/stacking/confirmation.json`에 있습니다.

## TabICLv2 실행 환경과 메모리

기존 ML 환경과 별도로 프로젝트 안에 `.tfm_venv`를 만들었습니다.
PyTorch 2.8.0+cu128, TabICL 2.2.0, transformers 4.57.1을 사용했습니다.
공식 체크포인트는 약 110MB이며 메타데이터는 `data/tabicl_model/source.json`에 있습니다.
공식 PyTorch GPU 패키지 다운로드 크기는 약 3.5GB입니다.

체크포인트 SHA-256:
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.

RTX 4060 Laptop 8GB에서 400,000행 실험은 처음 FFN 계산의 메모리 부족으로 중단됐습니다.
추론 시 행별 FFN 계산을 16,384행씩 나눠 동일한 층 연산을 수행한 재실행은 완료됐습니다.
12개 ICL 층의 작은 입력 비교에서 최대 오차 0을 확인했습니다.
최종 전체 검증 실행의 PyTorch 최고 할당량은 7,363MiB, 실행 시간은 약 308초였습니다.
학습된 가중치와 attention 연산은 바꾸지 않았고 패키지 설치 파일도 수정하지 않았습니다.

미세조정의 내부 검증 점수 0.962469는 대회 전체 외부 검증 점수가 아닙니다.
이 값을 0.963 목표 달성으로 계산하지 않습니다. 20,000행 예비 평가도 전체 검증 점수와 구별했습니다.

## 코드와 재현

현재 설치된 환경과 이전 실험 산출물을 이용하는 PowerShell 명령입니다.

```powershell
cd C:\study\kaggle\playground-series-s6e10
$ml = 'C:\Users\Admin\miniconda3\envs\ml311\python.exe'
& $ml challenge_models.py experts
& $ml original_neighbors.py
& $ml challenge_models.py neighbors
& $ml challenge_models.py soft
& $ml challenge_models.py trees
& $ml challenge_models.py attention
& $ml blend_diagnostics.py
& $ml residual_stack.py
& $ml challenge_models.py select
& $ml challenge_models.py select --selection-file selection_with_trees.json
& $ml stacking_challenge.py confirm
& $ml challenge_models.py confirm
& $ml challenge_models.py refit --selection-file selection_with_trees.json
```

Keras 코드는 기존 `tf29x-gpu` 환경을 하위 프로세스로 실행합니다.
TabICL을 새 환경에 설치할 때는 기본 ML 의존성과 호환되는 Python 3.11 환경을 먼저 준비합니다.
현재 `.tfm_venv`는 `ml311`의 기본 패키지를 공유합니다.

```powershell
& $ml -m venv --system-site-packages .tfm_venv
& .\.tfm_venv\Scripts\python.exe -m pip install --upgrade pip
& .\.tfm_venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& .\.tfm_venv\Scripts\python.exe -m pip install -r requirements_tfm.txt
& .\.tfm_venv\Scripts\python.exe fetch_tabicl.py
& .\.tfm_venv\Scripts\python.exe tabicl_candidate.py --phase validation --context 100000 --members 1 --cache repr --predict-batch 10000
& .\.tfm_venv\Scripts\python.exe tabicl_candidate.py --phase validation --context 50000 --members 4 --cache repr --predict-batch 50000
& .\.tfm_venv\Scripts\python.exe tabicl_candidate.py --phase validation --context 400000 --members 1 --cache repr --offload cpu --ffn-chunk 16384 --predict-batch 139927
& .\.tfm_venv\Scripts\python.exe finetune_tabicl.py --phase validation --context 100000 --epochs 3
```

`challenge_models.py`는 완료한 실험을 해당 이름의 캐시에서 재사용합니다.
다른 설정을 비교할 때는 기존 이름/산출물에 덮어쓰지 말고 새 실험 이름과 폴더를 사용하세요.
TabICL 명령은 같은 문맥 수와 member 수의 산출물을 덮어씁니다.
재현에는 앞선 `improve_960_gpu`, `improve_963_gpu` 결과 및 공개 원본 데이터가 필요합니다.
확률 배열만 다른 데이터셋에 옮겨 재사용하지 않도록 원본 해시를 확인합니다.

주요 파일은 `challenge_models.py`, `keras_attention.py`, `original_neighbors.py`,
`tabicl_candidate.py`, `finetune_tabicl.py`, `stacking_challenge.py`입니다.
실험 기록은 `outputs/challenge_963/`에 저장합니다.

## 출처

- [TabICL 공식 저장소](https://github.com/soda-inria/tabicl)
- [공식 체크포인트](https://huggingface.co/jingang/TabICL)
- [LightGBM의 cross_entropy 설정](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [공개 원본 항공사 만족도 데이터](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

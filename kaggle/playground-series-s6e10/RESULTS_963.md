# 0.963 목표 후속 실험

2026-10-27 실험입니다. **이번 실험에서 검증 ROC-AUC 0.963에는 도달하지 못했습니다.**
선택한 모델은 기존과 같은 분할에서 0.960446358, 설정을 고정한 추가 분할에서
0.961928894를 기록했습니다. 폴더 이름의 963은 달성 점수가 아니라 실험 목표입니다.

## 이전 결과와 비교

| 평가 | 이전 모델 | 새 선택 모델 | 차이 |
|---|---:|---:|---:|
| seed=42, 검증 20% | 0.960013249 | 0.960446358 | +0.000433109 |
| seed=2026, 검증 20% | 0.961432061 | 0.961928894 | +0.000496833 |

seed=42 점수는 모델 선택에 사용했습니다. 추가 분할은 새 모델의 설정, 트리 수,
결합 가중치를 고정하고 평가했지만, 동일 데이터셋의 다른 무작위 분할이며
독립 테스트나 Kaggle 리더보드 점수는 아닙니다.

## 무엇을 추가했는가

1. 공개 원본 항공사 만족도 데이터 129,880행으로 XGBoost와 CatBoost를 각각 학습했습니다.
   원본 데이터 자체의 검증 AUC는 약 0.9947이지만, 이는 대회 데이터의 점수가 아닙니다.
2. 원본 모델이 대회 각 행에 예측한 만족 확률과 logit을 보조 특징으로 추가했습니다.
3. 원본 데이터의 거리, 나이, 서비스 평가, 거리/좌석 등급별 타깃 통계도 비교했습니다.
4. LightGBM과 Keras 신경망을 추가해 모델 구조가 다른 예측의 결합을 비교했습니다.

원본 데이터의 `train.csv`와 `test.csv`는 모두 공개 정답이 있는 별개의 2020년 데이터입니다.
대회의 숨겨진 test 정답을 가져온 것이 아닙니다. 원본에만 있는 `Inflight service`,
id, 행 번호는 보조 모델 입력에서 제외하고, 대회와 공통인 21개 입력 변수만 사용했습니다.
대회 레이블을 이용하는 인코딩은 각 학습 부분 안에서 5-fold 교차 계산했습니다.

## 같은 분할의 완료 후보

| 후보 | 검증 ROC-AUC |
|---|---:|
| 이전 0.960 모델 | 0.960013249 |
| 원본 예측을 추가한 XGBoost | 0.960243898 |
| 원본 통계까지 추가한 XGBoost | 0.960300954 |
| 원본 예측을 추가한 CatBoost | 0.960086601 |
| 원본 통계까지 추가한 CatBoost | 0.960213380 |
| LightGBM, 31개 leaf | 0.960256559 |
| LightGBM, 63개 leaf | 0.960334360 |
| Keras 신경망 | 0.959250808 |

Keras 신경망은 학습 AUC가 0.963을 넘었지만 검증 성능이 낮았습니다.
학습 점수를 목표 달성으로 계산하지 않았고, 최종 제출 모델에도 포함하지 않았습니다.

결합 후보 중 가장 높은 관측 AUC는 0.960480488이었습니다.
최고값과 0.00005 이내에서는 신경망과 결합 모델 수가 적은 조합을 우선하는 규칙을 적용해
0.960446358의 조합을 선택했습니다. 두 값의 차이는 약 0.000034입니다.

## 최종 선택

- 새 LightGBM: 75%
- 이전 CatBoost/XGBoost 결합 모델: 25%
- 확률을 logit으로 바꿔 가중 평균한 뒤 sigmoid로 되돌림
- LightGBM: num_leaves=63, learning_rate=0.035, min_child_samples=150,
  reg_lambda=15, max_bin=255, 선택한 트리 수 529개

추가 분할에서 새 LightGBM 단독은 0.961846870,
고정한 결합 모델은 0.961928894였습니다.

## 실행과 산출물

주 실행 파일은 `improve_963.py`이고, Keras 비교 코드는 `keras_candidate.py`입니다.
현재 환경에서는 다음 순서로 실행합니다.

```powershell
cd C:\study\kaggle\playground-series-s6e10
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_963.py --phase prepare
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_963.py --phase tune
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_963.py --phase select
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_963.py --phase confirm
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_963.py --phase refit
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_963.py --phase verify
```

완료한 후보는 캐시에서 읽습니다. 기본 후보는 최종 선택에 사용한 `lgb_original_deep`입니다.
다른 후보를 재실험하려면 `--candidates`로 이름을 지정합니다.
Keras 비교는 `--candidates keras_original_prior`이며, 이 경우 기존 `tf29x-gpu` 환경의
TensorFlow 2.9.3을 하위 프로세스로 실행합니다. 최종 제출 모델에는 TensorFlow가 필요하지 않습니다.

현재 폴더에는 원본 데이터와 `outputs/improve_960_gpu/`의 이전 결과가 준비되어 있습니다.
이전 모델의 분할 및 예측 파일도 이번 코드의 재현에 필요합니다.
LightGBM 4.7.0은 프로젝트의 `.local_packages/`에 설치해 기존 Python 환경을 바꾸지 않았습니다.
다른 환경에서 설치할 때의 의존성 목록은 `requirements_963.txt`에 있습니다.

`outputs/improve_963_gpu/`의 주요 파일:

- `submission_candidate.csv`: 전체 데이터 재학습 후 생성한 제출 파일
- `selection.json`: 목표 달성 여부, 최고 관측값, 실제 선택, 후보별 점수
- `confirmation.json`: 고정한 설정으로 평가한 추가 분할 점수
- `verification.json`: 제출 형식, id, 확률 범위, CSV 저장 오차 확인
- `*_reload_check.json`: 새 선택 모델의 저장/재로딩 예측 비교
- `*_full_model.*`, `*_full_preprocessing.joblib`: 새 선택 모델과 전처리
- `original_*`: 원본 데이터에서 학습한 보조 모델과 입력 특징
- `data_manifest.json`: 대회 train/test와 원본 ZIP의 SHA-256

이전 0.960 제출 파일과 모델은 기존 폴더에 보존했습니다.

## 출처

공개 원본 데이터:
[teejmahal20의 Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).
ZIP은 Kaggle의 공개 데이터 다운로드 API에서 내려받았으며 파일 해시를 기록했습니다.
대회 평가 방식은
[공식 대회 설명](https://www.kaggle.com/competitions/playground-series-s6e10/overview/about-the-tabular-playground-series)을 따릅니다.
LightGBM 학습 방식은
[공식 LGBMClassifier 문서](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)를 참고했습니다.

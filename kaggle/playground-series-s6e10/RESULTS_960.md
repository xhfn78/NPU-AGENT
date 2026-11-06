# ROC-AUC 개선 실험 결과

2026-10-27, 학습 데이터 699,635행과 테스트 데이터 299,844행을 사용했습니다.
추가 외부 데이터 없이 로컬 CSV만 사용했습니다. 입력 변수는 id와 정답을 제외한 21개입니다.

## 같은 검증 분할에서의 비교

모든 점수는 seed=42, stratify=y, 검증 비율 20%의 동일한 행 분할입니다.

| 모델 | 검증 ROC-AUC |
|---|---:|
| 기존 CatBoost | 0.957516547 |
| 기존 XGBoost | 0.957830801 |
| XGBoost + 원본 21열 교차 타깃 인코딩 | 0.958855997 |
| XGBoost + 여행 조건별 교차 타깃 인코딩 | 0.959259742 |
| CatBoost + 서비스/여행 조건 교차 인코딩 | 0.959762038 |
| XGBoost + 서비스/여행 조건 교차 인코딩 + 거리/좌석 등급별 통계 | 0.959832464 |
| 최종 50:50 logit 평균 | **0.960013249** |

기존 XGBoost 대비 약 +0.00218245입니다.
정확한 원시 점수와 모든 완료 후보의 설정은 각 출력 폴더의 JSON에 기록했습니다.
상기 마지막 자리의 표시는 소수점 반올림이며 원시 JSON을 기준으로 확인하세요.

## 최종 설정

- CatBoost: depth=8, learning_rate=0.07, l2_leaf_reg=10, 선택한 트리 수 546개
- XGBoost: max_depth=6, learning_rate=0.04, min_child_weight=30, reg_lambda=15,
  subsample=0.9, colsample_bytree=0.9, max_bin=512, 선택한 트리 수 1,060개
- 각 모델의 확률을 logit으로 변환해 0.5씩 평균한 뒤 sigmoid로 되돌림
- 두 모델 모두 NVIDIA RTX 4060 Laptop GPU에서 학습

타깃 인코딩은 각 학습 부분 안에서 StratifiedKFold 5-fold의 `fit_transform()`으로 계산했습니다.
외부 검증 행과 테스트 행은 학습 부분에서 만든 인코더의 `transform()`만 사용합니다.
거리/좌석 등급별 통계 역시 해당 학습 부분에서만 계산합니다.
최종 제출용 재학습에서도 학습 행의 타깃 인코딩은 5-fold 교차 인코딩을 사용합니다.

## 고정 설정의 추가 분할 확인

모델 설정, 트리 수, 결합 가중치를 먼저 고정한 뒤,
seed=2026의 80:20 계층 분할에서 조기 종료나 가중치 재선택 없이 평가했습니다.

| 모델 | 추가 분할 ROC-AUC |
|---|---:|
| CatBoost | 0.961222338 |
| XGBoost | 0.961233133 |
| 고정한 50:50 logit 평균 | **0.961432061** |

이는 동일 데이터셋의 추가 무작위 분할입니다. 별개의 독립 테스트 데이터는 아닙니다.
seed=42 점수는 여러 모델과 결합을 선택하는 데 사용해 낙관적일 수 있습니다.
Kaggle 점수는 실제 제출 후 확인해야 하며, 여기의 검증 점수로 보장할 수 없습니다.

## 산출물과 재현

`improve_auc.py`의 기본 후보는 최종 두 모델입니다.
`--phase tune`, `--phase check`, `--phase refit`, `--phase verify` 순서로 실행합니다.
`README.md`에 현재 환경에서 실행할 전체 명령이 있습니다.

산출물은 `outputs/improve_960_gpu/`에 저장합니다.

- `submission_improved.csv`: 전체 학습 데이터로 재학습한 두 모델의 테스트 예측
- `selection.json`: 검증 분할, 후보별 점수, 최종 결합 방식과 가중치
- `second_split_check.json`: 고정된 설정의 추가 분할 점수
- `verification.json`: 제출 파일의 행 수/id/확률 범위 및 모델 재로딩 확인 결과
- `*_full_model.*`, `*_full_encoder.joblib`, `*_full_profiles.joblib`, `*_schema.json`: 모델과 전처리
- `data_manifest.json`: 원본 train.csv의 SHA-256과 분할 설정

이전 기준 모델과 제출 파일은 각각 기존 출력 폴더에 남아 있습니다.

제출 파일 검증을 완료했습니다. 299,844행의 id 순서가 test.csv와 일치하고,
결측 예측은 없으며 확률은 0.002544~0.996745 범위입니다.
저장 모델과 전처리를 다시 불러와 첫 1,000행을 예측했을 때 제출 파일과의
최대 절대 오차는 약 3.32e-7로, 허용 오차 1e-6 이내였습니다.

아이디어 참고:
[scikit-learn TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html),
[해당 대회의 공개 교차 인코딩/거리별 통계 실험](https://github.com/alvis1111/s6e10-airline-solution).
코드와 위 점수는 이 작업 공간에서 구현하고 실행해 얻었습니다.

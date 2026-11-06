# 항공사 만족도: CatBoost / XGBoost 기준 모델

`catboost_baseline.py`를 CSV 세 개와 같은 폴더에 두고 실행합니다.
코드의 한국어 주석은 Keras의 학습 흐름과 연결해 설명합니다.

## 설치와 실행

Python 환경을 활성화한 터미널에서 다음 명령을 실행합니다.

```powershell
cd C:\study\kaggle\playground-series-s6e10
python -m pip install -r requirements.txt
python catboost_baseline.py
```

설치 중 Python 버전을 지원하는 패키지가 없다는 오류가 나면 Python 3.12 환경을 사용하세요.
VS Code에서는 `# %%`로 구분한 셀을 순서대로 실행할 수도 있습니다.
Jupyter에 복사해 실행할 경우 CSV가 있는 폴더에서 노트북을 여세요.

## Keras와 연결해서 읽기

| 익숙한 Keras 개념 | 이 코드에서의 역할 |
|---|---|
| 모델 생성과 `compile()` | `CatBoostClassifier(...)`에 손실과 평가 지표 설정 |
| `model.fit()` | `model.fit(X_train, y_train, ...)` |
| `validation_data` | `eval_set=(X_valid, y_valid)` |
| `EarlyStopping` | `early_stopping_rounds=100`과 `use_best_model=True` |
| sigmoid 출력 확률 | `model.predict_proba(X)[:, 1]` |

CatBoost의 `predict()`는 기본적으로 클래스 라벨을 반환합니다.
이 대회에서는 확률이 필요하므로 `predict_proba()`를 사용합니다.
`iterations`는 트리를 추가하는 횟수이며, 신경망의 `epochs`와 같은 개념은 아닙니다.

## 검증과 출력

만족/불만족 비율을 유지해 학습 데이터의 20%를 검증에 사용합니다.
검증 ROC-AUC가 가장 좋았던 트리 수를 고른 뒤, 전체 학습 데이터로 제출용 모델을 다시 학습합니다.
출력되는 검증 점수는 재학습 전 모델의 점수이며, Kaggle 점수와는 다를 수 있습니다.

실행이 끝나면 `outputs/catboost_baseline/`에 다음 파일이 생깁니다.

- `submission_catboost.csv`: Kaggle에 업로드할 `id,satisfaction` 형식의 제출 파일
- `model.cbm`: 전체 데이터로 학습한 제출용 모델
- `metrics.json`: 검증 점수와 설정
- `validation_feature_importance.csv`: 검증 단계 모델의 변수 중요도

같은 코드로 재실행하면 위 출력 파일을 갱신합니다.
빠르게 코드를 확인하려면 `MAX_ITERATIONS`를 50으로 줄이고 `REFIT_ALL_DATA = False`로 설정할 수 있습니다.
성능 비교에는 동일한 검증 분할과 설정을 유지하세요. 기준 모델 다음 단계는 교차검증입니다.

## XGBoost 실행과 비교

`xgboost_baseline.py`는 CatBoost와 같은 원본 CSV, seed=42, 검증 비율 20%,
타깃 비율을 유지하는 분할을 사용합니다. 기본 Python 환경 대신 필요한 패키지가
설치된 `ml311` 환경에서 실행할 수 있습니다.

```powershell
cd C:\study\kaggle\playground-series-s6e10
& C:\Users\Admin\miniconda3\envs\ml311\python.exe xgboost_baseline.py
```

다른 Python 환경에서는 위의 `requirements.txt`를 설치한 뒤
`python xgboost_baseline.py`로 실행하면 됩니다.

CatBoost는 문자열을 직접 받지만, XGBoost는 네 범주형 변수를 `category`로 변환하고
`enable_categorical=True`로 학습합니다. 수치형 결측치는 그대로 전달합니다.
최적 트리 수를 선택한 뒤 전체 데이터로 재학습하며,
`outputs/xgboost_baseline/`에 제출 파일 `submission_xgboost.csv`, 모델 `model.json`,
검증 점수 `metrics.json`, 변수 중요도, 범주 매핑 `feature_schema.json`을 저장합니다.
JSON 모델을 다시 불러와 예측할 때도 범주 매핑과 입력 열 순서를 맞춰야 합니다.

기존 CatBoost 점수가 저장되어 있으면 XGBoost 실행 마지막에 AUC 차이도 출력합니다.
이 비교는 동일한 데이터와 검증 분할을 전제로 합니다. 두 모델의 기본 설정 비교이며
동일한 계산량의 비교나 튜닝된 모델의 우열을 의미하지 않습니다.
두 모델의 변수 중요도는 정의와 척도가 달라 수치를 직접 비교하지 않습니다.

XGBoost 참고: [학습과 조기 종료](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html),
[범주형 변수 처리](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).

## ROC-AUC 0.960 개선 실험

`improve_auc.py`에서 교차 타깃 인코딩과 두 모델의 결합을 실험했습니다.
기존과 같은 seed=42의 80:20 분할에서 **0.960013**을 기록했고,
설정·트리 수·결합 가중치를 고정한 뒤 seed=2026 분할에서도 **0.961432**를 확인했습니다.
첫 점수는 모델 선택에 사용한 점수입니다. 두 번째도 동일한 데이터셋의 추가 무작위 분할이며,
독립 테스트 점수나 Kaggle 리더보드 점수는 아닙니다.

최종 조합은 `cat_service_gpu`와 `xgb_service_routes_gpu`를 50:50으로 결합합니다.
각 모델의 확률을 logit으로 바꾸어 평균한 뒤 sigmoid로 확률로 되돌립니다.
학습 데이터만 사용해 만든 5-fold 교차 타깃 인코딩으로
비행 거리, 여행 조건, 조건별 서비스 평점의 관계를 반영합니다.
XGBoost에는 같은 거리·좌석 등급의 서비스 평점 평균과 표본 수를 추가했습니다.
외부 데이터나 다른 참가자의 예측 파일은 사용하지 않았습니다.

RTX 4060이 있는 현재 `ml311` 환경에서 다음 순서로 재현할 수 있습니다.
타깃 인코딩의 CV splitter 사용을 위해 scikit-learn 1.9 이상이 필요합니다.

```powershell
cd C:\study\kaggle\playground-series-s6e10
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_auc.py --phase tune
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_auc.py --phase check
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_auc.py --phase refit
& C:\Users\Admin\miniconda3\envs\ml311\python.exe improve_auc.py --phase verify
```

같은 원본과 설정으로 완료한 실험은 재사용합니다.
처음부터 다른 폴더에서 실험하려면 모든 명령에 동일한
`--output outputs/my_new_run`을 지정하세요.
GPU 학습에는 작은 수치 재현 차이가 생길 수 있습니다.

提出ファイルは `outputs/improve_960_gpu/submission_improved.csv` です。
`selection.json`に候補の点数と選択したモデル・重み、`second_split_check.json`に追加検証結果、
`verification.json`に提出形式と保存モデルの再読み込み検証結果を保存します。
学習済みモデルと前処理も同じフォルダに保存します。

詳しい結果は `RESULTS_960.md` を参照してください。
전처리 참고: [TargetEncoder의 교차 인코딩](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html).

## 0.963 목표 후속 실험

공개 원본 데이터의 보조 예측, LightGBM, Keras를 추가 비교한 결과는
`RESULTS_963.md`에 있습니다. **0.963에는 아직 도달하지 못했습니다.**
새 선택 모델의 같은 분할 AUC는 0.960446, 추가 분할은 0.961929입니다.

주 코드는 `improve_963.py`, Keras 비교 코드는 `keras_candidate.py`입니다.
새 제출 파일은 `outputs/improve_963_gpu/submission_candidate.csv`에 저장합니다.
이전 제출 파일과 모델은 보존했습니다.

참고: [CatBoost fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit),
[확률 예측](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict_proba),
[공식 대회 설명](https://www.kaggle.com/competitions/playground-series-s6e10/overview/about-the-tabular-playground-series).

## 실제 목표와 리더보드 기준

최신 트리 수/학습 시드 실험은 `xgboost_1000.py`와 `RESULTS_XGBOOST_1000.md`에 있습니다.
고정 분할·시드 42에서 실제 1,000개까지 학습하고 786/800/900/1,000개를 비교합니다.
이후 XGBoost에서 학습 시드 1~41을 검증하며, 이 단계의 점수도 로컬 검증 점수입니다.
실제로 1~41 전체 검증을 완료했습니다. 1,000개 모델 중 최고는 시드 20의 0.960397,
41개 확률 평균은 0.960392입니다. 확인한 트리 수까지 포함한 최고 관측값은
800개·시드 20의 0.960411이며, 모두 같은 로컬 검증 데이터에서 선택한 값입니다.

2026-10-28 확인: 목표는 **Kaggle 리더보드 ROC-AUC 0.963**입니다.
사용자가 알려준 최근 리더보드 점수는 **0.96068**이며, 해당 제출 파일명은 확인되지 않았습니다.
아래 기존 실험의 0.960497/0.961986은 로컬 점수이므로 리더보드 달성 점수로 해석하지 않습니다.
사용자 제공 점수는 `leaderboard_progress.json`에 기록합니다.

`leaderboard_cv.py`는 같은 5-fold에서 LightGBM/XGBoost/CatBoost의 OOF 예측을 만들고
폴드별 테스트 예측을 평균합니다. OOF로 조합을 고르고
`outputs/leaderboard_cv_20261028/submission_cv.csv`를 출력합니다.
이 파일의 리더보드 목표 달성 여부는 실제 Kaggle 평가 전까지 미확인입니다.
완료한 전체 OOF는 0.961271이며 세 모델 동일 비중의 logit 평균을 선택했습니다.
같은 OOF의 최고 단독 모델은 0.961151이고, 결합 모델은 5개 폴드 모두에서 단독 모델보다 높았습니다.
실험 결과와 제출 검증은 `RESULTS_LEADERBOARD_CV.md`에 있습니다.

```powershell
cd C:\study\kaggle\playground-series-s6e10
& C:\Users\Admin\miniconda3\envs\ml311\python.exe leaderboard_cv.py all
```

데이터·설정·코드 해시가 일치하는 완료 폴드만 재사용합니다.
추가 Python 패키지 설치는 필요하지 않으며, 기존 원본 보조 특징과 ML 환경을 이용합니다.

## 다른 조합을 사용한 추가 도전

조건별 전문 모델, 원본 데이터의 이웃 특징, 부드러운 정답, Keras Attention,
TabICLv2와 미세조정, 더 큰 LightGBM을 포함한 15개 후보 및 조합을 실제 비교했습니다.
실험 결과와 재현 명령은 `RESULTS_CHALLENGE_963.md`에 있습니다.
0.963에는 아직 도달하지 못했으며, 점수는 Kaggle 리더보드가 아닌 로컬 ROC-AUC입니다.

주 코드는 `challenge_models.py`, 신경망은 `keras_attention.py`,
학습/결합/검증을 분리한 스태킹 코드는 `stacking_challenge.py`입니다.
실험 산출물은 `outputs/challenge_963/`에 별도로 저장합니다.
선택한 조합은 기존 기준 모델 80% + Keras Attention 20%의 확률 평균입니다.
모델 선택 분할 AUC는 0.960497, 고정한 설정의 추가 분할은 0.961986입니다.
두 분할 모두 앞선 탐색에 사용한 데이터이므로 독립 테스트 점수로 보지 않습니다.

전체 재학습 제출 파일은 `outputs/challenge_963/submission_challenge.csv`로 출력합니다.
기존 제출 파일 `outputs/improve_963_gpu/submission_candidate.csv`도 보존했습니다.

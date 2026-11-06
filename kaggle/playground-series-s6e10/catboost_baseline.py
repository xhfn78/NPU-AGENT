# %% 1. 라이브러리와 설정
"""항공사 만족도 CatBoost 기준 모델.

실행: python catboost_baseline.py
VS Code에서는 # %% 단위로 셀을 실행할 수도 있습니다.
"""

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# 스크립트와 CSV를 같은 폴더에 둡니다.
# Jupyter에 복사해서 실행한다면 CSV 폴더에서 노트북을 열면 됩니다.
DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = DATA_DIR / "outputs" / "catboost_baseline"
SEED = 42
MAX_ITERATIONS = 1500
EARLY_STOPPING_ROUNDS = 100
REFIT_ALL_DATA = True


# %% 2. 데이터 읽기
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("train:", train.shape, "test:", test.shape)

# True(만족)를 1, False(불만족)를 0으로 바꿉니다.
target = train["satisfaction"].astype(str).str.strip().str.lower()
y = target.map({"true": 1, "false": 0})
if y.isna().any():
    raise ValueError("satisfaction에 True/False 이외의 값이 있습니다.")
y = y.astype("int8")

# id는 식별용으로 보관하고 학습 입력에서는 제외합니다.
X = train.drop(columns=["id", "satisfaction"]).copy()
X_test = test.drop(columns="id").copy()
if set(X.columns) != set(X_test.columns):
    raise ValueError("학습 데이터와 테스트 데이터의 입력 열이 다릅니다.")
X_test = X_test.loc[:, X.columns]


# %% 3. 범주형 변수 준비
cat_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print("범주형 변수:", cat_features)
print("만족 비율:", f"{y.mean():.2%}")

# CatBoost는 문자열 범주를 직접 처리하므로 원핫 인코딩이 필요 없습니다.
# 범주형의 결측치는 문자열로 바꾸고, 수치형 NaN은 그대로 둡니다.
# 트리 모델이므로 StandardScaler / MinMaxScaler도 필요 없습니다.
for column in cat_features:
    X[column] = X[column].astype("string").fillna("__MISSING__").astype(str)
    X_test[column] = X_test[column].astype("string").fillna("__MISSING__").astype(str)


# %% 4. 학습 / 검증 데이터 나누기
# stratify=y: 두 데이터에서 만족/불만족 비율을 비슷하게 유지합니다.
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)


# %% 5. 모델 만들기와 fit()
# Keras의 모델 생성 + compile()에 해당하는 설정을 생성자에 넣습니다.
# Logloss는 Keras의 binary_crossentropy와 같은 목적의 이진 분류 손실입니다.
# iterations는 추가할 트리 수의 상한입니다. Keras의 epochs와는 다릅니다.
params = dict(
    iterations=MAX_ITERATIONS,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=cat_features,
    random_seed=SEED,
    task_type="CPU",
    thread_count=8,
    allow_writing_files=False,
    verbose=100,
)
model = CatBoostClassifier(**params)

# eval_set은 Keras의 validation_data와 비슷한 역할을 합니다.
# 검증 AUC가 100회 동안 개선되지 않으면 학습을 멈춥니다.
# use_best_model=True는 검증 AUC가 가장 좋았던 지점의 모델을 남깁니 다.
model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    use_best_model=True,
)


# %% 6. 검증 성능 확인
# Keras의 sigmoid 출력처럼, 만족(1)일 확률을 가져옵니다.
# predict_proba()의 두 열은 각각 불만족(0), 만족(1)의 확률입니다.
valid_probability = model.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_probability)
best_iterations = model.get_best_iteration() + 1  # 반환 인덱스는 0부터 시작

print(f"Validation ROC-AUC: {valid_auc:.6f}")
print(f"Best iterations: {best_iterations}")
if best_iterations == MAX_ITERATIONS:
    print("최고 성능이 반복 상한에서 나왔습니다. 다음 실험에서는 상한을 늘려보세요.")

# 검증 성능은 전체 데이터 재학습 전에 측정한 점수입니다.
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
print(feature_importance.head(10).to_string(index=False))


# %% 7. 제출용 모델을 전체 학습 데이터로 다시 학습
# 검증에서 고른 트리 수를 고정한 뒤, 라벨이 있는 모든 행으로 학습합니다.
# 전체 데이터 재학습에는 별도의 검증 세트나 조기 종료를 사용하지 않습니다.
final_model = model
if REFIT_ALL_DATA:
    print("전체 학습 데이터로 제출용 모델을 학습합니다.")
    final_params = {**params, "iterations": best_iterations}
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X, y)


# %% 8. 테스트 확률 예측과 제출 파일 저장
test_probability = final_model.predict_proba(X_test)[:, 1]
submission = sample_submission.copy()
if not submission["id"].equals(test["id"]):
    raise ValueError("제출 양식의 id와 테스트 데이터의 id 순서가 다릅니다.")
if len(test_probability) != len(submission):
    raise ValueError("예측 개수와 제출 파일 행 수가 다릅니다.")

submission["satisfaction"] = test_probability
if not submission["satisfaction"].between(0, 1).all():
    raise ValueError("예측에 결측치 또는 0~1 범위 밖의 값이 있습니다.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
submission_path = OUTPUT_DIR / "submission_catboost.csv"
submission.to_csv(submission_path, index=False)
feature_importance.to_csv(OUTPUT_DIR / "validation_feature_importance.csv", index=False)
final_model.save_model(str(OUTPUT_DIR / "model.cbm"))

metrics = {
    "validation_roc_auc": float(valid_auc),
    "validation_size": 0.2,
    "seed": SEED,
    "best_iterations": int(best_iterations),
    "refit_all_data": REFIT_ALL_DATA,
    "parameters": params,
}
(OUTPUT_DIR / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(submission.head().to_string(index=False))
print("제출 파일:", submission_path)

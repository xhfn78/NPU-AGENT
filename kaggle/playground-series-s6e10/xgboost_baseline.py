# %% 1. 라이브러리와 설정
"""항공사 만족도 XGBoost 기준 모델. 실행: python xgboost_baseline.py"""

import json
from pathlib import Path

import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = DATA_DIR / "outputs" / "xgboost_baseline"
SEED = 42
MAX_ITERATIONS = 1500
EARLY_STOPPING_ROUNDS = 100
REFIT_ALL_DATA = True


# %% 2. 데이터 읽기와 입력 / 정답 분리
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
print("train:", train.shape, "test:", test.shape)

y = train["satisfaction"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
if y.isna().any():
    raise ValueError("satisfaction에 True/False 이외의 값이 있습니다.")
y = y.astype("int8")
X = train.drop(columns=["id", "satisfaction"]).copy()
X_test = test.drop(columns="id").copy()
if set(X.columns) != set(X_test.columns):
    raise ValueError("학습 데이터와 테스트 데이터의 입력 열이 다릅니다.")
X_test = X_test.loc[:, X.columns]


# %% 3. 문자열을 category 자료형으로 변환
# CatBoost에는 문자열을 직접 전달했지만, XGBoost에는 category로 전달합니다.
# 학습과 테스트에서 범주 순서를 동일하게 고정합니다.
# 테스트에만 있는 범주는 NaN으로 처리하며, 수치형 NaN도 모델이 처리합니다.
# 정답을 사용하지 않는 자료형 변환이므로 스케일링이나 타깃 인코딩은 없습니다.
cat_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
category_levels = {}
for column in cat_features:
    values = X[column].astype("string").fillna("__MISSING__")
    categories = sorted(set(values.tolist()) | {"__MISSING__"})
    category_levels[column] = categories
    category_type = pd.CategoricalDtype(categories=categories)
    X[column] = values.astype(category_type)
    X_test[column] = (
        X_test[column].astype("string").fillna("__MISSING__").astype(category_type)
    )
print("범주형 변수:", cat_features)
print("만족 비율:", f"{y.mean():.2%}")


# %% 4. CatBoost와 같은 검증 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)


# %% 5. 모델 생성과 fit()
# binary:logistic은 이진 분류용이며, predict_proba()로 만족 확률을 얻습니다.
# Keras의 compile()에 해당하는 설정을 생성자에 넣습니다.
# hist는 히스토그램 방식의 트리 학습, enable_categorical은 범주형 처리 설정입니다.
params = dict(
    n_estimators=MAX_ITERATIONS,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    device="cpu",
    n_jobs=8,
    random_state=SEED,
    importance_type="gain",
)
model = XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

# eval_set은 Keras의 validation_data, early_stopping_rounds는 EarlyStopping 역할입니다.
# 최신 XGBoost에서는 early_stopping_rounds를 fit()이 아닌 생성자에 전달합니다.
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)


# %% 6. 검증 ROC-AUC와 변수 중요도
# 조기 종료를 설정한 모델의 predict_proba()는 최적 반복 지점을 자동 사용합니다.
valid_probability = model.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_probability)
best_iterations = model.best_iteration + 1  # 인덱스는 0부터 시작
print(f"Validation ROC-AUC: {valid_auc:.6f}")
print(f"Best iterations: {best_iterations}")
if best_iterations == MAX_ITERATIONS:
    print("최고 성능이 반복 상한에서 나왔습니다. 다음 실험에서는 상한을 늘려보세요.")

feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
print(feature_importance.head(10).to_string(index=False))


# %% 7. 선택한 트리 수로 전체 학습 데이터 재학습
# 검증 AUC는 위에서 측정한 점수입니다. 전체 학습에는 조기 종료를 적용하지 않습니다.
final_model = model
if REFIT_ALL_DATA:
    print("전체 학습 데이터로 제출용 모델을 학습합니다.")
    final_model = XGBClassifier(**{**params, "n_estimators": best_iterations})
    final_model.fit(X, y)


# %% 8. 확률 예측과 제출 파일 저장
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
submission_path = OUTPUT_DIR / "submission_xgboost.csv"
submission.to_csv(submission_path, index=False)
feature_importance.to_csv(OUTPUT_DIR / "validation_feature_importance.csv", index=False)
# 범주형 정보가 보존되는 JSON 형식으로 저장합니다.
final_model.save_model(OUTPUT_DIR / "model.json")

metrics = {
    "validation_roc_auc": float(valid_auc),
    "validation_size": 0.2,
    "seed": SEED,
    "best_iterations": int(best_iterations),
    "refit_all_data": REFIT_ALL_DATA,
    "parameters": params,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "versions": {"xgboost": xgboost.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
}
(OUTPUT_DIR / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)
# 모델을 다시 불러와 예측할 때도 동일한 열 순서와 범주 매핑을 사용해야 합니다.
(OUTPUT_DIR / "feature_schema.json").write_text(
    json.dumps({"columns": X.columns.tolist(), "categories": category_levels}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

catboost_metrics_path = DATA_DIR / "outputs" / "catboost_baseline" / "metrics.json"
if catboost_metrics_path.exists():
    catboost_metrics = json.loads(catboost_metrics_path.read_text(encoding="utf-8"))
    if catboost_metrics.get("seed") == SEED and catboost_metrics.get("validation_size") == 0.2:
        catboost_auc = catboost_metrics["validation_roc_auc"]
        print(f"저장된 CatBoost AUC: {catboost_auc:.6f}")
        print(f"XGBoost - CatBoost: {valid_auc - catboost_auc:+.6f}")
        print("같은 원본 CSV와 검증 분할을 사용했을 때 비교할 수 있는 점수입니다.")

print(submission.head().to_string(index=False))
print("제출 파일:", submission_path)

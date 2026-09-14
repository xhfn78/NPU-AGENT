# 31번 모델/전처리를 불러와 동일한 예측인지 확인
# 공통 모델/분할/학습 조건을 유지하고, 파일명에 해당하는 기능을 더한다.
# 기존 코드와 과거 결과는 _save/diabetes_compare/original_sources/에 보관.
import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.initializers import GlorotUniform
from diabetes_result import save_result, weights_hash
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import joblib
import json

# 비교용 공통 설정. 같은 설정으로 실행한 결과끼리만 비교한다.
SEED = int(os.environ.get('DIABETES_SEED', '221'))
EPOCHS = int(os.environ.get('DIABETES_EPOCHS', '3000'))
BATCH_SIZE = 10
if EPOCHS < 1:
    raise ValueError('EPOCHS must be positive')
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
result_dir = Path(__file__).resolve().parents[1] / '_save' / 'diabetes_compare' / f'seed{SEED}_epochs{EPOCHS}_batch{BATCH_SIZE}'
result_dir.mkdir(parents=True, exist_ok=True)
config = dict(seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, split_seed=221,
              train_size=0.75, validation_fraction=0.2, shuffle=False,
              dense_units=[3, 10, 15, 20, 10, 1], activation='relu',
              optimizer='adam', learning_rate=0.001, patience=15,
              tensorflow=tf.__version__, comparison_version=1)

#1.데이터: test는 모든 단계에서 같은 111개이며, 모델 선택에 사용하지 않는다.
datasets = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.75, random_state=221,
)

# 17번부터 학습 데이터의 뒤 20%를 검증용으로 분리한다.
# validation_split=0.2와 같은 분할. 28번에서 scaler가 검증 데이터를 보지 않도록 미리 나눈다.
split = int(len(x_train) * 0.8)
x_train, x_val = x_train[:split], x_train[split:]
y_train, y_val = y_train[:split], y_train[split:]

#2.31번에서 저장한 모델과 scaler를 함께 복원한다. 새로 fit하지 않는다.
source_name = 'keras31_MCP_save_02_diabetes'
source_result = result_dir / (source_name + '.json')
if not source_result.exists():
    raise FileNotFoundError('같은 SEED/EPOCHS 설정으로 keras31_MCP_save_02_diabetes.py를 먼저 실행하세요.')
saved = json.loads(source_result.read_text(encoding='utf-8'))
if saved['config'] != config:
    raise ValueError('31번과 실행 설정/버전이 다릅니다. 31번부터 다시 실행하세요.')
start_time = time.perf_counter()
model = load_model(result_dir / (source_name + '.keras'))
scaler = joblib.load(result_dir / (source_name + '_scaler.joblib'))
x_test = scaler.transform(x_test)
initial_hash = saved['initial_weights_sha256']
hist = None  #3.훈련 생략: 저장/복원 자체는 성능을 바꾸지 않아야 한다.
seconds = time.perf_counter() - start_time

#4.평가,예측
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0).ravel()
print('loss:', loss)
# 12번 추가: MSE/RMSE는 낮을수록, R2는 높을수록 좋다.
print('r2:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))
expected = np.load(Path(saved['run_dir']) / 'predictions.npy')
np.testing.assert_allclose(y_predict, expected, rtol=1e-6, atol=1e-5)
print('31번 저장 전 / 32번 복원 후 예측 일치 확인 완료')

# 결과 기록은 모든 단계 공통. 19번부터 loss.png도 저장한다.
save_result(__file__, result_dir, config, hist, loss, y_test, y_predict,
            seconds, initial_hash, previous='keras31_MCP_save_02_diabetes', plot=False)

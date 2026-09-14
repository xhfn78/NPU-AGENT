# 19번 + EarlyStopping
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
from tensorflow.keras.callbacks import EarlyStopping

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

#2.모델구성: 28번 이후 사용하던 구조를 모든 비교 단계의 공통 기준으로 맞춤.
model = Sequential()
model.add(Dense(3, input_dim=10, activation='relu', name='dense1', kernel_initializer=GlorotUniform(seed=SEED + 1)))
model.add(Dense(10, activation='relu', name='dense2', kernel_initializer=GlorotUniform(seed=SEED + 2)))
model.add(Dense(15, activation='relu', name='dense3', kernel_initializer=GlorotUniform(seed=SEED + 3)))
model.add(Dense(20, activation='relu', name='dense4', kernel_initializer=GlorotUniform(seed=SEED + 4)))
model.add(Dense(10, activation='relu', name='dense5', kernel_initializer=GlorotUniform(seed=SEED + 5)))
model.add(Dense(1, name='dense6', kernel_initializer=GlorotUniform(seed=SEED + 6)))
initial_hash = weights_hash(model)  # Dropout을 추가해도 Dense 초기값은 동일

#3.컴파일,훈련
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # 20번 추가
start_time = time.perf_counter()
hist = model.fit(
    x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    shuffle=False, verbose=0,  # 데이터 순서도 고정하여 단계별 비교
    validation_data=(x_val, y_val),
    callbacks=[es],
)
seconds = time.perf_counter() - start_time

#4.평가,예측
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0).ravel()
print('loss:', loss)
# 12번 추가: MSE/RMSE는 낮을수록, R2는 높을수록 좋다.
print('r2:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))

# 결과 기록은 모든 단계 공통. 19번부터 loss.png도 저장한다.
save_result(__file__, result_dir, config, hist, loss, y_test, y_predict,
            seconds, initial_hash, previous='keras19_overfit2_diabetes', plot=True)

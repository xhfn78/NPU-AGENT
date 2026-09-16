# [실습] ModelCheckpoint 저장 - 캘리포니아 주택 가격 (회귀)
#
# 훈련하면서 val_loss가 가장 좋았던 시점의 모델을
# ./_save/keras30/ 아래에 k31_01_시각-epoch-val_loss.keras 형태로 저장한다.
# 짝이 되는 불러오기 파일은 keras32_MCP_load_01 이다.
import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler

path = './_save/keras30/'

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=333,
)

##############################################################################
scaler = RobustScaler()
##############################################################################
# RobustScaler
#   X_scaled = (X - 중앙값) / IQR       (IQR = 3사분위수 - 1사분위수)
#   중심을 평균 대신 중앙값으로, 폭을 표준편차 대신 IQR로 잡아 이상치에 강하다.

# [ 스케일러는 x_train에만 fit한다 ]
#   x_test는 x_train에서 학습한 기준으로 transform만 해야 한다.
x_train = scaler.fit_transform(x_train)   # x_train 기준을 학습(fit)하고 동시에 변환(transform)
x_test = scaler.transform(x_test)         # test는 transform만 (fit 금지)

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=8, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
# 파일명에 날짜와 성능을 박아두는 이유:
#   파일명을 고정해두면 실행할 때마다 이전 결과가 덮어써져서 비교를 못 한다.
#     {epoch:04d}     → 4자리 정수  (예: 0088)
#     {val_loss:.4f}  → 소수점 4자리 (예: 0.5173)
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path, 'k31_01_', date, '-', filename])

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
    verbose=1,
)

# ModelCheckpoint(MCP)란?
#   훈련 도중 val_loss가 가장 좋았던 순간의 모델을 파일로 자동 저장해주는 콜백이다.
#
#   EarlyStopping의 restore_best_weights=True 와 뭐가 다른가?
#     EarlyStopping : 최적 가중치를 "메모리 안의 model"에 되돌려준다. 프로그램이 끝나면 사라진다.
#     ModelCheckpoint: 최적 시점의 모델을 "파일"로 남긴다. 나중에 다시 불러 쓸 수 있다.
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,   # 좋아졌을 때만 저장 (False면 매 epoch 저장되어 파일이 쏟아진다)
    filepath=filepath,
    verbose=1,
)

hist = model.fit(
    x_train, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es, mcp],
    verbose=1,
)

#4. 평가, 예측
print("=========================================")
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

mse = mean_squared_error(y_test, y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)

# ------------------------------------------------------------
# 이번 파일을 실행한 결과 기록
# 실행 날짜: 2026-09-14
# random_state = 333 / train_size = 0.75
# epochs = 500 / batch_size = 32 / validation_split = 0.2 / scaler = RobustScaler
# 실제 훈련한 epoch 수: 34에서 EarlyStopping (최적 epoch 14의 가중치로 복원)
# loss: 0.5921303033828735
# r2:   0.5372739128508388
# mse:  0.5921301747928108
# RMSE: 0.7694999511324291
# 저장된 체크포인트: k31_01_0914_2142-0014-0.5756.keras
# ------------------------------------------------------------

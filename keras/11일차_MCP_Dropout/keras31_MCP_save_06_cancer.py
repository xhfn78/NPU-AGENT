# [실습] ModelCheckpoint 저장 - 유방암 (이진 분류)
#
# 훈련하면서 val_loss가 가장 좋았던 시점의 모델을
# ./_save/keras30/ 아래에 k31_06_시각-epoch-val_loss.keras 형태로 저장한다.
# 짝이 되는 불러오기 파일은 keras32_MCP_load_06 이다.
import datetime
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1. 데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=333, stratify=datasets.target)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential([Dense(30, input_dim=30, activation='relu'), Dense(60, activation='relu'), Dense(70, activation='relu'), Dense(80, activation='relu'), Dense(60, activation='relu'), Dense(32, activation='relu'), Dense(1, activation='sigmoid')])

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
stamp = datetime.datetime.now().strftime('%m%d_%H%M')
# 파일명에 날짜와 성능을 박아두는 이유:
#   파일명을 고정해두면 실행할 때마다 이전 결과가 덮어써져서 비교를 못 한다.
#   그래서 실행 시각(stamp)과 epoch, val_loss를 파일명에 넣는다.
#     {epoch:04d}     → 4자리 정수  (예: 0088)
#     {val_loss:.4f}  → 소수점 4자리 (예: 23.6130)
#   이 중괄호는 케라스가 저장하는 순간의 실제 값으로 바꿔준다.
filepath = './_save/keras30/k31_06_' + stamp + '-{epoch:04d}-{val_loss:.4f}.keras'
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
# ModelCheckpoint(MCP)란?
#   훈련 도중 val_loss가 가장 좋았던 순간의 모델을 파일로 자동 저장해주는 콜백이다.
#
#   EarlyStopping의 restore_best_weights=True 와 뭐가 다른가?
#     EarlyStopping : 최적 가중치를 "메모리 안의 model"에 되돌려준다. 프로그램이 끝나면 사라진다.
#     ModelCheckpoint: 최적 시점의 모델을 "파일"로 남긴다. 나중에 다시 불러 쓸 수 있다.
#
#   주요 옵션
#     monitor='val_loss'    → 무엇을 기준으로 좋고 나쁨을 볼지
#     save_best_only=True   → 좋아졌을 때만 덮어쓴다 (False면 매 epoch 저장해서 파일이 쏟아진다)
#     filepath              → 저장할 경로와 파일명
mcp = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.rint(model.predict(x_test)).ravel()
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_test == y_predict))

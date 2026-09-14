# import ssl
# ssl._create_default_https_context = ssl.create_default_context  다운로드 안될때 사용할것

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
# 여기서부터는 직접 만든 숫자가 아니라 사이킷런이 제공하는 실제 데이터를 쓴다.
# 캘리포니아 집값 데이터: feature 8개로 집값(target)을 맞히는 회귀 문제.
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(datasets)   # 데이터 설명까지 통째로 출력된다 (양이 많아서 필요할 때만 켠다)

print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=42
)

# exit()   # 여기서 프로그램을 멈추고 데이터만 확인하려고 썼던 줄. 켜두면 아래가 실행되지 않는다.

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=8))  # feature가 8개이므로 input_dim=8
model.add(Dense(9))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))               # 회귀 문제라 출력은 1개

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=16)

#4. 평가, 예측
print("=========================================")

loss = model.evaluate(x_test, y_test)
print("loss:", loss)
results = model.predict(x)
print('결과값: ', results)

# =========================================
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 682us/step - loss: 0.6196
# loss: 0.6195971369743347
# 645/645 ━━━━━━━━━━━━━━━━━━━━ 0s 350us/step
# 결과값:  [[4.222328 ]
#  [3.9601626]
#  [3.791273 ]
#  ...
#  [0.7560466]
#  [0.8599502]
#  [1.0874628]]

# ============================================================
# California 단계별 결과 누적 비교
# 기존 주석에서 확인한 과거 기록이며, 이번에 새로 훈련한 결과는 아님.
# 단계마다 random_state, 층 구성, epochs, batch_size 등이 달라 기능 하나의 효과로 단정하지 않기.
# loss / MSE / RMSE는 낮을수록, R2는 높을수록 좋음.
# 미기록 칸은 실제 실행 후 채우기. 아래쪽 파일일수록 앞 단계 기록을 누적함.
# ============================================================
#
# 11. 기본 회귀 모델
# 파일: keras11_1_califonia.py
# 기존 주석 기록: loss = 0.6195971369743347
# r2 / mse / RMSE: 당시 별도 기록 없음
#
# ------------------------------------------------------------
# 이번 파일을 다시 실행한 결과 기록
# 실행 날짜: 
# 추가 / 변경한 내용: 
# 현재 코드 설정 (과거 결과의 실행 조건을 뜻하지 않음):
# random_state = 42 / train_size = 미지정 (기본 분할)
# epochs = 200 / batch_size = 16 / validation_split = 없음
# 실제 훈련한 epoch 수: 
# loss: 
# r2: 
# mse: 
# RMSE: 
# 이전 비교 대상 파일: 
# 이전 결과보다 좋아진 점 / 나빠진 점: 
# 다음 실험에서 바꿀 내용: 
# ============================================================

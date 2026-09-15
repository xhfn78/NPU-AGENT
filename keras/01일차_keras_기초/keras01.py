import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# y = wx + b 에서 w(가중치=기울기)를 조정하는 걸 역전파라고 한다.
# 최적의 w를 찾는 게 학습이다.
#
# 예) x가 1일 때 y가 1이고, x가 2일 때 y가 2인 걸 아는 건 역전파(기울기 조정) 덕분이다.
#
# x = 고양이가 물고기를 본다
# y = 고양이가 침을 흘린다
# 중간에 굉장히 많은 뉴런(신경망)이 있고, 그 신경망을 연결하는 선의 개수만큼 y = ax + b 가 있다.
# (그것도 순차적으로 이어져 있다. 눈 - o layer1 - o - o - o layer4 - 입)
# 이렇게 신경망으로 이루어진 형태를 딥러닝(깊이 있는 학습)이라고 한다.
#
# AI 안에 ML(머신러닝)이 있고, ML 안에 딥러닝이 있다.
# LLM도 딥러닝의 한 종류다. transformer라는 딥러닝 모델을 사용한다.
# 신경망이 깊으면 학습을 더 잘한다.
#
# 목표: 최소의 오차, 최적의 weight

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
# 최소의 오차, 최적의 weight
model = Sequential()  # 신경망을 순차적으로 쌓는 시퀀셜 모델
model.add(Dense(1, input_dim=1))
# input_dim=1 은 입력 x의 feature가 1개(1차원)라는 뜻
# Dense(1) 의 1 은 이 층에서 나가는 출력 뉴런의 개수

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# mse = mean squared error(평균 제곱 오차). 오차를 재는 방법 중 하나다.
# y = ax + b 선과 실제 데이터 점들 사이의 거리 차이가 loss(손실률)이다.
# optimizer는 그 손실률을 줄이는 방법. adam은 최적화 알고리즘 중 하나다.

model.fit(x, y, epochs=100)
# fit은 훈련시키다. (정확히는 선을 100번 다시 그려가며 오차를 줄이는 과정을 반복)
# 너무 많이 훈련시키면 과적합이 되어 오히려 안 좋다.

#4. 평가, 예측
result = model.predict(np.array([4]))
print("4의 예측값 : ", result)

# 결과 4의 예측값 : [[3.8398268]]  Epoch 100/100

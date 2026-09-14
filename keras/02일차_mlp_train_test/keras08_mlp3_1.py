import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# 여기서는 y도 열이 2개다. (출력이 2개인 문제)
x = np.array([range(10),range(21,31),range(201,211)]).T
y = np.array(([1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1])).transpose()
print(x.shape, y.shape)  # (3, 10) (2, 10) >>>> (10, 3) (10, 2) 전치행렬로 바꿔줌

# [실습] [10, 31, 211]의 예측값이 [11.00, 0.00] 이 나오게 만들기

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))  # x의 feature가 3개
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))               # y의 열이 2개이므로 마지막 출력도 2

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss:", loss)
results = model.predict(np.array([[10, 31, 211]]))
print("results:", results)

# loss: 7.203358048935016e-11
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step
# results: [[ 1.0999991e+01 -1.0058284e-06]]

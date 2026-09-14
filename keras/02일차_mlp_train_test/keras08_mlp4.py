import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# 이번에는 반대로 입력이 1개, 출력이 3개인 경우다.
x = np.array([range(10)]).T
y = np.array(([1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0])).transpose()
print(x.shape, y.shape)  # (1, 10) (3, 10) >>>> (10, 1) (10, 3) 전치행렬로 바꿔줌

# [실습] 11, 0, -1 이 나오게 만들기

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))  # x의 feature가 1개
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))               # y의 열이 3개이므로 마지막 출력도 3

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict(np.array([[10]]))
print("loss:", loss)
print("results:", results)

# loss가 높은데 예측 결과가 좋아 보이는 것보다,
# 예측 결과가 조금 나빠도 loss가 낮은 쪽이 더 믿을 만한 모델이다.

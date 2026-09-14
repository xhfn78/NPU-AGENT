from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))  # input_dim=1 → 입력 1개. 앞의 숫자는 출력 노드 개수, 노드 개수는 조절 가능
model.add(Dense(10))              # 레이어 층은 늘리고 줄일 수 있다
model.add(Dense(10))
model.add(Dense(1))
# dim은 차원(dimension)을 뜻한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=6)
# 데이터가 클 때 잘라서 학습하는 단위를 batch라고 한다.
# batch_size=3 이면 데이터를 3개씩 끊어서 학습하고, 안 적으면 디폴트는 32개씩이다.
# 여기서는 데이터가 6개뿐이라 batch_size=6 = 한 번에 전체를 학습한다는 뜻이다.

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :  ", loss)
# result = model.predict(np.array([1,2,3,4,5]))
# print(" 예측값: ", result)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))  # input_dim=1 → 입력 1개. 앞의 숫자는 출력 노드 개수
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(1))
# 노드 개수와 층 수를 바꿔보는 것이 하이퍼파라미터 튜닝이다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=6)
# 데이터 크기가 클 때 잘라서 작업하는 단위를 batch라고 한다.
# batch_size를 안 적으면 디폴트는 32다.
# batch_size가 작으면 → 가중치를 자주 갱신해서 세밀하지만 느리다.
# batch_size가 크면   → 한 번에 많이 보고 갱신해서 빠르지만 거칠다.
# 크다고 좋은 것도, 작다고 좋은 것도 아니다.

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :  ", loss)
# result = model.predict(np.array([1,2,3,4,5]))
# print(" 예측값: ", result)

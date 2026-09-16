from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터를 꼬았을 때
# y가 x를 그대로 따라가지 않고 3번째와 4번째가 뒤집혀 있다.
# 직선 하나로는 모든 점을 정확히 지날 수 없으므로 loss가 0까지 내려가지 않는다.
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
# 층이 하나뿐이므로 이 모델이 그릴 수 있는 건 직선 y = wx + b 하나뿐이다.
# (층을 더 쌓아서 성능을 올리는 건 keras04_deep1 에서 한다.)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :  ", loss)
result = model.predict(np.array([1,2,3,4,5]))
print(" 예측값: ", result)

# loss :   0.3819732666015625
#  예측값:  [[1.1263199]
#  [2.0561028]
#  [2.9858856]
#  [3.9156685]
#  [4.8454514]]

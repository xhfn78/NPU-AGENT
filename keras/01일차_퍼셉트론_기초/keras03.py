from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.데이터를 꼬았을때 
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=500)

#4.평가,예측.
loss = model.evaluate(x, y)
print("loss :  ", loss)
result = model.predict(np.array([1,2,3,4,5]))
print(" 예측값: ", result)
#loss :   0.3819732666015625
# 예측값:  [[1.1263199]
# [2.0561028]
# [2.9858856]
# [3.9156685]
# [4.8454514]]
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# train, test를 수동으로 적지 말고 넘파이 슬라이싱으로 잘라보기.
# 인덱스 시작은 항상 0이라 x[0:7] 의 0은 생략할 수 있다.

x_train = x[:7]
y_train = y[:7]

print(x_train)
# [1 2 3 4 5 6 7]

x_test = x[7:]
y_test = y[7:]
print(x_test)
# [ 8  9 10]

# x_train = np.array([1,2,3,4,5,6,7])# 훈련용 데이터 70프로
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10]) #테스트용 데이터 30프로
# y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=800, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
# 훈련에 안 쓴 데이터로 평가해야 신뢰할 수 있다.
# 다만 이렇게 앞뒤로 자르는 방식에는 문제가 있다 → keras09_train_test3 에서 이어서 본다.

# Epoch 800/800
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.1079
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - loss: 0.4022
# loss: 0.402214914560318

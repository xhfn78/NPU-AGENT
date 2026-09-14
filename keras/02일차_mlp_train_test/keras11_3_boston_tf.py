from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1. 데이터
# 보스턴 집값 데이터는 사이킷런이 아니라 텐서플로에서 가져온다.
# 그리고 이 데이터는 처음부터 train / test가 나뉘어서 들어온다.
# 그래서 train_test_split을 따로 쓰지 않아도 된다.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))  # feature가 13개이므로 input_dim=13
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=5)

#4. 평가, 예측
print("=========================================")

loss = model.evaluate(x_test, y_test)
print("loss:", loss)

# 아래는 조건을 바꿔가며 돌린 기록. 반복할수록 loss가 63 → 23 까지 내려갔다.
# #4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 63.2345 
# loss: 63.234466552734375

# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 33.9610 
# loss: 33.96104431152344

# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 26.3309 
# loss: 26.330886840820312

# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 25.0492 
# loss: 25.049177169799805

# 81/81 ━━━━━━━━━━━━━━━━━━━━ 0s 818us/step - loss: 25.7187
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 23.3918 
# loss: 23.391815185546875

# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 23.0776 
# loss: 23.077598571777344

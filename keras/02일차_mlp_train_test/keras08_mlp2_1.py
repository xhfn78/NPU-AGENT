import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# feature(열)가 3개인 데이터.
# 지금은 (3, 10) 모양이라 행이 feature, 열이 샘플로 뒤집혀 있다.
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.3],
              [9,8,7,6,5,4,3,2,1,0]
              ])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = x.T   # 전치해서 (10, 3) → 샘플 10개, feature 3개

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))  # 행 무시 열 우선 → feature가 3개이므로 input_dim=3
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))  # y는 값이 1개씩이므로 출력 1

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : " , loss)
results = model.predict(np.array([[10, 1.3, 0]]))
print("[10, 1.3, 0]의 예측값 : " , results)

# loss :  2.8521964168248815e-07
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step
# [10, 1.3, 0]의 예측값 :  [[10.001418]]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
# 이번 y는 일직선이 아니라 위아래로 흔들리는 데이터다.
# 모델이 그린 선이 점들 사이를 어떻게 지나가는지 그래프로 직접 확인해본다.
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.7, # 같이 써도 상관없음, 통상적으로 하나만 써도 됨
    # test_size=0.3,
    # shuffle=True,   # 디폴트값으로 shuffle=True가 들어감

    # 고정하지 않으면 실행시킬 때마다 값이 달라짐 >>>
    random_state=444,  # 난수표에 있는 값을 넣어서 데이터 랜덤값을 고정시킴
)

print('x_train :', x_train)
print('x_test  :', x_test)
print('y_train :', y_train)
print('y_test  :', y_test)

# x_train : [ 2  9  4  3  1 10  5]
# x_test  : [7 8 6]

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=2)

print("=========================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
# evaluate에는 배치 개념이 없다. 통째로 넣어서 평가하고,
# fit과 달리 순전파(추론)만 하기 때문에 w 가중치를 변경시키지 않는다.

results = model.predict(x)   # 그래프를 그리려고 전체 x에 대해 예측한다
print('결과값: ', results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)                  # scatter = 데이터의 위치를 점으로 흩뿌려 표현
plt.plot(x, results, color='red')  # plot = 선 긋기 (모델이 학습한 선)
plt.show()

# =========================================
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 71ms/step - loss: 0.1316
# loss: 0.13159851729869843

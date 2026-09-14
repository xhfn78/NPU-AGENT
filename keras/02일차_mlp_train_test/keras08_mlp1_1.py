import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# 아래처럼 주면 학습이 안 된다.
# x는 2덩이인데 y는 5덩이라서 행(샘플) 개수가 안 맞기 때문이다.
# x = np.array([[1,2,3,4,5],   # x.shape : (2, 5)
#               [6,7,8,9,10]])

# 그래서 x도 5덩이가 되도록 이렇게 줘야 한다. (5행 2열)
x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

print("x.shape : ", x.shape)  # x.shape :  (5, 2) (행, 열)
print("y.shape : ", y.shape)  # y.shape :  (5,)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))
# 행 무시, 열 우선.
# input_dim의 개수는 x.shape의 열 개수와 같아야 한다.
# x는 feature(열)가 2개이므로 input_dim=2
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))  # y는 값이 1개씩이므로 마지막 출력은 1

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# compile은 학습을 시작하기 전에 학습 환경(손실 함수, 최적화 방법)을 설정하고 준비하는 단계다.

model.fit(x, y, epochs=100, batch_size=3)
# 훈련하려면 x, y가 필요하고 epochs, batch_size를 설정해준다.
# 배치 사이즈는 크다고 좋은 것도, 작다고 좋은 것도 아니다.
#
# batch_size=3 이면 위에서부터 3덩이씩 잘라서 학습한다.
#   1번째 epoch 시작
#     1번째 훈련: x = [1,6],[2,7],[3,8] / y = [1,2,3]
#     2번째 훈련: x = [4,9],[5,10]      / y = [4,5]
#   1번째 epoch 끝
#   2번째 epoch부터도 동일하게 위에서부터 잘라서 훈련한다.

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
results = model.predict(np.array([[6, 11]]))
print("[6, 11]의 예측값 : ", results)

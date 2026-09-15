from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# keras03과 같은, 꼬인 데이터다.
# 직선 하나로 안 되니 층을 깊게(deep) 쌓아서 성능을 올려본다.
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))  # input_dim=1 → 입력 feature 1개, 앞의 5 → 출력 노드 5개
model.add(Dense(7, input_dim=5))
model.add(Dense(7, input_dim=7))
model.add(Dense(1, input_dim=7))
# 노드 개수와 층 수를 바꿔가며 성능을 조정하는 것을 하이퍼파라미터 튜닝이라고 한다.
# 참고: 두 번째 층부터는 input_dim을 안 적어도 된다.
#       앞 층의 출력 개수가 자동으로 다음 층의 입력이 되기 때문이다. (여기서는 흐름을 보려고 적어둔 것)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000)  # 에포크 숫자를 조정해서 훈련 횟수를 조정할 수 있다

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :  ", loss)
result = model.predict(np.array([1,2,3,4,5]))
print(" 예측값: ", result)

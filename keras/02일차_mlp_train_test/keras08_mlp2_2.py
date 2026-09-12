import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array(range(10)) #[0,1,2,3,4,5,6,7,8,9]
print(x)

x = np.array(range(1,11))
print(x)  #[ 1  2  3  4  5  6  7  8  9 10]

x = np.array([range(10),range(21,31),range(201,211)]).T  #(끝에 .T를 붙여서 전치행렬로 바꿔줌)

print(x.shape) #(3, 10) -> (10, 3)으로 바뀜 

y = np.array(range(1,11))
print(y.shape) #(10,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y, epochs=1000, batch_size=2)



#4.평가예측
loss = model.evaluate(x,y)
print("loss:", loss)
results = model.predict(np.array([[10, 31, 211]]))
print("results:", results)

# loss: 2.504293661331758e-09
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
# results: [[11.000075]]
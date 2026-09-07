import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_tran,x_test,y_train,y_test = train_test_split(
  x,y,
  random_state=555

)

#2.모델구성

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3.컴파일 ,훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_tran,y_train , epochs=200 ,batch_size=4)

print("=========================================")

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)
results = model.predict(x)
print('결과값: ' ,results)

#그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)        #스캐터=데이터의 위치를 씨를 뿌려서 표현함
plt.plot(x, results, color='red')            #plot=선 긋기
plt.show()

#랜덤 333 배치2
# loss: 4.36538028717041
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
# 결과값:  [[ 0.7596344]

#랜덤 555 배치4
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 11.5880
# =========================================
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 10.7442
# loss: 10.744222640991211
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
# 결과값:  [[ 1.2244586]
#  [ 2.1026316]
#  [ 2.980805 ]
#  [ 3.8589783]
#  [ 4.737152 ]
#  [ 5.615325 ]
#  [ 6.493498 ]
#  [ 7.3716717]
#  [ 8.2498455]
#  [ 9.128017 ]
#  [10.006191 ]
#  [10.884364 ]
#  [11.762537 ]
#  [12.64071  ]
#  [13.518885 ]
#  [14.397057 ]
#  [15.27523  ]
#  [16.153406 ]
#  [17.031578 ]
#  [17.909752 ]]

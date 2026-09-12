import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1.데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#train,test 수동말고 자동으로 하는법 ,시작점은 0이라 생략가능 

x_train = x[:7]
y_train = y[:7]

print(x_train)
#[1 2 3 4 5 6 7]

x_test = x[7:]
y_test = y[7:]
print(x_test)
# [ 8  9 10]


# x_train = np.array([1,2,3,4,5,6,7])# 훈련용 데이터 70프로
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10]) #테스트용 데이터 30프로
# y_test = np.array([8,9,10])
#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=800 , batch_size=4)

#4.평가,예측

loss = model.evaluate(x_test,y_test)
print("loss:", loss)

# Epoch 800/800
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.1079
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - loss: 0.4022
# loss: 0.402214914560318
 

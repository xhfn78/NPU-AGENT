#11_3 COPY
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing


#1.데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)


#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam') #mse= 원값에서 예측값 뺴고 나온값을 제곱 > 다 더해서 갯수만큼 엔빵
model.fit(x_train,y_train, epochs=1000, batch_size=5 )


#4.평가,예측
print("=========================================")

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)


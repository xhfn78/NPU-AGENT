from sklearn.datasets import fetch_california_housing, load_diabetes #캘리포니아 집값 데이터셋,로드 디아벳
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

#1.데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=21
)

#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=10))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train,y_train, epochs=500, batch_size=8)


#4.평가,예측
print("=========================================")

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)


print('결과값: ' ,r2)

#R2기준 0.62 이상 만들기
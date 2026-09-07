# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score , mean_squared_error


#1.데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=500 
)

print(x.shape,y.shape) #(20640, 8) (20640,)

#2.모델구성
model = Sequential()
model.add(Dense(7, input_dim=8))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train,y_train, epochs=200, batch_size=60  )


print("=========================================")

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('결과값: ' ,r2)

# R2기준 0.55 만들기
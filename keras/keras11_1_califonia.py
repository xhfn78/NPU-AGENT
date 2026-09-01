# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=42
)

print(x.shape,y.shape) #(20640, 8) (20640,)

#2.모델구성
model = Sequential()
model.add(Dense(9, input_dim=8))
model.add(Dense(9))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train,y_train, epochs=200, batch_size=16  )


#4.평가,예측
print("=========================================")

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)
results = model.predict(x)
print('결과값: ' ,results)

# =========================================
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 682us/step - loss: 0.6196
# loss: 0.6195971369743347
# 645/645 ━━━━━━━━━━━━━━━━━━━━ 0s 350us/step
# 결과값:  [[4.222328 ]
#  [3.9601626]
#  [3.791273 ]
#  ...
#  [0.7560466]
#  [0.8599502]
#  [1.0874628]]
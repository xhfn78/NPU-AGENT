from sklearn.datasets import fetch_california_housing, load_diabetes #캘리포니아 집값 데이터셋,로드 디아벳
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
import numpy as np

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
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train,y_train, epochs=1000, batch_size=2 )


#4.평가,예측
print("=========================================")

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)
# results = model.predict(x)
# print('결과값: ' ,results)
#랜덤 442
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2857.6667 
# loss: 2857.666748046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[210.36603 ]

#랜덤 221
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2459.2292 
# loss: 2459.229248046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

#랜덤 3333
# 42/42 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2882.3848 
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 3071.2979 
# loss: 3071.2978515625
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[204.20064 ]
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1.데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test, y_train,y_test = train_test_split(
    x, y, 
    #train_size=0.7, #같이써도 상관없음,통상적으로 하나만써도됨
    # test_size=0.3,
    #shuffle=True, #디폴트값으로 Shuffle값이 들어감 

    #실행 시킬떄 마다 값이 틀려짐>>>
    random_state=444,  #>>난수표에 있는 값을 넣어서 데이터 랜덤값을 고정시킴 

    #결과값
    #x_train : [ 2  9  5  8  7 10  4]
    # x_train : [6 3 1]
    # x_train : [ 2  9  5  8  7 10  4]
    # x_train : [6 3 1]
)

print('x_train :', x_train)
print('x_train :', x_test)
print('x_train :', y_train)
print('x_train :', y_test)

# x_train : [ 2  9  4  3  1 10  5]
# x_train : [7 8 6]
# x_train : [ 2  9  4  3  1 10  5]
# x_train : [7 8 6]

#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=2)

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)
results = model.predict(np.array([11]))
print('11의결과값: ' ,results)

#랜덤값=333
# loss: 0.09532500058412552
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
# 11의결과값:  [[10.784775]]

#랜덤값=444
# loss: 0.14206752181053162
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
# 11의결과값:  [[10.672892]]
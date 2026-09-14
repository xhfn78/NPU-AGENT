#9-1 카피
# [실습] verbose 옵션 이해하기
# 훈련 과정을 화면에 얼마나 자세히 찍을지 정하는 것이 verbose다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])# 훈련용 데이터 70프로
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10]) #테스트용 데이터 30프로
y_test = np.array([8,9,10])



#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=100 , batch_size=4, 
          verbose=1,
          )
#verbose = 0 : 침묵. 프로그레스 바 안 나오고 결과만 나옴
#verbose = 1 : 디폴트값 (기존에 쓰던 방식)
#verbose = 2 : 프로그레스 바 안 보임
#verbose = 3 : 프로그레스 바 안 보임, 에포크 횟수만 나옴
#verbose = 나머지 : 에포크만 나옴.
# 출력을 줄이면 화면에 찍는 시간이 줄어서 훈련이 조금 빨라진다. 




#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss:", loss)

#1st
# loss: 0.1232  여기까지가 훈련한 데이터의 로스 값
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 0.9059 평가했을때 로스값은 0.9로나옴
# loss: 0.905925452709198

#2st
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.0395
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 0.1439
# loss: 0.1439265012741089

#3st
# Epoch 1000/1000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 1.0692e-04
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - loss: 3.7291e-04
# loss: 0.0003729090967681259
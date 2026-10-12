# x1은 2-1모델의 인풋으로 x2는 2-2모델의 인풋으로 들어간뒤 각각의 출력값을 2-3에서 concat한뒤에 dense 통과하여 y값을 뽑아냄 
# -> 2개의 모델을 이용한 ensemble
# train_test_split하는 부분 -> x1과 x2를 train_test_split 한번에 쪼개기 가능

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T    # (100, 2)
                       # 삼성 종가      하이닉스 종가  
x2_datasets = np.array([range(101, 201), range(411, 511),  # (100, 3)
                        # 원유가               환율
                        range(150, 250)]).transpose()     
                         # 금시세  

y = np.array(range(3001, 3101))
                # 화성의 화씨 온도


# train_test_split   
x1_train, x1_val, x2_train, x2_val, y_train, y_val= train_test_split(x1_datasets, x2_datasets, y,
                                train_size=0.7,
                                random_state = 1)


#2-1. 모델 (함수형)
input1 = Input(shape = (2,))
dense1 = Dense(10, activation = 'relu', name = 'han1')(input1)
dense2 = Dense(20, activation = 'relu', name = 'han2')(dense1)
dense3 = Dense(30, activation = 'relu', name = 'han3')(dense2)
output1 = Dense(5, activation = 'relu', name = 'han4')(dense3)
# model1 = Model(inputs = input1, outputs = output1)

#2-2. 모델(함수형)
input21 = Input(shape = (3,))
dense21 = Dense(50, name = 'han21')(input21)
dense22 = Dense(40, name = 'han22')(dense21)
dense23 = Dense(30, name = 'han23')(dense22)
dense24 = Dense(20, name = 'han24')(dense23)
output21 = Dense(3, name = 'han25')(dense24)
# model2 = Model(inputs = input21, outputs = output21)

#2-3. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate
# 대문자 Concatenate는 클래스, 소문자는 method 또는 함수
# 대문자 concat, 소문자 concat 기능은 같고 사용 방법이 다름

# merge1 = concatenate([output1, output21], name = 'mg1')
merge1 = Concatenate(name = 'mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs = [input1, input21], outputs = last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train,
          epochs =100,
          batch_size =8)

#4. 평가, 예측
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516),
                    range(249,255)]).T


y_pred = model.predict([x1_pred, x2_pred])
print("y_pred: ", y_pred)

result = model.evaluate([x1_val, x2_val], y_val)
print('loss : ', result)



# 예측값까지 완성
# loss :  1.6490618293119041e-07
# y_pred:  [[3100.0005]
#  [3101.0005]
#  [3102.0005]
#  [3103.0007]
#  [3104.0007]
#  [3105.0005]]
# 케라스 69-1 카피
# 69-1에서는 x1, x2만 있었고 동일한 코드에서 x3를 추가하여 수정
# train_test_split 여러개 만들지 않고 여러개의 x의 그룹과 y그룹을 하나의 train_test_split으로 나눌 수 있다. 

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
x3_datasets = np.array([range(100), range(301,401),
                        range(77,177), range(33,133)]).T # (100, 4)

y = np.array(range(3001, 3101))
                # 화성의 화씨 온도

# train_test_split   
x1_train, x1_val, x2_train, x2_val, x3_train, x3_val, y_train, y_val= train_test_split(x1_datasets, x2_datasets, x3_datasets, y,
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


#2-3. 모델(함수형)
input31 = Input(shape = (4,))
dense31 = Dense(50, name = 'han31')(input31)
dense32 = Dense(40, name = 'han32')(dense31)
dense33 = Dense(30, name = 'han33')(dense32)
dense34 = Dense(20, name = 'han34')(dense33)
output31 = Dense(3, name = 'han35')(dense34)



#2-3. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate
# 대문자 Concatenate는 클래스, 소문자는 method 또는 함수
# 대문자 concat, 소문자 concat 기능은 같고 사용 방법이 다름

# merge1 = concatenate([output1, output21], name = 'mg1')
merge1 = Concatenate(name = 'mg1')([output1, output21, output31])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs = [input1, input21, input31], outputs = last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y_train,
          epochs =100,
          batch_size =8)

#4. 평가, 예측
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516),
                    range(249,255)]).T
x3_pred = np.array([range(100,106), range(400, 406),
                    range(177,183), range(133,139)]).T


y_pred = model.predict([x1_pred, x2_pred,  x3_pred])
print("y_pred: ", y_pred)

result = model.evaluate([x1_val, x2_val, x3_val], y_val)
print('loss : ', result)



# 예측값까지 완성
# y_pred:  [[3094.454 ]
#  [3095.7002]
#  [3096.9468]
#  [3098.1929]
#  [3099.4397]
#  [3100.6863]]
# loss :  0.021815184503793716
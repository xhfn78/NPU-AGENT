# 케라스 69-2 카피
# 69-2에서는 (x1, x2, x3), (y)를 데이터로 사용했고 동일한 코드에서 y2를 추가하여 수정 
# 아래 두부분에서 y값 들어가는 부분 수정
"""
model = Model(inputs = [input1, input21, input31], outputs = [last_output1, last_output2]) # outputs 두개 값 나오도록 수정

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train],     # 훈련데이터 y값 y1_train, y2_train 사용
          epochs =100,
          batch_size =8)
"""
# train_test_split 여러개 만들지 않고 여러개의 x의 그룹과 y그룹을 하나의 train_test_split으로 나눌 수 있다. 

# 결과값 출력시 69-1. 69-2코드와 다른점: loss를 출력하면 3개값을 가진 리스트로 나온다. 
# 첫 번째 값 (전체 Loss), 두 번째 값 (분기1 Loss), 세 번째 값 (분기2 Loss)

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

y1 = np.array(range(3001, 3101))
                # 화성의 화씨 온도

y2 = np.array(range(13001, 13101))
               # 비트코인


# train_test_split   
x1_train, x1_val, x2_train, x2_val, x3_train, x3_val, y1_train, y1_val, y2_train, y2_val= train_test_split(x1_datasets, x2_datasets, x3_datasets, y1, y2,
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


# 2-5 분기1.
last_dense1 = Dense(10, name = 'ld1')(merge3)
last_dense2 = Dense(10, name = 'ld2')(last_dense1)
last_output1 = Dense(1, name = 'last1')(merge3)

# 2-6 분기2.
last_output2 = Dense(1, name = 'last2')(merge3)



model = Model(inputs = [input1, input21, input31], outputs = [last_output1, last_output2])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train],
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

result = model.evaluate([x1_val, x2_val, x3_val], [y1_val, y2_val])
print('loss : ', result)
# 첫 번째 값 (전체 Loss), 두 번째 값 (분기1 Loss), 세 번째 값 (분기2 Loss)




# 예측값까지 완성
# x값 3종류, y값 1종류
# y_pred:  [[3094.454 ]
#  [3095.7002]
#  [3096.9468]
#  [3098.1929]
#  [3099.4397]
#  [3100.6863]]
# loss :  0.021815184503793716

# x값 3종류, y값 2종류
# y_pred:  [array([[3100.125 ],
#        [3101.3481],
#        [3102.5703],
#        [3103.7761],
#        [3104.9814],
#        [3106.1873]], dtype=float32), array([[13056.265],
#        [13057.224],
#        [13058.182],
#        [13059.152],
#        [13060.122],
#        [13061.089]], dtype=float32)]
# loss :  [41.34425735473633, 38.26511764526367, 3.079138994216919]


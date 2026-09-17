#36-2 카피
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
 
#1데이터 
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape,y_train.shape)#(60000, 28, 28,1) (60000,) #흑백데이터라 (60000,28,28)로 표현됨
# print(x_test.shape,y_test.shape) #(10000, 28, 28,1) (10000,)

# print(np.max(x_train),np.min(x_train))  #255 0
# print(np.max(x_test),np.min(x_test))    #255 0


###스케일링 1
x_train = x_train/255.
x_test = x_test/255.

# print(np.max(x_train),np.min(x_train))  #1.0 0.0
# print(np.max(x_test),np.min(x_test))    #1.0 0.0


# ###스케일링 2
# x_train = (x_train-127.5)/127.5
# x_test = (x_test- 127.5)/127.5

x_train =x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
# print(np.max(x_train),np.min(x_train))  #1.0 -1.0
# print(np.max(x_test),np.min(x_test))    #1.0 -1.0

#######################원한 인코더###########################
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)  
y_test = y_test.reshape(-1,1) 
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape,y_test.shape)  #(60000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = (28,28,1)))  #  (26 ,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu',padding='same')) #(24,24,32)
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64,(2,2),padding='same',activation='relu')) #(None, 23, 23, 32)
model.add(Conv2D(32,(2,2),activation='relu')) #(None, 22, 22, 16)
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2),activation='relu')) #(None, 22, 22, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2),activation='relu')) #(None, 20, 20, 16) 4차원데이터에서 마지막 (None, 10) 로 붙이기 위해서 flatten 사용
model.add(Flatten())   #(None, 6400)

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu')) #아웃풋 node의 갯수= units

model.add(Dense(10, activation='softmax'))#(None, 10) 
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
                                                                 
#  dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 23, 23, 64)        8256      
                                                                 
#  conv2d_3 (Conv2D)           (None, 22, 22, 32)        8224      
                                                                 
#  dropout_1 (Dropout)         (None, 22, 22, 32)        0         
                                                                 
#  conv2d_4 (Conv2D)           (None, 21, 21, 16)        2064      
                                                                 
#  dropout_2 (Dropout)         (None, 21, 21, 16)        0         
                                                                 
#  conv2d_5 (Conv2D)           (None, 20, 20, 16)        1040      
                                                                 
#  flatten (Flatten)           (None, 6400)              0         
                                                                 
#  dense (Dense)               (None, 32)                204832    
                                                                 
#  dropout_3 (Dropout)         (None, 32)                0         
                                                                 
#  dense_1 (Dense)             (None, 16)                528       
                                                                 
#  dense_2 (Dense)             (None, 10)                170       
                                                                 
# =================================================================
# Total params: 244,218
# Trainable params: 244,218
# Non-trainable params: 0
# _______________________________________________________________

#3 컴파일,훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics= ['acc']
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train,y_train,
          epochs=100,
          batch_size=128,
          verbose=1,
          validation_split =0.2,
          callbacks =[es,],
          )
end_time = time.time()

#평가예측
print(('=====================model.evaluate=================='))
loss = model.evaluate(x_test,y_test, verbose=1)
print('loss: ',loss[0])
print('acc: ',loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test,y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 :',round(end_time-start_time,2),'초')

'''
cpu걸린시간
loss:  0.04604562371969223
acc:  0.9908000230789185
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
accuracy_score :  0.9908
걸린시간 : 595.72 초

gpu 걸린시간 
loss:  0.04410000890493393
acc:  0.9908999800682068
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9909
걸린시간 : 122.55 초

loss:  0.03946596384048462
acc:  0.9912999868392944
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9913
걸린시간 : 225.34 초
'''
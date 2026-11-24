#36-2 카피
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,BatchNormalization,Activation
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
 
#1데이터 
(x_train,y_train),(x_test,y_test) = mnist.load_data()


###스케일링 1
x_train = x_train/255.
x_test = x_test/255.



x_train =x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

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
model.add(Conv2D(filters=32, kernel_size=(3,3),activation=None)) #(24,24,32)
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64,(2,2),activation=None)) #(None, 23, 23, 32)
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
# model.add(BatchNormalization(axis=-1))
model.add(Flatten())   #(None, 6400)
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu')) #아웃풋 node의 갯수= units
model.add(Dense(10, activation='softmax'))#(None, 10) 

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
          epochs=50,
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


#배치 >>>렐루
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0657 - acc: 0.9840
# loss:  0.06570719927549362
# acc:  0.984000027179718
# 313/313 [==============================] - 1s 1ms/step
# accuracy_score :  0.984
# 걸린시간 : 69.06 초
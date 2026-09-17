import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
 
#1데이터 
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# print(x_train.shape,y_train.shape)#(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# print(np.max(x_train),np.min(x_train))  #255 0
# print(np.max(x_test),np.min(x_test))    #255 0

# print(np.unique(y,return_counts=True))
# exit()
###스케일링 1
x_train = x_train/255.
x_test = x_test/255.

# print(np.max(x_train),np.min(x_train))  #1.0 0.0
# print(np.max(x_test),np.min(x_test))    #1.0 0.0


# ###스케일링 2
# x_train = (x_train-127.5)/127.5
# x_test = (x_test- 127.5)/127.5

x_train =x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,32*32*3)
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
# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(units=3000, input_shape=(32*32*3,),activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=6000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=3000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=500, activation='relu')) #아웃풋 node의 갯수= units
model.add(Dense(units=300, activation='relu')) #아웃풋 node의 갯수= units
model.add(Dense(units=200, activation='relu')) #아웃풋 node의 갯수= units
model.add(Dense(10, activation='softmax'))#(None, 10) 
model.summary()


#3 컴파일,훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics= ['acc']
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train,y_train,
          epochs=1000,
          batch_size=256,
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
#1
# loss:  0.9772673845291138
# acc:  0.661899983882904
# accuracy_score :  0.6619

#2
loss:  0.9795273542404175
acc:  0.6651999950408936
accuracy_score :  0.6652
걸린시간 : 230.82 초

#3
loss:  0.9311016798019409
acc:  0.6764000058174133
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.6764
걸린시간 : 416.05 초

#4
loss:  0.6919620037078857
acc:  0.7577000260353088
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.7577
걸린시간 : 1222.64 초

#5 gap 적용
loss:  0.704610288143158
acc:  0.7556999921798706
accuracy_score :  0.7557
걸린시간 : 1079.85 초


#6 
#6 cnn-> dnn
loss:  1.5248686075210571
acc:  0.46209999918937683
313/313 [==============================] - 1s 3ms/step
accuracy_score :  0.4621
걸린시간 : 787.09 초

loss:  1.4667251110076904
acc:  0.4832000136375427
313/313 [==============================] - 1s 3ms/step
accuracy_score :  0.4832
걸린시간 : 1125.71 초
'''
#36-2 카피
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
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
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu')) #(24,24,32)
model.add(Dropout(0.2))
model.add(Conv2D(64,(2,2),activation='relu')) #(None, 23, 23, 32)
model.add(Dropout(0.2))
model.add(Flatten())   #(None, 6400)

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu')) #아웃풋 node의 갯수= units

model.add(Dense(10, activation='softmax'))#(None, 10) 
# model.summary()


#3 컴파일,훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics= ['acc']
              )

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

tb =TensorBoard(log_dir='c:/study/_save/tensorboard/_keras80 ',
                histogram_freq=0,
                write_graph=True,
                write_images=True,
              )


start_time = time.time()
model.fit(x_train,y_train,
          epochs=100,
          batch_size=128,
          verbose=1,
          validation_split =0.2,
          callbacks =[es,tb],
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

# #커맨드창에서 경로 따라 들어간뒤.
# tensorboard --logdir=.


#경로 안따라 간다면?
# tensorboard --logdir=c:/study/_save/tensorboard

# pip install setuptools==81.0.0 <<<82버젼이상은 안나와서 이걸로 다운그레이드 해야함
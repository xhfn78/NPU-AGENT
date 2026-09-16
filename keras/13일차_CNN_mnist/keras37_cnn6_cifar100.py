import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
 
#1데이터 
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape,y_train.shape)#(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# print(np.max(x_train),np.min(x_train))  #255 0
# print(np.max(x_test),np.min(x_test))    #255 0

# print(np.unique(y,return_counts=True))
# exit()
###스케일링 1
x_train = x_train/255.
x_test = x_test/255.

# print(np.max(x_train),np.min(x_train))  #1.0 0.0
# print(np.max(x_test),np.min(x_test))    #1.0 0.0
# exit()

# ###스케일링 2
# x_train = (x_train-127.5)/127.5
# x_test = (x_test- 127.5)/127.5

x_train =x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)
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

# =========================================
# Block 1
# 32x32x3 → 16x16x64
# =========================================
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# =========================================
# Block 2
# 16x16x64 → 8x8x128
# =========================================
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# =========================================
# Block 3
# 8x8x128 → 4x4x256
# =========================================
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# =========================================
# Block 4
# 4x4x256 → 2x2x512
# =========================================
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# =========================================
# Block 5
# 2x2x512 → 1x1x512
# =========================================
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# =========================================
# 분류기
# =========================================
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# CIFAR-100 = 100개 클래스
model.add(Dense(100, activation='softmax'))
# exit()

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
          batch_size=32,
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
loss:  3.166536808013916
acc:  0.21089999377727509
accuracy_score :  0.2109
걸린시간 : 583.76 초

#2
loss:  2.940565586090088
acc:  0.24529999494552612
accuracy_score :  0.2453

#3
accuracy_score :  0.3334
걸린시간 : 899.52 초

#4
accuracy_score :  0.3439
걸린시간 : 1079.55 초

'''
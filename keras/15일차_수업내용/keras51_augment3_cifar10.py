import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# x_train = x_train /225.0
# x_test = x_test /225.0
#############데이터 증폭#########################
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,  #수평 뒤집기,
    # vertical_flip=True,    #수직 뒤집기(상하 반전)
    # width_shift_range=0.1, #평형이동 
    height_shift_range=0.1, #
    # rotation_range=5,  #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,
    # shear_range=0.7, #좌표하나를 고정하고 다른 몇개의 좌표로 이동(한마디로 찌부)
    fill_mode='nearest',
)

augment_size = 40000  
# print(x_train.shape[0])
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False) #60000개중 40000개 랜덤뽑기
# print(randidx)  #[15000  3128 45286 ... 54111 32130 21793]
# print(randidx.shape) #(40000,) 
# print(len(randidx)) #40000
# print(np.min(randidx),np.max(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape,y_augmented.shape) 
#(40000, 28, 28) (40000,) 아직 데이터변형은 되지않음
# exit()
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3,)  #
print(x_augmented.shape) #(40000, 28, 28, 1)


xy_augmented = datagen.flow(
    x_augmented,y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]
 #### 변환 완료 #########
print(x_augmented.shape) # (40000, 28, 28, 1)
print(x_train.shape)
# exit()
x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)

x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))

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
model.add(Conv2D(64,(5,5),input_shape = (32,32,3),padding='same'))  #  (26 ,26,64)
model.add(Conv2D(filters=64, kernel_size=(3,3),activation='relu',padding='same')) #(24,24,32)
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),activation='relu',padding='same')) #(None, 23, 23, 32)
model.add(Conv2D(128,(3,3),activation='relu')) #(None, 22, 22, 16)
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(256,(2,2),activation='relu')) #(None, 22, 22, 16)
# model.add(Dropout(0.2))
# model.add(MaxP
model.add(Conv2D(256,(2,2),activation='relu')) #(None, 20, 20, 16) 4차원데이터에서 마지막 (None, 10) 로 붙이기 위해서 flatten 사용
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
# model.add(Conv2D(8,(2,2),activation='relu')) #(None, 20, 20, 16) 4차원데이터에서 마지막 (None, 10) 로 붙이기 위해서 flatten 사용 
# model.add(Flatten())   #(None, 6400)
model.add(GlobalAveragePooling2D())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu')) #아웃풋 node의 갯수= units

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
  1/313 [..............................] - ETA: 21 43/313 [===>..........................] - ETA: 0s313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.7557
걸린시간 : 1079.85 초



loss:  0.6610863208770752
acc:  0.7864000201225281
313/313 [==============================] - 1s 1ms/step
accuracy_score :  0.7864
걸린시간 : 316.09 초

313/313 [==============================] - 1s 2ms/step - loss: 0.6772 - acc: 0.7974
loss:  0.6771965622901917
acc:  0.7973999977111816
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.7974
걸린시간 : 374.58 초

데이터 증폭후 수치 
loss:  0.9604931473731995
acc:  0.8371000289916992
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.8371
걸린시간 : 2336.0 초



'''